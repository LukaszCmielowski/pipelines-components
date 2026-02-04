import os
from json import dump as json_dump
from pathlib import Path
from typing import Optional

import pandas as pd
from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.hpo.base_optimiser import OptimiserSettings
from kfp import dsl
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

from components.training.autorag.rag_templates_optimization.src.event_handlers import TmpEventHandler
from components.training.autorag.rag_templates_optimization.src.proxy_objects import (
    DisconnectedAI4RAGExperiment,
    StdoutEventHandler,
)
from components.training.autorag.rag_templates_optimization.src.utils import (
    load_as_langchain_doc,
    load_search_space_from,
)


@dsl.component(target_image="rag_base:test")
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
    leaderboard: dsl.Output[dsl.Artifact],  # contains metadata on the patterns hierarchy
    rag_patterns: dsl.OutputPath(dsl.Artifact),
    autorag_run_artifact: dsl.Output[dsl.Artifact],  # uri to log, WML status object-like (from WML)
    vector_database_id: Optional[str] = None,
    optimization_settings: Optional[dict] = None,
):
    """
    Rag Templates Optimization component.

    Carries out the iterative RAG optimization process.

    Args:
        extracted_text
            A path pointing to a folder containg extracted texts from input documents.

        test_data
            A path pointing to test data used for evaluating RAG pattern quality.

        search_space_prep_report
            A path pointing to a .yml file containig short report on the experiment's first phase
            (search space preparation).

        vector_database
            An identificator of the vector store used in the experiment.

        optimization_settings
            Additional settings customising the experiment.

    Returns:
        leaderboard
            An artifact containing ordered (metric-wise) RAG patterns along with related metadata.
        rag_patterns
            A path pointing to a folder containg all of the generated RAG patterns.

    """

    MAX_NUMBER_OF_RAG_PATTERNS = 8
    METRIC = "faithfulness"

    optimization_settings = optimization_settings if optimization_settings else {}
    if not (optimization_metric := optimization_settings.get("metric", None)):
        optimization_metric = METRIC

    documents = load_as_langchain_doc(extracted_text)

    # recreate the search space
    search_space = load_search_space_from(search_space_prep_report)

    # TODO chunking should be handled externally
    # OR
    # should be a separate component run previously (or within text extraction)
    # documents_splits = MarkdownTextSplitter().create_documents(documents)

    event_handler = TmpEventHandler()
    optimiser_settings = OptimiserSettings(
        max_evals=optimization_settings.get("max_number_of_rag_patterns", MAX_NUMBER_OF_RAG_PATTERNS)
    )
    benchmark_data = pd.read_json(test_data)

    # TODO exact naming of the secret is yet to be defined
    client_connection = os.environ.get("LLAMASTACK_CLIENT_CONNECTION", None)

    rag_exp = AI4RAGExperiment(
        client=client_connection,
        event_handler=event_handler,
        optimiser_settings=optimiser_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_type=vector_database_id,
        documents=documents,
        optimization_metric=optimization_metric,
        # TODO some necessary kwargs (if any at all)
    )

    if not client_connection:
        rag_exp = DisconnectedAI4RAGExperiment(rag_exp)

    # retrieve documents && run optimisation loop
    best_pattern = rag_exp.search()

    rag_patterns_dir = Path(rag_patterns)

    for eval in rag_exp.results.evaluations:
        patt_dir = rag_patterns_dir / eval.pattern_name
        patt_dir.mkdir()
        with (patt_dir / "pattern.json").open("w") as pattern_details:
            # json_dump(rag_exp._stream_finished_pattern(eval, []), pattern_details)
            json_dump(eval.to_dict(), pattern_details)

        with (patt_dir / "inference_notebook.ipynb").open("w") as inf_notebook:
            json_dump({"inference_notebook_cell": "cell_value"}, inf_notebook)

        with (patt_dir / "indexing_notebook.ipynb").open("w") as ind_notebook:
            json_dump({"ind_notebook_cell": "cell_Value"}, ind_notebook)

    # TODO leaderboard artifact
    # TODO autorag_run_artifact


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
