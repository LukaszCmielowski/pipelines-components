from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="quay.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9:rhoai-3.2",
    packages_to_install=[
        "ai4rag@git+https://github.com/IBM/ai4rag.git",
        "pyyaml",
        "pysqlite3-binary",  # ChromaDB requires sqlite3 >= 3.35; base image has older sqlite
    ],
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
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
            Additional settings customising the experiment. May include "metric" (str): quality
            metric for optimization. Supported values: "faithfulness", "answer_correctness",
            "context_correctness". Defaults to "faithfulness" if omitted.

    Returns:
        rag_patterns
            A path pointing to a folder containing all of the generated RAG patterns
            (each subdir: pattern.json, indexing_notebook.ipynb, inference_notebook.ipynb).
        autorag_run_artifact
            Run log and experiment status (TODO).

    """
    # ChromaDB (via ai4rag) requires sqlite3 >= 3.35; RHEL9 base image has older sqlite.
    # Patch stdlib sqlite3 with pysqlite3-binary before any ai4rag import.
    import sys

    try:
        import pysqlite3

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass

    import os
    from json import dump as json_dump
    from pathlib import Path
    from random import random
    from typing import Any, List, TextIO

    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.experiment import AI4RAGExperiment
    from ai4rag.core.experiment.results import EvaluationData, EvaluationResult, ExperimentResults
    from ai4rag.core.hpo.base_optimiser import OptimiserSettings
    from ai4rag.rag.embedding.base_model import EmbeddingModel
    from ai4rag.rag.foundation_models.base_model import FoundationModel
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel

    from langchain_core.documents import Document

    MAX_NUMBER_OF_RAG_PATTERNS = 8
    METRIC = "faithfulness"
    SUPPORTED_OPTIMIZATION_METRICS = frozenset({"faithfulness", "answer_correctness", "context_correctness"})

    class TmpEventHandler(BaseEventHandler):
        """Exists temporarily only for the purpose of satisying type hinting checks"""

        def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
            pass

        def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
            pass

    class DisconnectedAI4RAGExperiment(AI4RAGExperiment):
        """Mock experiment that returns fake results when no llama-stack client is configured."""

        def __init__(self, rag_experiment: AI4RAGExperiment) -> None:
            self.rag_experiment = rag_experiment
            self.metrics = ["faithfulness"]

        def search(self, **kwargs):
            self.results = ExperimentResults()
            for i in range(3):
                eval_res = EvaluationResult(
                    f"pattern{i}",
                    f"collection{i}",
                    {"indexing_param_key": f"indexing_val{i}"},
                    {"rag_param_key": f"rag_param_val{i}"},
                    scores={
                        "scores": {"faithfulness": {"mean": 0.1 * i, "ci_low": 0.4, "ci_high": 0.6}},
                        "question_scores": {
                            "faithfulness": {
                                "q_id_0": 0.5,
                                "q_id_1": 0.8,
                            }
                        },
                    },
                    execution_time=0.5 * i,
                    final_score=0.1 * i,
                )
                eval_data = [
                    EvaluationData(
                        question="benchmark_data.questions[idx]",
                        answer='inference_response[idx]["answer"]',
                        contexts="contexts",
                        context_ids="context_ids",
                        ground_truths="benchmark_data.answers[idx]",
                        question_id="benchmark_data.questions_ids[idx]",
                        ground_truths_context_ids=None,
                    ),
                ]
                self.results.add_evaluation(eval_data, eval_res)
            return self.results.get_best_evaluations(k=1)

    def load_as_langchain_doc(path: str | Path) -> list[Document]:
        """
        Given path to a text-based file or a folder thereof load everything to memory and
        return as a list of langchain `Document` objects.

        Args:
            path
                A local path to either a text file or a folder of text files.
        Returns:
            A list of langchain `Document` objects.

        Note:

        """

        if isinstance(path, str):
            path = Path(path)

        documents = []
        if path.is_dir():
            for doc_path in path.iterdir():
                with doc_path.open("r", encoding="utf-8") as doc:
                    documents.append(Document(page_content=doc.read(), metadata={"file_name": doc_path.name}))

        elif path.is_file():
            with path.open("r", encoding="utf-8") as doc:
                documents.append(Document(page_content=doc.read(), metadata={"file_name": path.name}))

        return documents

    class MockGenerationModel(FoundationModel):
        def __init__(
            self,
            model_id: str,
            client: None = None,
            model_params: dict[str, Any] | None = None,
        ):
            super().__init__(client, model_id, model_params)

        def chat(self, system_message: str, user_message: str) -> str:
            return "Dummy response from a generation model!"

    class MockEmbeddingModel(EmbeddingModel):
        def __init__(self, model_id: str, params: dict[str, Any] | None = None, client: None = None):
            super().__init__(client, model_id, params)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            n = []
            for _ in texts:
                n.append([random() for _ in range(self.params["embedding_dimension"])])

            return n

        def embed_query(self, query: str) -> List[float]:
            return [random() for _ in range(self.params["embedding_dimension"])]

    optimization_settings = optimization_settings if optimization_settings else {}
    if not (optimization_metric := optimization_settings.get("metric", None)):
        optimization_metric = METRIC
    if optimization_metric not in SUPPORTED_OPTIMIZATION_METRICS:
        raise ValueError(
            "optimization_metric must be one of %s; got %r"
            % (sorted(SUPPORTED_OPTIMIZATION_METRICS), optimization_metric)
        )

    documents = load_as_langchain_doc(extracted_text)

    # recreate the search space
    with open(search_space_prep_report, "r") as f:
        search_space = yml.load(f, yml.SafeLoader)
    params = []
    for param, values in search_space.items():
        if param == "foundation_models":
            params.append(Parameter("foundation_model", "C", values=list(map(MockGenerationModel, values))))
        elif param == "embedding_models":
            params.append(Parameter("embedding_model", "C", values=list(map(MockEmbeddingModel, values))))
        else:
            params.append(Parameter(param, "C", values=values))
    search_space = AI4RAGSearchSpace(params=params)

    # TODO chunking should be handled externally
    # OR
    # should be a separate component run previously (or within text extraction)
    # documents_splits = MarkdownTextSplitter().create_documents(documents)

    event_handler = TmpEventHandler()
    optimiser_settings = OptimiserSettings(
        max_evals=optimization_settings.get("max_number_of_rag_patterns", MAX_NUMBER_OF_RAG_PATTERNS)
    )

    benchmark_data = pd.read_json(Path(test_data))

    # TODO exact naming of the secret is yet to be defined
    client_connection = os.environ.get("LLAMASTACK_CLIENT_CONNECTION", None)

    # ai4rag does not accept None for vector_store_type; use a supported default when omitted
    vector_store_type = vector_database_id if vector_database_id else "ls_milvus"

    rag_exp = AI4RAGExperiment(
        client=client_connection,
        event_handler=event_handler,
        optimiser_settings=optimiser_settings,
        search_space=search_space,
        benchmark_data=benchmark_data,
        vector_store_type=vector_store_type,
        documents=documents,
        optimization_metric=optimization_metric,
        # TODO some necessary kwargs (if any at all)
    )

    # When no llama-stack client is configured, use mocked experiment to avoid server dependency
    if not client_connection:
        rag_exp = DisconnectedAI4RAGExperiment(rag_exp)

    # retrieve documents && run optimisation loop
    best_pattern = rag_exp.search()

    rag_patterns_dir = Path(rag_patterns)

    for eval in rag_exp.results.evaluations:
        patt_dir = rag_patterns_dir / eval.pattern_name
        patt_dir.mkdir(parents=True, exist_ok=True)
        with (patt_dir / "pattern.json").open("w+") as pattern_details:
            # json_dump(rag_exp._stream_finished_pattern(eval, []), pattern_details)
            json_dump(eval.to_dict(), pattern_details)

        with (patt_dir / "inference_notebook.ipynb").open("w+") as inf_notebook:
            json_dump({"inference_notebook_cell": "cell_value"}, inf_notebook)

        with (patt_dir / "indexing_notebook.ipynb").open("w+") as ind_notebook:
            json_dump({"ind_notebook_cell": "cell_Value"}, ind_notebook)

    # TODO autorag_run_artifact


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
