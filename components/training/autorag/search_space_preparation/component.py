import os
from typing import Optional

import pandas as pd
import yaml as yml
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.search_space.prepare.prepare_search_space import prepare_ai4rag_search_space
from kfp import dsl

from components.training.autorag.rag_templates_optimization.src.utils import load_as_langchain_doc
from components.training.autorag.search_space_preparation.proxy_objects import DisconnectedModelsPreSelector


@dsl.component(target_image="rag_base:test")
def search_space_preparation(
    test_data: dsl.InputPath(dsl.Artifact),
    extracted_text: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.OutputPath(dsl.Artifact),
    constraints: Optional[dict] = None,
    models_config: Optional[dict] = None,  # ???
    metric: Optional[str] = None,
):
    """
    Runs an AutoRAG experiment's first phase which includes:
    - AutoRAG search space creation given the user's constraints,
    - embedding and foundation models number limitation and initial selection,

    Args:
        test_data
            A path to a .json file containing questions and expected answers that can be retrieved
            from `extracted_data`. Necessary baseline for calculating quality metrics of RAG pipeline.

        extracted_text
            A path to either a single file or a folder of files. The document(s) will be used during
            model selection process.

        constraints
            User defined constraints for the AutoRAG search space.

        models_config
            User defined models limited selection.

        metric
            Quality metric to evaluate the intermediate RAG patterns.

    Returns:
        search_space_prep_report
            A .yml-formatted report including results of this experiment's phase.
            For its exact content please refer to the `search_space_prep_report_schema.yml` file.
    """
    # TODO whole component has to be run conditionally
    # TODO these defaults should be exposed by ai4rag library
    Top_N_FM = 3  # change names (topNmodels? )
    Top_K_EM = 2
    METRIC = "faithfulness"
    SAMPLE_SIZE = 5
    SEED = 17

    # build search space
    constraints = constraints if constraints else {}
    search_space = prepare_ai4rag_search_space(constraints)

    benchmark_data = BenchmarkData(pd.read_json(test_data))
    documents = load_as_langchain_doc(extracted_text)

    mps = ModelsPreSelector(
        benchmark_data=benchmark_data.get_random_sample(n_records=SAMPLE_SIZE, random_seed=SEED),
        documents=documents,
        foundation_models=search_space._search_space["inference_model_id"].values,
        embedding_models=search_space._search_space["embedding_model"].values,
        metric=metric if metric else METRIC,
    )

    if not os.environ.get("LLAMASTACK_CLIENT_CONNECTION", None):
        # TODO the exact env variable name is yet to be defined
        mps = DisconnectedModelsPreSelector(mps)

    mps.evaluate_patterns()

    selected_models = mps.select_models(n_em=Top_K_EM, n_fm=Top_N_FM)

    verbose_search_space_repr = {
        k: v.all_values()
        for k, v in search_space._search_space.items()
        if k not in ("inference_model_id", "embedding_model")
    }
    verbose_search_space_repr |= selected_models

    with open(search_space_prep_report, "w") as report_file:
        yml.dump(verbose_search_space_repr, report_file)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
