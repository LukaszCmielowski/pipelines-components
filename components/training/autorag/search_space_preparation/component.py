import os
from pathlib import Path
from random import random
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml as yml
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.rag.embedding.base_model import EmbeddingModel
from ai4rag.rag.foundation_models.base_model import FoundationModel
from ai4rag.search_space.prepare.prepare_search_space import prepare_ai4rag_search_space
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from kfp import dsl
from langchain_core.documents import Document

from components.training.autorag.rag_templates_optimization.src.utils import load_as_langchain_doc
from components.training.autorag.search_space_preparation.proxy_objects import DisconnectedModelsPreSelector


@dsl.component(base_image="quay.io/fkomarzy/sandbox_public/rag_base:1.0b1")
def search_space_preparation(
    test_data: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Input[dsl.Artifact],
    search_space_prep_report: dsl.Output[dsl.Artifact],
    # test_data: dsl.InputPath(dsl.Artifact),
    # extracted_text: dsl.InputPath(dsl.Artifact),
    # search_space_prep_report: dsl.OutputPath(dsl.Artifact),
    # constraints: Dict = None,
    embeddings_models: Optional[List] = None,
    generation_models: Optional[List] = None,
    models_config: Dict = None,  # ???
    metric: str = None,
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

    import os
    from pathlib import Path
    from random import random
    from typing import Any, Optional

    import pandas as pd
    import yaml as yml
    from ai4rag.core.experiment.benchmark_data import BenchmarkData
    from ai4rag.core.experiment.mps import ModelsPreSelector
    from ai4rag.rag.embedding.base_model import EmbeddingModel
    from ai4rag.rag.foundation_models.base_model import FoundationModel
    from ai4rag.search_space.prepare.prepare_search_space import prepare_ai4rag_search_space
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from kfp import dsl
    from langchain_core.documents import Document
    from proxy_objects import DisconnectedModelsPreSelector
    from utils import load_as_langchain_doc

    # TODO whole component has to be run conditionally
    # TODO these defaults should be exposed by ai4rag library
    TOP_N_GENERATION_MODELS = 3  # change names (topNmodels? )
    TOP_K_EMBEDDING_MODELS = 2
    METRIC = "faithfulness"
    SAMPLE_SIZE = 5
    SEED = 17

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

    def prepare_ai4rag_search_space():
        if not generation_models:
            generation_models_local = ["mocked_gen_model1"]
        if not embeddings_models:
            embedding_models_local = ["mocked_em_model1"]

        generation_models_local = list(map(MockGenerationModel, generation_models))
        embedding_models_local = list(map(MockEmbeddingModel, embeddings_models))

        return AI4RAGSearchSpace(
            params=[
                Parameter("foundation_model", "C", values=generation_models_local),
                Parameter("embedding_model", "C", values=embedding_models_local),
            ]
        )

    # build search space
    # constraints = constraints if constraints else {}
    search_space = prepare_ai4rag_search_space()

    benchmark_data = BenchmarkData(pd.read_json(Path(test_data.path)))
    documents = load_as_langchain_doc(extracted_text.path)

    mps = ModelsPreSelector(
        benchmark_data=benchmark_data.get_random_sample(n_records=SAMPLE_SIZE, random_seed=SEED),
        documents=documents,
        foundation_models=search_space._search_space["foundation_model"].values,
        embedding_models=search_space._search_space["embedding_model"].values,
        metric=metric if metric else METRIC,
    )

    if not os.environ.get("LLAMASTACK_CLIENT_CONNECTION", None):
        # TODO the exact env variable name is yet to be defined
        mps = DisconnectedModelsPreSelector(mps)

    mps.evaluate_patterns()

    selected_models = mps.select_models(n_em=TOP_K_EMBEDDING_MODELS, n_fm=TOP_N_GENERATION_MODELS)
    selected_models_names = {k: list(map(str, v)) for k, v in selected_models.items()}

    verbose_search_space_repr = {
        k: v.all_values()
        for k, v in search_space._search_space.items()
        if k not in ("foundation_model", "embedding_model")
    }
    verbose_search_space_repr |= selected_models_names

    with open(search_space_prep_report.path, "w") as report_file:
        yml.dump(verbose_search_space_repr, report_file, yml.SafeDumper)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
