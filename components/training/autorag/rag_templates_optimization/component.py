from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
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
            embedding_models = ["mock-embed-a", "mock-embed-b", "mock-embed-a"]
            generation_models = ["mock-llm-1", "mock-llm-1", "mock-llm-2"]
            for i in range(3):
                indexing_params = {
                    "chunking": {
                        "method": "recursive",
                        "chunk_size": 256,
                        "chunk_overlap": 128,
                    },
                }
                rag_params = {
                    "embeddings": {"model_id": embedding_models[i]},
                    "retrieval": {"method": "window", "number_of_chunks": 5},
                    "generation": {"model_id": generation_models[i]},
                }
                eval_res = EvaluationResult(
                    f"pattern{i}",
                    f"collection{i}",
                    indexing_params,
                    rag_params,
                    scores={
                        "scores": {
                            "answer_correctness": {"mean": 0.5 + 0.1 * i, "ci_low": 0.4, "ci_high": 0.7},
                            "faithfulness": {"mean": 0.2 + 0.1 * i, "ci_low": 0.1, "ci_high": 0.5},
                            "context_correctness": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.0},
                        },
                        "question_scores": {
                            "answer_correctness": {"q_id_0": 0.5, "q_id_1": 0.8},
                            "faithfulness": {"q_id_0": 0.5, "q_id_1": 0.8},
                            "context_correctness": {"q_id_0": 1.0, "q_id_1": 1.0},
                        },
                    },
                    execution_time=0.5 * i,
                    final_score=0.5 + 0.1 * i,
                )
                eval_data = [
                    EvaluationData(
                        question="What foundation models are available in watsonx.ai?",
                        answer="I cannot answer this question, because I am just a mocked model.",
                        contexts=[
                            "Model architecture: encoder-only, decoder-only, encoder-decoder.",
                            "Regional availability: same IBM Cloud regional data center as watsonx.",
                        ],
                        context_ids=[
                            "120CAE8361AE4E0B6FE4D6F0D32EEE9517F11190_1.txt",
                            "391DBD504569F02CCC48B181E3B953198C8F3C8A_8.txt",
                        ],
                        ground_truths=["flan-t5-xl-3b", "granite-13b-chat-v2", "llama-2-70b-chat"],
                        question_id="q_id_0",
                        ground_truths_context_ids=None,
                    ),
                    EvaluationData(
                        question="What is the difference between fine-tuning and prompt-tuning?",
                        answer="I cannot answer this question, because I am just a mocked model.",
                        contexts=[
                            "Fine-tuning: changes the parameters of the underlying foundation model.",
                            "Prompt-tuning: adjusts the content of the prompt; model parameters not edited.",
                        ],
                        context_ids=[
                            "15A014C514B00FF78C689585F393E21BAE922DB2_0.txt",
                            "B2593108FA446C4B4B0EF5ADC2CD5D9585B0B63C_0.txt",
                        ],
                        ground_truths=[
                            "Fine-tuning changes model parameters; prompt-tuning only alters the prompt input.",
                        ],
                        question_id="q_id_1",
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

    def _evaluation_result_fallback(eval_data_list, evaluation_result):
        """Build evaluation_result.json-style list when question_scores missing or incomplete."""
        out = []
        for ev in eval_data_list:
            answer_contexts = []
            if getattr(ev, "contexts", None) and getattr(ev, "context_ids", None):
                answer_contexts = [
                    {"text": t, "document_id": doc_id}
                    for t, doc_id in zip(ev.contexts, ev.context_ids)
                ]
            scores = {}
            q_scores = (evaluation_result.scores or {}).get("question_scores") or {}
            for key in q_scores:
                if isinstance(q_scores[key], dict) and getattr(ev, "question_id", None) in q_scores[key]:
                    scores[key] = q_scores[key][ev.question_id]
            out.append({
                "question": getattr(ev, "question", ""),
                "correct_answers": getattr(ev, "ground_truths", None),
                "question_id": getattr(ev, "question_id", ""),
                "answer": getattr(ev, "answer", ""),
                "answer_contexts": answer_contexts,
                "scores": scores,
            })
        return out

    rag_patterns_dir = Path(rag_patterns)
    evaluation_data_list = getattr(rag_exp.results, "evaluation_data", [])

    for i, eval in enumerate(rag_exp.results.evaluations):
        patt_dir = rag_patterns_dir / eval.pattern_name
        patt_dir.mkdir(parents=True, exist_ok=True)

        # pattern.json: ai4rag EvaluationResult (to_dict) + schema fields for consumer compliance
        pattern_data = {
            "pattern_name": eval.pattern_name,
            "collection": eval.collection,
            "scores": eval.scores,
            "execution_time": eval.execution_time,
            "final_score": eval.final_score,
            "schema_version": "1.0",
            "producer": "ai4rag",
            "settings": {**(eval.indexing_params or {}), **(eval.rag_params or {})},
        }
        with (patt_dir / "pattern.json").open("w+", encoding="utf-8") as pattern_details:
            json_dump(pattern_data, pattern_details, indent=2)

        with (patt_dir / "inference_notebook.ipynb").open("w+") as inf_notebook:
            json_dump({"inference_notebook_cell": "cell_value"}, inf_notebook)

        with (patt_dir / "indexing_notebook.ipynb").open("w+") as ind_notebook:
            json_dump({"ind_notebook_cell": "cell_Value"}, ind_notebook)

        eval_data = evaluation_data_list[i] if i < len(evaluation_data_list) else []
        try:
            q_scores = (eval.scores or {}).get("question_scores") or {}
            if q_scores and all(
                isinstance(q_scores.get(k), dict) for k in q_scores
            ):
                evaluation_result_list = ExperimentResults.create_evaluation_results_json(
                    eval_data, eval
                )
            else:
                evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        except (KeyError, TypeError):
            evaluation_result_list = _evaluation_result_fallback(eval_data, eval)
        with (patt_dir / "evaluation_result.json").open("w+", encoding="utf-8") as f:
            json_dump(evaluation_result_list, f, indent=2)

    # TODO autorag_run_artifact


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
