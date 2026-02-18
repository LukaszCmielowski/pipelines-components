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
                            "answer_correctness": {"q_id_0": 0.0, "q_id_1": 0.0, "q_id_2": 0.0146},
                            "faithfulness": {"q_id_0": 0.0909, "q_id_1": 0.1818, "q_id_2": 0.1818},
                            "context_correctness": {"q_id_0": 0.0, "q_id_1": 0.2, "q_id_2": 0.0},
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
                            "*  asset_name_or_item: (Required) Either a string with the name of a stored data asset or an item like those returned by list_stored_data().",
                            "Model architecture   The architecture of the model influences how the model behaves.",
                            "Learn more \n\nParent topic:[Governing assets in AI use cases]",
                        ],
                        context_ids=[
                            "0ECEAC44DA213D067B5B5EA66694E6283457A441_9.txt",
                            "120CAE8361AE4E0B6FE4D6F0D32EEE9517F11190_1.txt",
                            "391DBD504569F02CCC48B181E3B953198C8F3C8A_8.txt",
                        ],
                        ground_truths=[
                            "The following models are available in watsonx.ai: \nflan-t5-xl-3b\nFlan-t5-xxl-11b\nflan-ul2-20b\ngpt-neox-20b\ngranite-13b-chat-v2\ngranite-13b-chat-v1\ngranite-13b-instruct-v2\ngranite-13b-instruct-v1\nllama-2-13b-chat\nllama-2-70b-chat\nmpt-7b-instruct2\nmt0-xxl-13b\nstarcoder-15.5b",
                        ],
                        question_id="q_id_0",
                        ground_truths_context_ids=None,
                    ),
                    EvaluationData(
                        question="What foundation models are available on Watsonx, and which of these has IBM built?",
                        answer="I cannot answer this question, because I am just a mocked model.",
                        contexts=[
                            "Retrieval-augmented generation \n\nYou can use foundation models in IBM watsonx.ai to generate factually accurate output.",
                            "Methods for tuning foundation models \n\nLearn more about different tuning methods and how they work.",
                            "Foundation models built by IBM \n\nIn IBM watsonx.ai, you can use IBM foundation models that are built with integrity and designed for business.",
                        ],
                        context_ids=[
                            "752D982C2F694FFEE2A312CEA6ADF22C2384D4B2_0.txt",
                            "15A014C514B00FF78C689585F393E21BAE922DB2_0.txt",
                            "B2593108FA446C4B4B0EF5ADC2CD5D9585B0B63C_0.txt",
                        ],
                        ground_truths=[
                            "The following foundation models are available on Watsonx:\n\n1. flan-t5-xl-3b\n2. flan-t5-xxl-11b\n3. flan-ul2-20b\n4. gpt-neox-20b\n5. granite-13b-chat-v2 (IBM built)\n6. granite-13b-chat-v1 (IBM built)\n7. granite-13b-instruct-v2 (IBM built)\n8. granite-13b-instruct-v1 (IBM built)\n9. llama-2-13b-chat\n10. llama-2-70b-chat\n11. mpt-7b-instruct2\n12. mt0-xxl-13b\n13. starcoder-15.5b\n\n The Granite family of foundation models, including granite-13b-chat-v2, granite-13b-chat-v1, and granite-13b-instruct-v2 has been build by IBM.",
                        ],
                        question_id="q_id_1",
                        ground_truths_context_ids=None,
                    ),
                    EvaluationData(
                        question="How can I ensure that the generated answers will be accurate, factual and based on my information?",
                        answer="I cannot answer this question, because I am just a mocked model.",
                        contexts=[
                            "Functions used in Watson Pipelines's Expression Builder \n\nUse these functions in Pipelines code editors.",
                            "Table 1. Supported values, defaults, and usage notes for sampling decoding\n\n Parameter        Supported values                                                                                 Default  Use",
                            "applygmm properties \n\nYou can use the Gaussian Mixture node to generate a Gaussian Mixture model nugget.",
                        ],
                        context_ids=[
                            "E933C12C1DF97E13CBA40BCD54E4F4B8133DA10C_0.txt",
                            "42AE491240EF740E6A8C5CF32B817E606F554E49_1.txt",
                            "F2D3C76D5EABBBF72A0314F29374527C8339591A_0.txt",
                        ],
                        ground_truths=[
                            "To ensure a language model provides the most accurate and factual answers to questions based on your data, you can follow these steps:\n1. Utilize Retrieval-augmented generation pattern. In this pattaern, you provide the relevant facts from your dataset as context in your prompt text. This will guide the model to generate responses grounded in the provided data\n2. Prompt Engineering: Experiment with prompt engineering techniques to shape the model's output. Understand the capabilities and limitations of the foundation model by fine-tuning prompts and adjusting inputs to align with the desired output. This process helps in refining the generated responses for accuracy.\n3. Review and Validate Output: Regularly review the generated output for biased, inappropriate, or incorrect content. Third-party models may produce outputs containing misinformation, offensive language, or biased content. Implement mechanisms to evaluate and validate the accuracy of the model's responses, ensuring alignment with factual information from your dataset.\n",
                        ],
                        question_id="q_id_2",
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
        """Build evaluation_results.json-style list when question_scores missing or incomplete."""
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
                "answer": getattr(ev, "answer", ""),
                "answer_contexts": answer_contexts,
                "scores": scores,
            })
        return out

    rag_patterns_dir = Path(rag_patterns)
    evaluation_data_list = getattr(rag_exp.results, "evaluation_data", [])

    def _build_pattern_json(evaluation_result, iteration: int, max_combinations: int) -> dict:
        """Build pattern.json content as flat schema: name, iteration, max_combinations, duration_seconds, settings, scores, final_score."""
        idx = evaluation_result.indexing_params or {}
        rp = evaluation_result.rag_params or {}
        chunking = idx.get("chunking") or {}
        embeddings = rp.get("embeddings") or rp.get("embedding") or {}
        retrieval = rp.get("retrieval") or {}
        generation = rp.get("generation") or {}
        return {
            "name": getattr(evaluation_result, "pattern_name", ""),
            "iteration": iteration,
            "max_combinations": max_combinations,
            "duration_seconds": getattr(evaluation_result, "execution_time", 0) or 0,
            "settings": {
                "vector_store": {
                    "datasource_type": idx.get("vector_store", {}).get("datasource_type")
                    or rp.get("vector_store", {}).get("datasource_type")
                    or "ls_milvus",
                    "collection_name": getattr(evaluation_result, "collection", "") or "",
                },
                "chunking": {
                    "method": chunking.get("method", "recursive"),
                    "chunk_size": chunking.get("chunk_size", 2048),
                    "chunk_overlap": chunking.get("chunk_overlap", 256),
                },
                "embedding": {
                    "model_id": embeddings.get("model_id", ""),
                    "distance_metric": embeddings.get("distance_metric", "cosine"),
                },
                "retrieval": {
                    "method": retrieval.get("method", "simple"),
                    "number_of_chunks": retrieval.get("number_of_chunks", 5),
                },
                "generation": {
                    "model_id": generation.get("model_id", ""),
                    "context_template_text": generation.get(
                        "context_template_text", "{document}"
                    ),
                    "user_message_text": generation.get(
                        "user_message_text",
                        "\n\nContext:\n{reference_documents}:\n\nQuestion: {question}. \nAgain, please answer the question based on the context provided only. If the context is not related to the question, just say you cannot answer. Respond exclusively in the language of the question, regardless of any other language used in the provided context. Ensure that your entire response is in the same language as the question.",
                    ),
                    "system_message_text": generation.get(
                        "system_message_text",
                        "Please answer the question I provide in the Question section below, based solely on the information I provide in the Context section. If the question is unanswerable, please say you cannot answer.",
                    ),
                },
            },
        }

    evaluations_list = list(rag_exp.results.evaluations)
    max_combinations = getattr(
        rag_exp.results, "max_combinations", len(evaluations_list)
    ) or 24

    for i, eval in enumerate(evaluations_list):
        patt_dir = rag_patterns_dir / eval.pattern_name
        patt_dir.mkdir(parents=True, exist_ok=True)

        pattern_data = _build_pattern_json(eval, iteration=i, max_combinations=max_combinations)
        # Flat schema: scores = per-metric aggregates (mean, ci_low, ci_high); final_score
        pattern_data["scores"] = (getattr(eval, "scores", None) or {}).get("scores") or {}
        pattern_data["final_score"] = getattr(eval, "final_score", None)
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
        with (patt_dir / "evaluation_results.json").open("w+", encoding="utf-8") as f:
            json_dump(evaluation_result_list, f, indent=2)

    # TODO autorag_run_artifact


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
