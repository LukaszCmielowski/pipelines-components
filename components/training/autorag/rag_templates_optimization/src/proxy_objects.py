"""
This module contains proxy classes for respective classes from `ai4rag` module.
The proxies defined here exist so to ease the local execution, debugging or unit/integration-testing by
allowing mocked runs of `ai4rag` code without an external llama-stack server setup.
"""

from typing import Sequence

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.core.experiment.results import EvaluationData, EvaluationResult, ExperimentResults
from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel


class StdoutEventHandler(BaseEventHandler):

    def __init__(self, event_handler: BaseEventHandler) -> None:
        self.event_handler = event_handler
        super().__init__()

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        pass

    def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
        pass


class DisconnectedAI4RAGExperiment(AI4RAGExperiment):

    def __init__(self, rag_experiment: AI4RAGExperiment) -> None:
        self.rag_experiment = rag_experiment
        self.metrics = ["faithfulness"]
        # self.metrics = rag_experiment.metrics

    def search(self, **kwargs) -> Sequence[EvaluationResult]:

        # set mocked self.results and return

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
                        "faithfulnesss": {
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
                EvaluationData(
                    question="benchmark_data.questions[idx]2",
                    answer='inference_response[idx]["answer"]2',
                    contexts="contexts2",
                    context_ids="context_ids2",
                    ground_truths="benchmark_data.answers[idx]2",
                    question_id="benchmark_data.questions_ids[idx]2",
                    ground_truths_context_ids=None,
                ),
            ]

            self.results.add_evaluation(eval_data, eval_res)

        self.search_output = self.results.get_best_evaluations(k=1)

        return self.search_output


class DisconnectedModelsPreSelector(ModelsPreSelector):

    def __init__(self, mps: ModelsPreSelector) -> None:
        self.mps: ModelsPreSelector = mps
        self.metric = mps.metric

    def evaluate_patterns(self):

        self.evaluation_results = [
            {
                "embedding_model": "granite_emb1",
                "foundation_model": "mistral1",
                "scores": {"faithfulness": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}},
                "question_scores": {
                    "faithfulnesss": {
                        "q_id_0": 0.5,
                        "q_id_1": 0.8,
                    }
                },
            },
            {
                "embedding_model": "granite_emb2",
                "foundation_model": "mistral2",
                "scores": {"faithfulness": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6}},
                "question_scores": {
                    "faithfulnesss": {
                        "q_id_0": 0.5,
                        "q_id_1": 0.8,
                    }
                },
            },
        ]

    # def select_models(self, n_em: int = 2, n_fm: int = 3) -> dict[str, list[EmbeddingModel | FoundationModel]]:
    #     pass
