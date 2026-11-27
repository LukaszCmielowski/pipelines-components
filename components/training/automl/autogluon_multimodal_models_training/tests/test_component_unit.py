"""Verify the artifact contract with an isolated predictor double."""

import importlib.metadata
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from kfp import dsl

from ..component import autogluon_multimodal_models_training


def test_trains_and_scores_only_the_expected_splits(tmp_path, monkeypatch):
    """Fit on train and tuning data, then score the reloaded model on holdout."""
    pytest.importorskip("pandas")

    class Predictor:
        class_labels = ["bag", "shoe"]

        def __init__(self, **settings):
            assert settings["problem_type"] == "binary"
            assert settings["label"] == "label"

        def fit(self, *, train_data, tuning_data, standalone, hyperparameters, **kwargs):
            assert list(train_data["image"].map(Path).map(lambda path: path.name)) == ["train.png"]
            assert list(tuning_data["image"].map(Path).map(lambda path: path.name)) == ["tune.png"]
            assert hyperparameters["model.names"] == ["timm_image"]
            assert hyperparameters["env.accelerator"] == "cpu"
            assert hyperparameters["env.num_gpus"] == 0
            assert standalone is True

        def save(self, path, *, standalone):
            assert standalone is True
            Path(path).mkdir()

        @classmethod
        def load(cls, path, *, verbosity):
            assert Path(path).is_dir()
            return cls(problem_type="binary", label="label")

        def set_num_gpus(self, count):
            assert count == 0

        def predict(self, data, *, as_pandas):
            assert list(data.columns) == ["image"]
            assert list(data["image"].map(Path).map(lambda path: path.name)) == ["bag.png", "shoe.png"]
            return ["bag", "bag"]

    automm = ModuleType("autogluon.multimodal")
    automm.MultiModalPredictor = Predictor
    monkeypatch.setitem(sys.modules, "autogluon", ModuleType("autogluon"))
    monkeypatch.setitem(sys.modules, "autogluon.multimodal", automm)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="2.8.0"))
    monkeypatch.setattr(importlib.metadata, "version", {"autogluon.multimodal": "1.5.0", "timm": "1.0.3"}.__getitem__)

    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": [{"image": "train.png", "label": "bag"}],
                    "tuning": [{"image": "tune.png", "label": "shoe"}],
                    "holdout": [{"image": "bag.png", "label": "bag"}, {"image": "shoe.png", "label": "shoe"}],
                },
                "data_quality": {"seed": 42},
                "task_type": "binary",
            }
        ),
        encoding="utf-8",
    )
    model = dsl.Model(uri=str(tmp_path / "model"))
    metrics = dsl.Metrics(uri=str(tmp_path / "metrics"))
    result = autogluon_multimodal_models_training.python_func(str(split_path), model, metrics, use_gpu=False)

    assert result == model.uri
    metadata = json.loads((Path(model.path) / "model.json").read_text())
    assert metadata["predictor_path"] == "predictor"
    assert metadata["class_order"] == ["bag", "shoe"]
    confusion = json.loads((Path(model.path) / "metrics" / "confusion_matrix.json").read_text())
    assert confusion["matrix"] == [[1, 0], [1, 0]]
    assert metrics.metadata == {"holdout_accuracy": 0.5, "holdout_balanced_accuracy": 0.5}
