"""Train and package one AutoMM image classifier."""

from kfp import dsl


@dsl.component(base_image="python:3.12-slim", install_kfp_package=False)
def autogluon_multimodal_models_training(
    split_manifest_path: str,
    model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
    use_gpu: bool = True,
) -> str:
    """Train one ResNet-18 AutoMM predictor and evaluate the exported model on the holdout.

    Args:
        split_manifest_path: Staged dataset split manifest in the run workspace.
        model: Standalone predictor, versioned model metadata, and evaluation JSON files.
        metrics: Holdout metric scalars for the pipeline run.
        use_gpu: Require one GPU. Disable only for development smoke tests.

    Returns:
        URI of the exported model directory.
    """
    import json
    from importlib.metadata import version
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from time import monotonic

    import pandas as pd
    import torch
    from autogluon.multimodal import MultiModalPredictor

    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("The prototype requires a CUDA GPU unless compiled for CPU development.")

    split_path = Path(split_manifest_path)
    staged = json.loads(split_path.read_text(encoding="utf-8"))

    def frame(split):
        return pd.DataFrame(
            [{"image": str(split_path.parent / row["image"]), "label": row["label"]} for row in staged["splits"][split]]
        )

    backbone = "resnet18"
    time_limit_seconds = 600
    seed = staged["data_quality"]["seed"]
    model_dir = Path(model.path)
    model_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = model_dir / "predictor"
    with TemporaryDirectory(dir=split_path.parent) as scratch:
        predictor = MultiModalPredictor(
            label="label",
            problem_type=staged["task_type"],
            eval_metric="accuracy",
            path=str(Path(scratch) / "fit"),
            verbosity=0,
        )
        started = monotonic()
        predictor.fit(
            train_data=frame("train"),
            tuning_data=frame("tuning"),
            time_limit=time_limit_seconds,
            seed=seed,
            column_types={"image": "image_path"},
            hyperparameters={
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": backbone,
                "env.accelerator": "auto" if use_gpu else "cpu",
                "env.num_gpus": int(use_gpu),
                "env.num_workers": 0,
                "env.num_workers_inference": 0,
            },
            standalone=True,
        )
        fit_seconds = round(monotonic() - started, 3)
        predictor.save(str(predictor_path), standalone=True)
        del predictor

    predictor = MultiModalPredictor.load(str(predictor_path), verbosity=0)
    predictor.set_num_gpus(int(use_gpu))
    holdout = frame("holdout")
    try:
        predictions = predictor.predict(holdout[["image"]], as_pandas=False)
    except Exception:
        raise RuntimeError("The saved predictor could not score the held-out images.") from None
    labels = [str(label) for label in predictor.class_labels]
    index = {label: pos for pos, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for actual, predicted in zip(holdout["label"], predictions, strict=True):
        matrix[index[str(actual)]][index[str(predicted)]] += 1
    correct = [row[i] for i, row in enumerate(matrix)]
    scores = {
        "accuracy": sum(correct) / len(holdout),
        "balanced_accuracy": sum(count / sum(row) for count, row in zip(correct, matrix, strict=True)) / len(labels),
    }

    def write_json(path, data):
        path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    results_dir = model_dir / "metrics"
    results_dir.mkdir(exist_ok=True)
    write_json(results_dir / "metrics.json", {"split": "holdout", "row_count": len(holdout), **scores})
    write_json(results_dir / "confusion_matrix.json", {"class_order": labels, "matrix": matrix})
    write_json(
        results_dir / "training_summary.json", {"fit_seconds": fit_seconds, "device": "gpu" if use_gpu else "cpu"}
    )
    write_json(
        model_dir / "model.json",
        {
            "schema_version": 1,
            "predictor_type": "autogluon.multimodal.MultiModalPredictor",
            "predictor_path": "predictor",
            "autogluon_version": version("autogluon.multimodal"),
            "torch_version": torch.__version__,
            "timm_version": version("timm"),
            "backbone_id": backbone,
            "task_family": "image_prediction",
            "task_type": staged["task_type"],
            "class_order": labels,
            "input_schema": {"image": {"type": "image", "column": "image", "formats": ["jpeg", "png"]}},
            "eval_metric": {"name": "accuracy", "higher_is_better": True, "split": "holdout"},
            "data": staged["data_quality"],
            "training": {"seed": seed, "time_limit_seconds": time_limit_seconds},
        },
    )
    for name, score in scores.items():
        metrics.log_metric(f"holdout_{name}", score)
    return model.uri
