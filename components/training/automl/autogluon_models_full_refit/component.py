from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",
    packages_to_install=["autogluon.tabular[all]==1.5.0"],
)
def autogluon_models_full_refit(
    model_name: str,
    full_dataset: dsl.Input[dsl.Dataset],
    predictor_path: str,
    model_artifact: dsl.Output[dsl.Model],
) -> NamedTuple("outputs", model_name=str):
    """Refit a specific AutoGluon model on the full training dataset.

    This component takes a trained AutoGluon TabularPredictor, loaded from
    predictor_path, and refits a specific model, identified by model_name, on
    the complete training dataset. The refitting process retrains the model
    architecture on the full data, typically improving performance compared to
    models trained on sampled data.

    After refitting, the component creates a cleaned clone of the predictor
    containing only the original model and its refitted version (with "_FULL"
    suffix). The refitted model is set as the best model and the predictor
    is optimized to save space by removing unnecessary models and files.
    Evaluation metrics, feature importance, and (for classification) confusion
    matrix are written under model_artifact.path / model_name_FULL / metrics.

    This component is typically used in a two-stage training pipeline where
    models are first trained on sampled data for exploration, then the best
    candidates are refitted on the full dataset for optimal performance.

    Args:
        model_name: The name of the model to refit. This should match a model
            name in the predictor. The refitted model will be saved with the
            suffix "_FULL" appended to this name.
        full_dataset: A Dataset artifact containing the complete training
            dataset in CSV format. This dataset will be used to retrain the
            specified model. The dataset should match the format and schema
            of the data used during initial model training.
        predictor_path: Path (string) to a trained AutoGluon TabularPredictor
            that includes the model specified by model_name. The predictor
            should have been trained previously, potentially on a sampled
            subset of the data.
        model_artifact: Output Model artifact where the refitted predictor
            will be saved. The artifact will contain a cleaned predictor with
            only the original model and its refitted "_FULL" version. Metrics
            are written under model_artifact.path / model_name_FULL / metrics.
            The metadata will include the model_name with "_FULL" suffix.

    Returns:
        None. The refitted model is saved to the model_artifact output.

    Raises:
        FileNotFoundError: If the predictor path or full_dataset path
            cannot be found.
        ValueError: If the predictor cannot be loaded, the model_name is not
            found in the predictor, or the refitting process fails.
        KeyError: If required model files are missing from the predictor.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_models_full_refit import (
            autogluon_models_full_refit
        )

        @dsl.pipeline(name="model-refit-pipeline")
        def refit_pipeline(train_data, predictor_path):
            "Refit the best model on full dataset."
            refitted = autogluon_models_full_refit(
                model_name="LightGBM_BAG_L1",
                full_dataset=train_data,
                predictor_path=predictor_path,
            )
            return refitted

    """
    import json
    import os
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    full_dataset_df = pd.read_csv(full_dataset.path)

    predictor = TabularPredictor.load(predictor_path)

    # save refitted model to output artifact
    model_name_full = model_name + "_FULL"
    path = Path(model_artifact.path) / model_name_full
    model_artifact.metadata["model_name"] = model_name_full
    predictor_clone = predictor.clone(path=path, return_clone=True, dirs_exist_ok=True)
    predictor_clone.delete_models(models_to_keep=[model_name])

    predictor_clone.refit_full(train_data_extra=full_dataset_df, model=model_name)

    predictor_clone.set_model_best(model=model_name_full, save_trainer=True)
    predictor_clone.save_space()

    eval_results = predictor_clone.evaluate(full_dataset_df)
    feature_importance = predictor_clone.feature_importance(full_dataset_df)

    # save evaluation results to output artifact
    os.makedirs(str(path / "metrics"), exist_ok=True)
    with (path / "metrics" / "metrics.json").open("w") as f:
        json.dump(eval_results, f)

    # save feature importance to output artifact
    with (path / "metrics" / "feature_importance.json").open("w") as f:
        json.dump(feature_importance.to_dict(), f)

    # generate confusion matrix for classification problem types
    if predictor.problem_type in {"binary", "multiclass"}:
        from autogluon.core.metrics import confusion_matrix

        confusion_matrix_res = confusion_matrix(
            solution=predictor_clone.predict(full_dataset_df),
            prediction=full_dataset_df[predictor.label],
            output_format="pandas_dataframe",
        )
        with (path / "metrics" / "confusion_matrix.json").open("w") as f:
            json.dump(confusion_matrix_res.to_dict(), f)

    return NamedTuple("outputs", model_name=str)(model_name=model_name_full)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_models_full_refit,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
