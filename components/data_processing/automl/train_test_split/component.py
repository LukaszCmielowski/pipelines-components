from typing import Dict, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="quay.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9:rhoai-3.2",
)
def train_test_split(
    dataset: dsl.Input[dsl.Dataset],
    sampled_train_dataset: dsl.Output[dsl.Dataset],
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    test_size: float = 0.3,
) -> NamedTuple("outputs", sample_row=str):
    """Splits a tabular dataset into train and test sets and writes them to output artifacts.

    Args:
        dataset: Input CSV dataset to split.
        sampled_train_dataset: Output dataset artifact for the train split.
        sampled_test_dataset: Output dataset artifact for the test split.
        test_size: Proportion of the data to include in the test split.

    Returns:
        sample_row: JSON string representing a sample row from the test set.
    """
    import pandas as pd

    # Split the data
    from sklearn.model_selection import train_test_split

    X_train, X_test = train_test_split(pd.read_csv(dataset.path), test_size=test_size, random_state=42)

    X_train.to_csv(sampled_train_dataset.path, index=False)
    X_test.to_csv(sampled_test_dataset.path, index=False)

    # Dumps to json string to avoid NaN in the output json
    # Format: '[{"col1": "val1","col2":"val2"},{"col1":"val3","col2":"val4"}]'
    sample_row = X_test.head(1).to_json(orient="records")
    return NamedTuple("outputs", sample_row=Dict)(sample_row=sample_row)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        train_test_split,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
