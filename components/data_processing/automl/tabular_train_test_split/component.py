from typing import Dict, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
)
def tabular_train_test_split(  # noqa: D417
    dataset: dsl.Input[dsl.Dataset],
    sampled_train_dataset: dsl.Output[dsl.Dataset],
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    test_size: float = 0.3,
) -> NamedTuple("outputs", sample_row=str, split_config=dict):
    """Splits a tabular dataset into train and test sets and writes them to output artifacts.

    Args:
        dataset: Input CSV dataset to split.
        sampled_train_dataset: Output dataset artifact for the train split.
        sampled_test_dataset: Output dataset artifact for the test split.
        test_size: Proportion of the data to include in the test split.

    Returns:
        NamedTuple: Contains a sample row and a split configuration dictionary.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Set constants
    DEFAULT_RANDOM_STATE = 42

    sampled_train_dataset.uri += ".csv"
    sampled_test_dataset.uri += ".csv"

    # Split the data
    X_train, X_test = train_test_split(
        pd.read_csv(dataset.path), test_size=test_size, random_state=DEFAULT_RANDOM_STATE
    )

    X_train.to_csv(sampled_train_dataset.path, index=False)
    X_test.to_csv(sampled_test_dataset.path, index=False)

    # Dumps to json string to avoid NaN in the output json
    # Format: '[{"col1": "val1","col2":"val2"},{"col1":"val3","col2":"val4"}]'
    sample_row = X_test.head(1).to_json(orient="records")
    return NamedTuple("outputs", sample_row=Dict, split_config=dict)(
        sample_row=sample_row, split_config={"test_size": test_size}
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        tabular_train_test_split,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
