from typing import Dict, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="quay.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9:rhoai-3.2",
)
def train_test_split(  # noqa: D417
    # Add your component parameters here
    dataset: dsl.Input[dsl.Dataset],
    sampled_train_dataset: dsl.Output[dsl.Dataset],
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    test_size: float = 0.3,
    # Add your output artifacts here
    # output_artifact: dsl.Output[dsl.Artifact]
) -> NamedTuple("outputs", sample_row=str):
    """Train Test Split component.

    TODO: Add a detailed description of what this component does.

    Args:
        input_param: Description of the component parameter.
        # Add descriptions for other parameters

    Returns:
        Description of what the component returns.
    """
    import json

    import pandas as pd

    # Split the data
    from sklearn.model_selection import train_test_split

    X_train, X_test = train_test_split(pd.read_csv(dataset.path), test_size=test_size, random_state=42)

    X_train.to_csv(sampled_train_dataset.path, index=False)
    X_test.to_csv(sampled_test_dataset.path, index=False)

    # Dumps to json string to avoid NaN in the output Dict
    sample_row = json.dumps(X_test.head(1).to_dict())
    return NamedTuple("outputs", sample_row=Dict)(sample_row=sample_row)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        train_test_split,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
