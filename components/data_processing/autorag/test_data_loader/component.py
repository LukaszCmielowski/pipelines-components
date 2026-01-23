from kfp import dsl


@dsl.component(
    base_image="python:3.11",
    # packages_to_install=["numpy", "pandas"],  # Add your dependencies here
)
def test_data_loader(
    # Add your component parameters here
    input_param: str,
    # Add your output artifacts here
    # output_artifact: dsl.Output[dsl.Artifact]
) -> str:  # Specify your return type
    """Test Data Loader component.

    TODO: Add a detailed description of what this component does.

    Args:
        input_param: Description of the component parameter.
        # Add descriptions for other parameters

    Returns:
        Description of what the component returns.
    """
    # TODO: Implement your component logic here


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        test_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
