"""Check the GPU, Secret, and artifact bindings in compiled AutoMM specs."""

import pytest
from kfp.compiler import Compiler

from ..pipeline import create_autogluon_multimodal_training_pipeline


@pytest.mark.parametrize("use_gpu", [True, False])
def test_compiled_prototype(tmp_path, use_gpu):
    """Keep credentials on the loader and request a GPU only for the GPU profile."""
    import yaml

    image = "quay.io/example/automm:prototype"
    output = tmp_path / "pipeline.yaml"
    Compiler().compile(create_autogluon_multimodal_training_pipeline(image, use_gpu=use_gpu), str(output))
    spec, platforms = yaml.safe_load_all(output.read_text())
    root = spec["root"]
    assert root["inputDefinitions"]["parameters"]["image_columns"]["parameterType"] == "LIST"
    assert set(root["outputDefinitions"]["artifacts"]) == {"model", "metrics", "data_quality"}
    assert "model_uri" in root["outputDefinitions"]["parameters"]

    executors = spec["deploymentSpec"]["executors"]
    assert all(executor["container"]["image"] == image for executor in executors.values())
    trainer = executors["exec-autogluon-multimodal-models-training"]["container"]
    accelerator = trainer["resources"].get("accelerator")
    if use_gpu:
        assert accelerator["resourceType"] == "nvidia.com/gpu"
        assert accelerator["resourceCount"] == "1"
    else:
        assert accelerator is None

    kubernetes = platforms["platforms"]["kubernetes"]
    assert kubernetes["pipelineConfig"]["workspace"]["size"] == "8Gi"
    assert set(kubernetes["deploymentSpec"]["executors"]) == {"exec-multimodal-image-data-loader"}
    assert kubernetes["deploymentSpec"]["executors"]["exec-multimodal-image-data-loader"]["secretAsEnv"][0][
        "secretNameParameter"
    ] == {"componentInputParameter": "train_data_secret_name"}
