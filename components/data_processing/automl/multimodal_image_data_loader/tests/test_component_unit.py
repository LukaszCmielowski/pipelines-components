"""Verify S3 key handling and duplicate-safe splits."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kfp import dsl

from ..component import multimodal_image_data_loader


@pytest.fixture
def s3(monkeypatch):
    """Serve small PNGs and a manifest from a simulated S3 client."""
    image = pytest.importorskip("PIL.Image")
    files = {}
    lines = ["image_key,category"]
    for i in range(6):
        body = io.BytesIO()
        image.new("RGB", (8, 8), (i * 30, 0, 0)).save(body, format="PNG")
        files[f"products/images/{i}.png"] = body.getvalue()
        lines.append(f"images/{i}.png,{'bag' if i < 3 else 'shoe'}")
    files["products/images/copy.png"] = files["products/images/0.png"]
    files["products/train.csv"] = ("\n".join([*lines, "images/copy.png,bag"]) + "\n").encode()
    client = MagicMock()
    client.get_object.side_effect = lambda Bucket, Key: {"Body": io.BytesIO(files[Key])}
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *args, **kwargs: client))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    monkeypatch.setenv("AWS_S3_ENDPOINT", "https://s3.example.invalid")
    return files, client


def run_loader(workspace):
    """Return the split path and public data quality document."""
    output = dsl.Artifact(uri=str(workspace / "quality.json"))
    split = multimodal_image_data_loader.python_func(
        "bucket", "products/train.csv", ["image_key"], "category", "binary", "image_prediction", str(workspace), output
    )
    return json.loads(Path(split).read_text()), json.loads(Path(output.path).read_text())


def test_splits_are_reproducible_and_deduplicate_image_content(tmp_path, s3):
    """Deduplicate copied bytes and keep each distinct image in one split."""
    staged, quality = run_loader(tmp_path / "first")
    second, _ = run_loader(tmp_path / "second")
    assert staged == second
    assert quality["manifest_rows"] == 7
    assert quality["distinct_images"] == 6
    assert quality["duplicate_rows"] == 1
    assert quality["split_sizes"] == {"train": 2, "tuning": 2, "holdout": 2}
    all_paths = []
    for split in staged["splits"].values():
        assert {row["label"] for row in split} == {"bag", "shoe"}
        all_paths.extend(row["image"] for row in split)
    assert len(all_paths) == len(set(all_paths))
    assert "products/" not in json.dumps(quality)
    assert len(s3[1].get_object.call_args_list) == 16


def test_rejects_path_traversal_before_fetching_images(tmp_path, s3):
    """Reject relative traversal before requesting any image objects."""
    files, client = s3
    files["products/train.csv"] = b"image_key,category\n../hidden.png,bag\nimages/3.png,shoe\n"
    with pytest.raises(ValueError, match="requires a relative image key"):
        run_loader(tmp_path)
    client.get_object.assert_called_once_with(Bucket="bucket", Key="products/train.csv")


@pytest.mark.parametrize(("image", "error"), [(b"not an image", "readable JPEG or PNG"), (None, "conflicting label")])
def test_rejects_bad_images_and_conflicting_duplicates(tmp_path, s3, image, error):
    """Fail the run on unreadable images or inconsistent labels."""
    files, _ = s3
    if image is None:
        files["products/train.csv"] = files["products/train.csv"].replace(b"copy.png,bag", b"copy.png,shoe")
    else:
        files["products/images/0.png"] = image
    with pytest.raises(ValueError, match=error):
        run_loader(tmp_path)
