"""Stage and split an S3 image classification dataset."""

from kfp import dsl


@dsl.component(base_image="python:3.12-slim", install_kfp_package=False)
def multimodal_image_data_loader(
    bucket_name: str,
    file_key: str,
    image_columns: list[str],
    label_column: str,
    task_type: str,
    task_family: str,
    workspace_path: str,
    data_quality: dsl.Output[dsl.Artifact],
) -> str:
    """Validate an image manifest, stage images, and write fixed train, tuning, and holdout splits.

    Args:
        bucket_name: S3 bucket containing the manifest and images.
        file_key: CSV manifest object key. Image keys are relative to its directory.
        image_columns: One manifest column holding JPEG or PNG object keys.
        label_column: Manifest column holding the class label.
        task_type: binary or multiclass.
        task_family: image_prediction.
        workspace_path: Run-scoped workspace shared with the trainer.
        data_quality: Aggregate validation and split information.

    Returns:
        Path to the split manifest in the shared workspace.
    """
    import csv
    import hashlib
    import io
    import json
    import os
    from collections import Counter, defaultdict
    from contextlib import closing
    from pathlib import Path
    from random import Random

    import boto3
    from kfp_components.components.training.automl.shared.user_test_data import (
        resolve_s3_env_credentials,
        validate_s3_env_credentials,
    )
    from PIL import Image

    if task_family != "image_prediction" or task_type not in {"binary", "multiclass"}:
        raise ValueError("Use image_prediction with task_type binary or multiclass.")
    if len(image_columns) != 1 or not image_columns[0].strip() or image_columns[0] == label_column:
        raise ValueError("Declare one image column distinct from the label column.")
    image_column = image_columns[0]
    if not bucket_name.strip() or not label_column.strip() or not workspace_path:
        raise ValueError("The S3 bucket, label column, and run workspace are required.")

    def relative_key(value):
        return (
            bool(value)
            and all(segment not in {"", ".", ".."} for segment in value.split("/"))
            and not any(char in value for char in "\\:?#")
            and not any(ord(char) < 32 for char in value)
        )

    if not relative_key(file_key) or not file_key.lower().endswith(".csv"):
        raise ValueError("Provide a relative S3 object key for a CSV manifest.")

    mib = 1024**2
    seed = 42
    data_dir = Path(workspace_path) / "automl-image-data"
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    credentials = resolve_s3_env_credentials()
    validate_s3_env_credentials(credentials)
    client = boto3.client(
        "s3",
        endpoint_url=credentials["endpoint_url"],
        region_name=credentials["region_name"],
        aws_access_key_id=credentials["access_key"],
        aws_secret_access_key=credentials["secret_key"],
        verify=os.getenv("SSL_CERT_FILE") or os.getenv("AWS_CA_BUNDLE") or True,
    )

    def read_object(key, limit, description):
        try:
            with closing(client.get_object(Bucket=bucket_name, Key=key)["Body"]) as body:
                data = body.read(limit + 1)
        except Exception:
            raise ValueError(f"Could not read {description} from the configured S3 bucket.") from None
        if len(data) > limit:
            raise ValueError(f"{description} exceeds the prototype size limit.")
        return data

    manifest = read_object(file_key, 16 * mib, "the CSV manifest")
    try:
        reader = csv.DictReader(io.StringIO(manifest.decode("utf-8-sig"), newline=""), strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not {image_column, label_column}.issubset(columns):
            raise ValueError("The CSV must have unique headers and include the image and label columns.")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if row_number > 10_001:
                raise ValueError("The prototype accepts at most 10,000 manifest rows.")
            key, label = row[image_column], row[label_column]
            if not isinstance(key, str) or not relative_key(key) or not label or not label.strip() or None in row:
                raise ValueError(f"CSV row {row_number} requires a relative image key and a class label.")
            rows.append((key, label.strip()))
    except (UnicodeError, csv.Error):
        raise ValueError("The manifest must be a well-formed UTF-8 CSV.") from None

    class_count = len({label for _, label in rows})
    if (task_type == "binary" and class_count != 2) or (task_type == "multiclass" and class_count < 3):
        raise ValueError("Binary requires two classes. Multiclass requires at least three.")

    prefix = file_key.rpartition("/")[0]
    by_image = {}
    by_digest = {}
    groups = defaultdict(list)
    formats = Counter()
    staged_bytes = 0
    for ordinal, (key, label) in enumerate(rows, start=2):
        if key not in by_image:
            object_key = f"{prefix}/{key}" if prefix else key
            data = read_object(object_key, 8 * mib, f"the image for CSV row {ordinal}")
            digest = hashlib.sha256(data).hexdigest()
            if digest not in by_digest:
                try:
                    with Image.open(io.BytesIO(data), formats=["JPEG", "PNG"]) as image:
                        if image.width * image.height > 16_000_000 or getattr(image, "n_frames", 1) != 1:
                            raise ValueError("Image dimensions or frame count exceed the limit.")
                        image.load()
                        image_format = image.format
                except (OSError, ValueError, Image.DecompressionBombError):
                    raise ValueError(
                        f"CSV row {ordinal} must reference a readable JPEG or PNG within limits."
                    ) from None
                staged_bytes += len(data)
                if staged_bytes > 4096 * mib:
                    raise ValueError("Staged images exceed the 4 GiB workspace budget.")
                suffix = "jpg" if image_format == "JPEG" else "png"
                relative_path = f"images/{digest}.{suffix}"
                (data_dir / relative_path).write_bytes(data)
                by_digest[digest] = (relative_path, label)
                groups[label].append(digest)
                formats[image_format] += 1
            by_image[key] = digest
        if by_digest[by_image[key]][1] != label:
            raise ValueError(f"CSV row {ordinal} assigns a conflicting label to a duplicate image.")

    split_rows = {"train": [], "tuning": [], "holdout": []}
    rng = Random(seed)
    for label, digests in sorted(groups.items()):
        if len(digests) < 3:
            raise ValueError("Each class needs at least three distinct images for the three splits.")
        digests.sort()
        rng.shuffle(digests)
        holdout_count = max(1, int(len(digests) * 0.15))
        selections = (digests[2 * holdout_count :], digests[holdout_count : 2 * holdout_count], digests[:holdout_count])
        for name, selection in zip(split_rows, selections, strict=True):
            split_rows[name].extend({"image": by_digest[digest][0], "label": label} for digest in selection)
    for split in split_rows.values():
        rng.shuffle(split)

    quality = {
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "seed": seed,
        "manifest_rows": len(rows),
        "distinct_images": len(by_digest),
        "duplicate_rows": len(rows) - len(by_digest),
        "image_formats": dict(formats),
        "staged_image_bytes": staged_bytes,
        "distinct_images_per_class": sorted(map(len, groups.values())),
        "split_sizes": {name: len(split) for name, split in split_rows.items()},
    }
    split_path = data_dir / "splits.json"
    split_path.write_text(
        json.dumps({"splits": split_rows, "data_quality": quality, "task_type": task_type}) + "\n",
        encoding="utf-8",
    )
    quality_path = Path(data_quality.path)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")
    return str(split_path)
