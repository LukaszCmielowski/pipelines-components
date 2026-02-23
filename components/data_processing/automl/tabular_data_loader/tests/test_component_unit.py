"""Tests for the tabular_data_loader component."""

import io
from unittest import mock

import pytest

from ..component import automl_data_loader


class TestAutomlDataLoaderUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(automl_data_loader)
        assert hasattr(automl_data_loader, "python_func")

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_with_default_parameters_first_n_rows(self, mock_boto_client, tmp_path):
        """Test component with default sampling_type (first_n_rows) and no target_column."""
        pd = pytest.importorskip("pandas")
        # CSV content: small dataset
        csv_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.return_value = {"Body": body_stream}
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "output.csv")

        result = automl_data_loader.python_func(
            file_key="data/file.csv",
            bucket_name="my-bucket",
            full_dataset=full_dataset,
        )

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["a", "b", "c"]
        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/file.csv")
        assert (tmp_path / "output.csv").exists()
        saved = pd.read_csv(full_dataset.path)
        pd.testing.assert_frame_equal(saved, result)

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_explicit_first_n_rows(self, mock_boto_client, tmp_path):
        """Test component with explicit sampling_type='first_n_rows'."""
        pd = pytest.importorskip("pandas")
        csv_content = "x,y,z\n10,20,30\n40,50,60\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.return_value = {"Body": body_stream}
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "out.csv")

        result = automl_data_loader.python_func(
            file_key="s3/path/data.csv",
            bucket_name="bucket",
            full_dataset=full_dataset,
            sampling_type="first_n_rows",
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["x", "y", "z"]

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_sampling_with_target_column(self, mock_boto_client, tmp_path):
        """Test component with sampling_type='stratified' and target_column."""
        pd = pytest.importorskip("pandas")
        # CSV with target column; multiple classes so stratified logic runs
        csv_content = "feature1,feature2,target\n1,2,A\n2,3,A\n3,4,A\n4,5,B\n5,6,B\n6,7,B\n7,8,C\n8,9,C\n9,10,C\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.return_value = {"Body": body_stream}
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "stratified_out.csv")

        result = automl_data_loader.python_func(
            file_key="data/train.csv",
            bucket_name="my-bucket",
            full_dataset=full_dataset,
            sampling_type="stratified",
            target_column="target",
        )

        assert isinstance(result, pd.DataFrame)
        assert "target" in result.columns
        assert set(result["target"].unique()) == {"A", "B", "C"}
        assert len(result) == 9  # all rows under 1GB, so all kept
        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/train.csv")
        assert (tmp_path / "stratified_out.csv").exists()

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_requires_target_column(self, mock_boto_client, tmp_path):
        """Test that sampling_type='stratified' without target_column raises ValueError."""
        pytest.importorskip("pandas")
        mock_s3 = mock.MagicMock()
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "out.csv")

        with pytest.raises(ValueError, match="target_column must be provided when sampling_type='stratified'"):
            automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                full_dataset=full_dataset,
                sampling_type="stratified",
                target_column=None,
            )

        mock_s3.get_object.assert_not_called()

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_target_column_not_in_dataset(self, mock_boto_client, tmp_path):
        """Test that stratified sampling with missing target column raises ValueError."""
        pytest.importorskip("pandas")
        csv_content = "a,b,c\n1,2,3\n4,5,6\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.return_value = {"Body": body_stream}
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "out.csv")

        with pytest.raises(ValueError, match=r"Target column 'label' not found|Error reading CSV from S3"):
            automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                full_dataset=full_dataset,
                sampling_type="stratified",
                target_column="label",
            )

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_drops_na_in_target(self, mock_boto_client, tmp_path):
        """Test that stratified sampling drops rows with NA in target_column."""
        pd = pytest.importorskip("pandas")
        csv_content = "f1,f2,target\n1,2,A\n2,3,\n3,4,B\n4,5,B\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.return_value = {"Body": body_stream}
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "out.csv")

        result = automl_data_loader.python_func(
            file_key="data/file.csv",
            bucket_name="bucket",
            full_dataset=full_dataset,
            sampling_type="stratified",
            target_column="target",
        )

        assert isinstance(result, pd.DataFrame)
        # Row with NA in target is dropped; singleton classes may also be removed
        assert result["target"].notna().all()
        assert len(result) >= 2
