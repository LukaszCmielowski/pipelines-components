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
    def test_component_with_default_parameters(self, mock_boto_client, tmp_path):
        """Test component with default sampling_method=None (resolved from task_type=regression -> random)."""
        pd = pytest.importorskip("pandas")
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
        assert hasattr(result, "sample_config")
        assert result.sample_config["n_samples"] == 3
        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/file.csv")
        assert (tmp_path / "output.csv").exists()
        saved = pd.read_csv(full_dataset.path)
        assert list(saved.columns) == ["a", "b", "c"]
        assert len(saved) == 3

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_explicit_first_n_rows(self, mock_boto_client, tmp_path):
        """Test component with explicit sampling_method='first_n_rows'."""
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
            sampling_method="first_n_rows",
        )

        assert hasattr(result, "sample_config")
        assert result.sample_config["n_samples"] == 2
        saved = pd.read_csv(full_dataset.path)
        assert list(saved.columns) == ["x", "y", "z"]
        assert len(saved) == 2

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_sampling_with_label_column(self, mock_boto_client, tmp_path):
        """Test component with sampling_method='stratified' and label_column."""
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
            sampling_method="stratified",
            label_column="target",
        )

        assert hasattr(result, "sample_config")
        assert result.sample_config["n_samples"] == 9
        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/train.csv")
        assert (tmp_path / "stratified_out.csv").exists()
        saved = pd.read_csv(full_dataset.path)
        assert "target" in saved.columns
        assert set(saved["target"].unique()) == {"A", "B", "C"}
        assert len(saved) == 9

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_requires_label_column(self, mock_boto_client, tmp_path):
        """Test that sampling_method='stratified' without label_column raises ValueError."""
        pytest.importorskip("pandas")
        mock_s3 = mock.MagicMock()
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "out.csv")

        with pytest.raises(ValueError, match="label_column must be provided when sampling_method='stratified'"):
            automl_data_loader.python_func(
                file_key="data/file.csv",
                bucket_name="bucket",
                full_dataset=full_dataset,
                sampling_method="stratified",
                label_column=None,
            )

        mock_s3.get_object.assert_not_called()

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_label_column_not_in_dataset(self, mock_boto_client, tmp_path):
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
                sampling_method="stratified",
                label_column="label",
            )

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_stratified_drops_na_in_target(self, mock_boto_client, tmp_path):
        """Test that stratified sampling drops rows with NA in label_column."""
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
            sampling_method="stratified",
            label_column="target",
        )

        assert hasattr(result, "sample_config")
        assert result.sample_config["n_samples"] >= 2
        saved = pd.read_csv(full_dataset.path)
        assert saved["target"].notna().all()
        assert len(saved) >= 2

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_random_sampling_basic(self, mock_boto_client, tmp_path):
        """Test component with sampling_method='random' writes valid CSV and returns sample_config."""
        pd = pytest.importorskip("pandas")
        csv_content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n"
        body_stream = io.BytesIO(csv_content.encode("utf-8"))

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.return_value = {"Body": body_stream}
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "random_out.csv")

        result = automl_data_loader.python_func(
            file_key="data/file.csv",
            bucket_name="my-bucket",
            full_dataset=full_dataset,
            sampling_method="random",
        )

        assert result.sample_config["n_samples"] == 4
        assert (tmp_path / "random_out.csv").exists()
        saved = pd.read_csv(full_dataset.path)
        assert list(saved.columns) == ["a", "b", "c"]
        assert len(saved) == 4
        mock_s3.get_object.assert_called_once_with(Bucket="my-bucket", Key="data/file.csv")

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_random_sampling_deterministic(self, mock_boto_client, tmp_path):
        """Test that random sampling with fixed random_state is reproducible."""
        pd = pytest.importorskip("pandas")
        csv_content = "x,y\n1,2\n3,4\n5,6\n7,8\n9,10\n"

        def get_object(**kwargs):
            return {"Body": io.BytesIO(csv_content.encode("utf-8"))}

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.side_effect = get_object
        mock_boto_client.return_value = mock_s3

        full_dataset1 = mock.MagicMock()
        full_dataset1.path = str(tmp_path / "out1.csv")
        full_dataset2 = mock.MagicMock()
        full_dataset2.path = str(tmp_path / "out2.csv")

        result1 = automl_data_loader.python_func(
            file_key="data/file.csv",
            bucket_name="bucket",
            full_dataset=full_dataset1,
            sampling_method="random",
        )
        result2 = automl_data_loader.python_func(
            file_key="data/file.csv",
            bucket_name="bucket",
            full_dataset=full_dataset2,
            sampling_method="random",
        )

        assert result1.sample_config["n_samples"] == result2.sample_config["n_samples"] == 5
        df1 = pd.read_csv(full_dataset1.path)
        df2 = pd.read_csv(full_dataset2.path)
        pd.testing.assert_frame_equal(df1, df2)

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "test_key", "AWS_SECRET_ACCESS_KEY": "test_secret"})
    @mock.patch("boto3.client")
    def test_component_random_sampling_multiple_chunks(self, mock_boto_client, tmp_path):
        """Test random sampling with CSV large enough to trigger multiple chunks (>10k rows)."""
        pd = pytest.importorskip("pandas")
        # Build CSV with header + 15000 rows so we get at least 2 chunks (chunk_size=10000)
        header = "col1,col2\n"
        rows = "\n".join(f"{i},{i * 2}" for i in range(15000))
        csv_content = header + rows
        body_stream = io.BytesIO(csv_content.encode("utf-8"))

        mock_s3 = mock.MagicMock()
        mock_s3.get_object.return_value = {"Body": body_stream}
        mock_boto_client.return_value = mock_s3

        full_dataset = mock.MagicMock()
        full_dataset.path = str(tmp_path / "random_multi.csv")

        result = automl_data_loader.python_func(
            file_key="data/large.csv",
            bucket_name="bucket",
            full_dataset=full_dataset,
            sampling_method="random",
        )

        assert result.sample_config["n_samples"] == 15000
        assert (tmp_path / "random_multi.csv").exists()
        saved = pd.read_csv(full_dataset.path)
        assert list(saved.columns) == ["col1", "col2"]
        assert len(saved) == 15000
