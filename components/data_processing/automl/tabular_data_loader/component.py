from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["numpy", "pandas", "boto3"],
)
def automl_data_loader(file_key: str, bucket_name: str, full_dataset: dsl.Output[dsl.Dataset]):
    """Automl Data Loader component.

    Loads tabular (CSV) data from S3 in batches, sampling up to 1GB of data.
    The component reads data in chunks to efficiently handle large files without
    loading the entire dataset into memory at once.

    Args:
        file_key: Location of the CSV file in the S3 bucket.
        bucket_name: Name of the S3 bucket containing the file.
        target_column: Name of the column containing labels/target values for stratified sampling.
        full_dataset: Output dataset artifact where the sampled data will be saved.
        sampling_type: Type of sampling strategy. Options: "first_n_rows" (default) or "stratified".

    Returns:
        pandas.DataFrame: A sampled pandas DataFrame containing up to 1GB of data.
                         For "first_n_rows": returns first N rows up to 1GB limit.
                         For "stratified": returns stratified sample preserving target column distribution.
    """
    import io
    import os

    import boto3
    import pandas as pd

    # 1GB limit in bytes
    MAX_SIZE_BYTES = 1024 * 1024 * 1024

    def get_s3_client():
        """Create and return an S3 client using credentials from environment variables."""
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
        region_name = os.environ.get("AWS_DEFAULT_REGION")

        if (access_key and not secret_key) or (secret_key and not access_key):
            raise ValueError(
                "S3 credentials misconfigured: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must either "
                "both be set and non-empty, or both be unset. Check the 's3-secret' Kubernetes secret."
            )
        if not access_key and not secret_key:
            raise ValueError(
                "S3 credentials missing: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be provided via "
                "the 's3-secret' Kubernetes secret when using s3:// dataset URIs."
            )

        return boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    def load_data_in_batches(
        s3_client,
        bucket_name,
        file_key,
        max_size_bytes=MAX_SIZE_BYTES,
        sampling_type="first_n_rows",
        target_column=None,
    ):
        """Load CSV data from S3 in batches, sampling up to max_size_bytes.

        Reads the file from S3 in streaming mode and processes it in chunks using pandas,
        accumulating data until reaching the maximum size limit (1GB by default).
        Supports different sampling strategies: first_n_rows or stratified.
        For stratified sampling, sampling is performed incrementally during batch reading.

        Args:
            s3_client: Boto3 S3 client instance.
            bucket_name: Name of the S3 bucket.
            file_key: Key/path of the file in S3.
            max_size_bytes: Maximum size of data to read in bytes (default: 1GB).
            sampling_type: Type of sampling strategy ("first_n_rows" or "stratified").
            target_column: Name of the target column for stratified sampling (required if sampling_type="stratified").

        Returns:
            pandas.DataFrame: Sampled dataframe containing up to max_size_bytes of data.
        """
        # Validate stratified sampling parameters
        if sampling_type == "stratified":
            if target_column is None:
                raise ValueError(
                    "target_column must be provided when sampling_type='stratified'"
                )

        # Get the S3 object for streaming
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

        # Create a text wrapper around the streaming body
        # This allows pandas to read directly from the S3 stream
        body_stream = response["Body"]
        text_stream = io.TextIOWrapper(body_stream, encoding="utf-8")

        # Initialize variables for batch reading
        pandas_chunk_size = 10000  # Read 10k rows at a time

        if sampling_type == "stratified":
            # For stratified sampling, maintain subsampled data incrementally
            subsampled_data = None
            accumulated_size = 0

            try:
                for chunk_df in pd.read_csv(text_stream, chunksize=pandas_chunk_size):
                    # Drop rows with NA values in target column
                    chunk_df = chunk_df.dropna(subset=[target_column])

                    if chunk_df.empty:
                        continue

                    # Check if target column exists
                    if target_column not in chunk_df.columns:
                        raise ValueError(
                            f"Target column '{target_column}' not found in the dataset. "
                            f"Available columns: {list(chunk_df.columns)}"
                        )

                    # Remove singleton classes (classes with only 1 sample in this chunk)
                    # This helps maintain class distribution quality
                    stats = chunk_df[target_column].value_counts()
                    singleton_indexes = stats[stats == 1].index.values
                    for idx in singleton_indexes:
                        chunk_df = chunk_df[chunk_df[target_column] != idx]

                    if chunk_df.empty:
                        continue

                    # Join previous subsampled batch with new one
                    if subsampled_data is not None:
                        combined_data = pd.concat(
                            [subsampled_data, chunk_df], ignore_index=True
                        )
                    else:
                        combined_data = chunk_df

                    # Calculate memory usage of combined data
                    combined_memory = combined_data.memory_usage(deep=True).sum()

                    if combined_memory <= max_size_bytes:
                        # If under limit, keep all data
                        subsampled_data = combined_data
                        accumulated_size = combined_memory
                    else:
                        # If over limit, perform stratified sampling to bring it under limit
                        # Calculate sampling fraction based on memory
                        sampling_frac = max_size_bytes / combined_memory

                        # Perform stratified sampling: sample proportionally from each class
                        # This preserves the class distribution while reducing size
                        subsampled_data = (
                            combined_data.groupby(target_column, group_keys=False)
                            .apply(
                                lambda x: x.sample(
                                    frac=min(sampling_frac, 1.0), random_state=42
                                )
                            )
                            .reset_index(drop=True)
                        )

                        accumulated_size = subsampled_data.memory_usage(deep=True).sum()

                        # Continue reading to potentially improve class distribution
                        # The sampling will be applied again if we exceed the limit

                # Return the final subsampled data
                if subsampled_data is None:
                    return pd.DataFrame()

                # Shuffle to mix classes
                result_df = subsampled_data.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )
                return result_df

            except Exception as e:
                if subsampled_data is None or subsampled_data.empty:
                    raise ValueError(f"Error reading CSV from S3: {str(e)}")
                # Return what we have so far
                return subsampled_data.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )

        else:
            # For "first_n_rows" sampling, use the original approach
            chunk_list = []
            accumulated_size = 0

            try:
                for chunk_df in pd.read_csv(text_stream, chunksize=pandas_chunk_size):
                    # Calculate memory usage of this chunk
                    chunk_memory = chunk_df.memory_usage(deep=True).sum()

                    if accumulated_size + chunk_memory > max_size_bytes:
                        # Take only a portion of this chunk to stay within limit
                        remaining_bytes = max_size_bytes - accumulated_size
                        # Estimate rows to take based on average memory per row
                        bytes_per_row = (
                            chunk_memory / len(chunk_df) if len(chunk_df) > 0 else 0
                        )
                        if bytes_per_row > 0:
                            rows_to_take = max(1, int(remaining_bytes / bytes_per_row))
                            chunk_df = chunk_df.head(rows_to_take)
                            chunk_list.append(chunk_df)
                        break

                    chunk_list.append(chunk_df)
                    accumulated_size += chunk_memory

                    if accumulated_size >= max_size_bytes:
                        break
            except Exception as e:
                # If there's an error, try to return what we have so far
                if not chunk_list:
                    raise ValueError(f"Error reading CSV from S3: {str(e)}")

            # Combine all chunks into a single dataframe
            if chunk_list:
                result_df = pd.concat(chunk_list, ignore_index=True)
            else:
                # If no chunks were read, return empty dataframe
                result_df = pd.DataFrame()

            return result_df

    # Load data in batches
    s3_client = get_s3_client()
    sampled_dataframe = load_data_in_batches(
        s3_client,
        bucket_name,
        file_key,
        sampling_type=sampling_type,
        target_column=target_column,
    )

    # Save the sampled dataframe to the output artifact
    sampled_dataframe.to_csv(full_dataset.path, index=False)

    return sampled_dataframe


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        automl_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
