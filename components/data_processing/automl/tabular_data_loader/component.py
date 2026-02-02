from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["numpy", "pandas", "boto3"],
)
def automl_data_loader(
    file_key: str, bucket_name: str, full_dataset: dsl.Output[dsl.Dataset]
):
    """AutoML Data Loader component.

    Loads tabular (CSV) data from S3 in batches, sampling up to 1GB of data.
    The component reads data in chunks to efficiently handle large files without
    loading the entire dataset into memory at once.

    Args:
        file_key: Location of the CSV file in the S3 bucket.
        bucket_name: Name of the S3 bucket containing the file.
        full_dataset: Output dataset artifact where the sampled data will be saved.

    Returns:
        pandas.DataFrame: A sampled pandas DataFrame containing up to 1GB of data
                         from the top of the file.
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
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        region_name = os.environ.get("AWS_REGION")

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
        s3_client, bucket_name, file_key, max_size_bytes=MAX_SIZE_BYTES
    ):
        """Load CSV data from S3 in batches, sampling up to max_size_bytes.

        Reads the file from S3 in streaming mode and processes it in chunks using pandas,
        accumulating data until reaching the maximum size limit (1GB by default).

        Args:
            s3_client: Boto3 S3 client instance.
            bucket_name: Name of the S3 bucket.
            file_key: Key/path of the file in S3.
            max_size_bytes: Maximum size of data to read in bytes (default: 1GB).

        Returns:
            pandas.DataFrame: Sampled dataframe containing up to max_size_bytes of data
                             from the top of the file.
        """
        # Get the S3 object for streaming
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

        # Create a text wrapper around the streaming body
        # This allows pandas to read directly from the S3 stream
        body_stream = response["Body"]
        text_stream = io.TextIOWrapper(body_stream, encoding="utf-8")

        # Read CSV in chunks using pandas for efficient memory usage
        chunk_list = []
        accumulated_size = 0
        pandas_chunk_size = 10000  # Read 10k rows at a time

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
    sampled_dataframe = load_data_in_batches(s3_client, bucket_name, file_key)

    # Save the sampled dataframe to the output artifact
    sampled_dataframe.to_csv(full_dataset.path, index=False)

    return sampled_dataframe


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        automl_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
