from kfp import dsl


@dsl.component(
    base_image="quay.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9:rhoai-3.2",
)
def notebook_generation(
    problem_type: str,
    model_name: str,
    notebook_artifact: dsl.Output[dsl.Artifact],
    pipeline_name: str,
    run_id: str,
    sample_row: str,
    label_column: str,
):
    """Generate a Jupyter notebook for reviewing and running an AutoGluon predictor.

    Produces a notebook artifact (automl_predictor_notebook.ipynb) that lets users
    review the experiment leaderboard, load a trained AutoGluon model from S3, and
    run predictions. The notebook is pre-filled with pipeline run details, model
    name, and a sample row for prediction.

    **Problem types:** Use ``problem_type`` to select the template:

    - **regression**: Template uses ``predict(score_df)`` for numeric targets.
    - **binary** or **multiclass**: Template uses ``predict_proba(score_df)`` and
      includes a confusion matrix section. Both values share the same classification
      template.

    Invalid values raise ``ValueError``.

    Args:
        problem_type: One of ``"regression"``, ``"binary"``, or ``"multiclass"``.
            Determines which notebook template is used.
        model_name: Name of the trained model to load, matching the leaderboard
            model column.
        notebook_artifact: Output artifact where the generated notebook file
            (automl_predictor_notebook.ipynb) is written.
        pipeline_name: Full pipeline run name (e.g. from KFP); used to locate
            artifacts in S3. The component strips the last hyphen-separated
            segment (run suffix) for the notebook path.
        run_id: Pipeline run ID; used with pipeline_name to form the S3 prefix
            for leaderboard and model artifacts.
        sample_row: JSON string of a single row (object of feature names to
            values), used in the notebook's prediction example. The component
            parses it, removes the label column, and injects the result.
        label_column: Key in the parsed sample_row for the target/label column;
            this column is omitted from the sample row in the notebook.

    Returns:
        None. Writes the notebook to notebook_artifact.path.
    """
    import json
    from pathlib import Path

    # TODO: Move to build package in next stages

    REGRESSION = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "a12d957a-c313-4e92-9578-44f6a48560d5",
                "metadata": {},
                "source": [
                    "![AutoML Banner](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxNzk2IDEwMCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTc5NiAxMDA7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbC1ydWxlOmV2ZW5vZGQ7Y2xpcC1ydWxlOmV2ZW5vZGQ7ZmlsbDp1cmwoI1NWR0lEXzFfKTt9Cgkuc3Qxe2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzdHJva2Utd2lkdGg6MjtzdHJva2UtbWl0ZXJsaW1pdDoxMDt9Cgkuc3Qye2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzdHJva2Utd2lkdGg6MS41O3N0cm9rZS1taXRlcmxpbWl0OjEwO30KCS5zdDN7ZmlsbDojRkZGRkZGO30KCS5zdDR7Zm9udC1mYW1pbHk6J0hlbHZldGljYSBOZXVlJywgQXJpYWwsIHNhbnMtc2VyaWY7fQoJLnN0NXtmb250LXNpemU6MzJweDt9Cgkuc3Q2e2ZpbGw6IzNEM0QzRDt9Cgkuc3Q3e2ZpbGw6IzkzOTU5ODt9Cgkuc3Q4e29wYWNpdHk6MC4yO2ZpbGw6dXJsKCNTVkdJRF8yXyk7ZW5hYmxlLWJhY2tncm91bmQ6bmV3O30KCS5zdDl7Zm9udC13ZWlnaHQ6NTAwO30KPC9zdHlsZT4KPHJlY3Qgd2lkdGg9IjE3OTYiIGhlaWdodD0iMTAwIiBmaWxsPSIjMTYxNjE2Ii8+CjxsaW5lYXJHcmFkaWVudCBpZD0iU1ZHSURfMV8iIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iNDIuODYiIHkxPSI1MCIgeDI9Ijc5LjcxIiB5Mj0iNTAiPgoJPHN0b3Agb2Zmc2V0PSIwIiBzdHlsZT0ic3RvcC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuMjEiIHN0eWxlPSJzdG9wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC43NSIgc3R5bGU9InN0b3AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjwhLS0gQXV0b01MIEljb24vTG9nbyBwbGFjZWhvbGRlciAtIHNpbXBsaWZpZWQgZ2VvbWV0cmljIHNoYXBlIC0tPgo8cGF0aCBjbGFzcz0ic3QwIiBkPSJNNTIuNCw0NS45YzAtMi4zLDEuOC00LjEsNC4xLTQuMXM0LjEsMS44LDQuMSw0LjFTNTguOCw1MCw1Ni41LDUwbDAsMGMtMi4yLDAuMS00LTEuNy00LjEtMy45CglDNTIuNCw0Niw1Mi40LDQ2LDUyLjQsNDUuOXogTTc3LjUsNTIuNWMtMC44LTEuMS0xLjQtMi4zLTEuOS0zLjVjMS4yLTQuNSwwLjctOC42LTEuOC0xMS45Yy0yLjktMy44LTguMi02LTE0LjUtNi4xCgljLTQuNS0wLjEtOC44LDEuNy0xMiw0LjhjLTMsMy00LjYsNy4yLTQuNSwxMS41Yy0wLjEsMi45LDAuOSw1LjgsMi43LDguMWMwLjgsMC44LDEuMywxLjksMS40LDN2NC41Yy0wLjgsMC41LTEuNCwxLjMtMS40LDIuMwoJYzAuMiwxLjUsMS41LDIuNiwzLDIuNGMxLjItMC4yLDIuMi0xLjEsMi40LTIuNGMwLTEtMC41LTEuOS0xLjQtMi4zdi00LjVjMC0yLTEtMy4zLTEuOS00LjZjLTEuNS0xLjktMi4yLTQuMi0yLjEtNi41CgljMC0zLjUsMS40LTYuOSwzLjgtOS40YzIuNy0yLjcsNi4zLTQuMSwxMC00LjFjNS41LDAsOS44LDEuOSwxMi4xLDVjMiwyLjgsMi41LDYuMywxLjQsOS42Yy0wLjQsMS4yLDAuNiwyLjcsMi4zLDUuNgoJYzAuNiwwLjksMS4yLDEuOSwxLjYsMi45Yy0wLjksMC43LTIsMS4yLTMuMSwxLjVjLTAuNSwwLjQtMC43LDAuOS0wLjgsMS41VjY1YzAsMC40LTAuMSwwLjgtMC40LDEuMWMtMC4zLDAuMi0wLjcsMC4zLTEuMSwwLjMKCWMtMS42LTAuMy0zLjQtMC43LTUuMi0xLjF2LTQuOGMwLjgtMC41LDEuNC0xLjQsMS40LTIuM2MwLTEuNS0xLjItMi43LTIuNy0yLjdzLTIuNywxLjItMi43LDIuN2MwLDEsMC41LDEuOSwxLjQsMi4zdjQuMQoJYy0wLjQtMC4xLTAuNy0wLjEtMS4xLTAuM2MtNC41LTEuMS00LjUtMi42LTQuNS0zLjR2LTguM2MzLjItMC43LDUuNC0zLjUsNS41LTYuN2MtMC4xLTMuOC0zLjMtNi43LTcuMS02LjZjLTMuNiwwLjEtNi40LDMtNi42LDYuNgoJYzAsMy4yLDIuMyw2LDUuNSw2Ljd2OC4zYzAsMiwwLjcsNC42LDYuNiw2LjFjMywwLjgsNiwxLjUsOS4xLDEuOWMwLjMsMCwwLjYsMC4xLDAuOCwwLjFjMSwwLDEuOS0wLjMsMi42LTEKCWMwLjktMC44LDEuNC0xLjksMS40LTMuMXYtNC41YzItMC44LDQuMS0yLDQuMS0zLjdDNzkuNyw1NS45LDc5LDU0LjYsNzcuNSw1Mi41eiIvPgo8Y2lyY2xlIGNsYXNzPSJzdDEiIGN4PSI1Ni41IiBjeT0iNDUuOSIgcj0iNS40Ii8+CjxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjQ4LjMiIGN5PSI2NSIgcj0iMS42Ii8+CjxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjY0LjgiIGN5PSI1OC4yIiByPSIxLjYiLz4KPHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAxMDEuMDIgNTkuMzMpIiBjbGFzcz0ic3QzIHN0NCBzdDUiPkF1dG9NTDwvdGV4dD4KPHJlY3QgeD0iMjMxLjEiIHk9IjM0IiBjbGFzcz0ic3Q2IiB3aWR0aD0iMSIgaGVpZ2h0PSIzMiIvPgo8dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDI1Ni4yOSA1OS42NikiIGNsYXNzPSJzdDcgc3Q0IHN0NSI+UGFydCBvZiBSZWQgSGF0IE9wZW5TaGlmdCBBSTwvdGV4dD4KPGxpbmVhckdyYWRpZW50IGlkPSJTVkdJRF8yXyIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiIHgxPSI3NzMuOCIgeTE9IjUwIiB4Mj0iMTc5NiIgeTI9IjUwIj4KCTxzdG9wIG9mZnNldD0iMCIgc3R5bGU9InN0b3AtY29sb3I6IzE2MTYxNiIvPgoJPHN0b3Agb2Zmc2V0PSIwLjUyIiBzdHlsZT0ic3RvcC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuNjIiIHN0eWxlPSJzdG9wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC44OCIgc3R5bGU9InN0b3AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjxyZWN0IHg9Ijc3My44IiBjbGFzcz0ic3Q4IiB3aWR0aD0iMTAyMi4yIiBoZWlnaHQ9IjEwMCIvPgo8dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDE0NDguMTY0MSA1OS40NikiIGNsYXNzPSJzdDMgc3Q0IHN0NSBzdDkiPlByZWRpY3RvciBOb3RlYm9vazwvdGV4dD4KPC9zdmc+Cg==)"  # noqa: E501
                ],
            },
            {
                "cell_type": "markdown",
                "id": "0e9aa72f",
                "metadata": {},
                "source": [
                    "## Notebook content\n",
                    "\n",
                    "This notebook lets you review the experiment leaderboard for insights into trained model evaluation quality, load a chosen AutoGluon model from S3, and run predictions. \n",  # noqa: E501
                    "\n",
                    "\n",
                    "**Tips:**\n",
                    "- Ensure the S3 connection to pipeline run results is configured so the notebook can access run artifacts.\n",  # noqa: E501
                    "- The model name must match one of the models listed in the leaderboard (the **model** column).\n",
                    "\n",
                    "### Contents\n",
                    "This notebook contains the following parts:\n",
                    "\n",
                    "**[Setup](#setup)**  \n",
                    "**[Experiment run details](#experiment-run-details)**  \n",
                    "**[Experiment leaderboard](#experiment-leaderboard)**  \n",
                    "**[Download trained model](#download-trained-model)**  \n",
                    "**[Model insights](#model-insights)**  \n",
                    "**[Load the predictor](#load-the-predictor)**  \n",
                    "**[Predict the values](#predict-the-values)**  \n",
                    "**[Summary and next steps](#summary-and-next-steps)**",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "a7d9cf2b-18cc-4ac9-87af-a74f8bf60322",
                "metadata": {},
                "source": ['<a id="setup"></a>\n', "## Setup"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "5bacd972",
                "metadata": {},
                "outputs": [],
                "source": ["import warnings\n", "\n", 'warnings.filterwarnings("ignore")'],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cec84205-8ee9-4aaf-a97e-4ef576e7b9da",
                "metadata": {},
                "outputs": [],
                "source": ["%pip install autogluon.tabular[all]==1.5 | tail -n 1"],
            },
            {
                "cell_type": "markdown",
                "id": "e8ff506e-f1a3-4990-a979-7790a5105251",
                "metadata": {},
                "source": [
                    '<a id="experiment-run-details"></a>\n',
                    "## Experiment run details\n",
                    "\n",
                    "Set the pipeline name, and run ID that identify the training run whose artifacts you want to load. These values are typically available from the pipeline run or workbench.",  # noqa: E501
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "fa7f736d-0b5c-4988-87a5-4d1a5cde0873",
                "metadata": {},
                "outputs": [],
                "source": ['pipeline_name = "<PIPELINE_NAME>"\n', 'run_id = "<RUN_ID>"'],
            },
            {
                "cell_type": "markdown",
                "id": "00cc5969-0f9b-406d-a6e9-dce42ed64331",
                "metadata": {},
                "source": [
                    '<a id="experiment-leaderboard"></a>\n',
                    "## Experiment leaderboard\n",
                    "\n",
                    "**Action:** Ensure the S3 connection is added to the workbench so the notebook can access the results.",  # noqa: E501
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "9be1f501-02e4-4107-906b-8f19448768bd",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import boto3\n",
                    "import os\n",
                    "from IPython.display import HTML\n",
                    "\n",
                    "s3 = boto3.resource('s3', endpoint_url=os.environ['AWS_S3_ENDPOINT'])\n",
                    "bucket = s3.Bucket(os.environ['AWS_S3_BUCKET'])\n",
                    "leaderboard_prefix = os.path.join(pipeline_name, run_id, 'leaderboard-evaluation')\n",
                    "leaderboard_artifact_name = 'html_artifact'\n",
                    "\n",
                    "for obj in bucket.objects.filter(Prefix=leaderboard_prefix):\n",
                    "    if leaderboard_artifact_name in obj.key:\n",
                    "        bucket.download_file(obj.key, leaderboard_artifact_name)\n",
                    "\n",
                    "HTML(leaderboard_artifact_name)",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "54525a94-7799-41cc-822e-91bae88b3b78",
                "metadata": {},
                "source": [
                    '<a id="download-trained-model"></a>\n',
                    "## Download trained model\n",
                    "\n",
                    "**Tip:** IF you want to download different model than the best one set `model_name` accordingly (must match a name from the leaderboard **model** column).",  # noqa: E501
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "55ce6aee-8c0c-445e-8b1b-62585ac7ddd6",
                "metadata": {},
                "outputs": [],
                "source": ['model_name = "<MODEL_NAME>"'],
            },
            {
                "cell_type": "markdown",
                "id": "fba16ca7-b15f-4d7a-95b4-d3cf73163440",
                "metadata": {},
                "source": ["Download model binaries and metrics."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "e88370df-ecda-453d-913a-9524088ccc36",
                "metadata": {},
                "outputs": [],
                "source": [
                    'full_refit_prefix = os.path.join(pipeline_name, run_id, "autogluon-models-full-refit")\n',
                    'best_model_subpath = os.path.join("model_artifact", model_name)\n',
                    "best_model_path = None\n",
                    "local_dir = None\n",
                    "\n",
                    "for obj in bucket.objects.filter(Prefix=full_refit_prefix):\n",
                    "    if best_model_subpath in obj.key:\n",
                    "        target = obj.key if local_dir is None else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))\n",  # noqa: E501
                    "        if not os.path.exists(os.path.dirname(target)):\n",
                    "            os.makedirs(os.path.dirname(target))\n",
                    "        if obj.key[-1] == '/':\n",
                    "            continue\n",
                    "        bucket.download_file(obj.key, target)\n",
                    "        best_model_path = os.path.join(obj.key.split(model_name)[0], model_name)\n",
                    "\n",
                    'print("Model artifact stored under", best_model_path)',
                ],
            },
            {
                "cell_type": "markdown",
                "id": "cf53ddb3-14af-44e2-9c5d-6636095cb2b5",
                "metadata": {},
                "source": [
                    '<a id="model-insights"></a>\n',
                    "## Model insights\n",
                    "\n",
                    "Display the features importances for selected model.",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "cc4f419b-2e28-406c-932b-de43182bef31",
                "metadata": {},
                "source": ["### Feature importance\n", "Top ten are displayed."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "0a7417fa-396f-4d83-ba20-1df01a3c0e2a",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "\n",
                    'feature_importance = pd.read_json(os.path.join(best_model_path, "metrics", "feature_importance.json"))\n',  # noqa: E501
                    "feature_importance.head(10)",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "6686ef6f-3251-43fa-bc9d-a9e911c7908c",
                "metadata": {},
                "source": [
                    '<a id="load-the-predictor"></a>\n',
                    "## Load the predictor\n",
                    "\n",
                    "Load the trained model as a TabularPredictor object.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "9ebc576f-17eb-49a7-8dcb-6b237dcc2218",
                "metadata": {},
                "outputs": [],
                "source": [
                    "from autogluon.tabular import TabularPredictor\n",
                    "\n",
                    "predictor = TabularPredictor.load(best_model_path)",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "57cc1e1a-707f-431f-9cb5-1d24e09d1249",
                "metadata": {},
                "outputs": [],
                "source": ["predictor.feature_metadata.to_dict()"],
            },
            {
                "cell_type": "markdown",
                "id": "064c76e4-1b44-4bba-8f2b-3178633a326a",
                "metadata": {},
                "source": [
                    '<a id="predict-the-values"></a>\n',
                    "## Predict the values\n",
                    "\n",
                    "Use sample records to predict values. ",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "d6955253-1891-4ff7-8b3e-ffa338d928f8",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "\n",
                    "score_data = <SAMPLE_ROW>\n",
                    "score_df = pd.DataFrame(data=score_data)\n",
                    "score_df.head()",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "f07e1d71-85e8-4484-877a-5af40547de4f",
                "metadata": {},
                "source": ["Predict the values using `predict` method."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "8441133b-2984-4ea9-92a1-4e427d25ee1b",
                "metadata": {},
                "outputs": [],
                "source": ["predictor.predict(score_df)"],
            },
            {
                "cell_type": "markdown",
                "id": "7ee6d313-4612-4fb9-bee2-b2dcc83772ef",
                "metadata": {},
                "source": [
                    '<a id="summary-and-next-steps"></a>\n',
                    "## Summary and next steps\n",
                    "\n",
                    "**Summary:** This notebook loaded a trained AutoGluon model from S3, displayed the experiment leaderboard, and ran predictions on sample data using `predict_proba`.\n",  # noqa: E501
                    "\n",
                    "**Next steps:**\n",
                    "- Run predictions on your own data (ensure columns match the training schema).\n",
                    "- Try another model from the leaderboard by changing `model_name` and re-running the download and load cells.\n",  # noqa: E501
                    "- Optionally create the Predictor online deployment using Kserve custom runtime.",
                ],
            },
            {"cell_type": "markdown", "id": "44a650c8-e5cc-4a2e-bebd-becd73944489", "metadata": {}, "source": ["---"]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    CLASSIFICATION = {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "a12d957a-c313-4e92-9578-44f6a48560d5",
                "metadata": {},
                "source": [
                    "![AutoML Banner](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxNzk2IDEwMCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTc5NiAxMDA7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbC1ydWxlOmV2ZW5vZGQ7Y2xpcC1ydWxlOmV2ZW5vZGQ7ZmlsbDp1cmwoI1NWR0lEXzFfKTt9Cgkuc3Qxe2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzdHJva2Utd2lkdGg6MjtzdHJva2UtbWl0ZXJsaW1pdDoxMDt9Cgkuc3Qye2ZpbGw6bm9uZTtzdHJva2U6I0ZGRkZGRjtzdHJva2Utd2lkdGg6MS41O3N0cm9rZS1taXRlcmxpbWl0OjEwO30KCS5zdDN7ZmlsbDojRkZGRkZGO30KCS5zdDR7Zm9udC1mYW1pbHk6J0hlbHZldGljYSBOZXVlJywgQXJpYWwsIHNhbnMtc2VyaWY7fQoJLnN0NXtmb250LXNpemU6MzJweDt9Cgkuc3Q2e2ZpbGw6IzNEM0QzRDt9Cgkuc3Q3e2ZpbGw6IzkzOTU5ODt9Cgkuc3Q4e29wYWNpdHk6MC4yO2ZpbGw6dXJsKCNTVkdJRF8yXyk7ZW5hYmxlLWJhY2tncm91bmQ6bmV3O30KCS5zdDl7Zm9udC13ZWlnaHQ6NTAwO30KPC9zdHlsZT4KPHJlY3Qgd2lkdGg9IjE3OTYiIGhlaWdodD0iMTAwIiBmaWxsPSIjMTYxNjE2Ii8+CjxsaW5lYXJHcmFkaWVudCBpZD0iU1ZHSURfMV8iIGdyYWRpZW50VW5pdHM9InVzZXJTcGFjZU9uVXNlIiB4MT0iNDIuODYiIHkxPSI1MCIgeDI9Ijc5LjcxIiB5Mj0iNTAiPgoJPHN0b3Agb2Zmc2V0PSIwIiBzdHlsZT0ic3RvcC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuMjEiIHN0eWxlPSJzdG9wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC43NSIgc3R5bGU9InN0b3AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjwhLS0gQXV0b01MIEljb24vTG9nbyBwbGFjZWhvbGRlciAtIHNpbXBsaWZpZWQgZ2VvbWV0cmljIHNoYXBlIC0tPgo8cGF0aCBjbGFzcz0ic3QwIiBkPSJNNTIuNCw0NS45YzAtMi4zLDEuOC00LjEsNC4xLTQuMXM0LjEsMS44LDQuMSw0LjFTNTguOCw1MCw1Ni41LDUwbDAsMGMtMi4yLDAuMS00LTEuNy00LjEtMy45CglDNTIuNCw0Niw1Mi40LDQ2LDUyLjQsNDUuOXogTTc3LjUsNTIuNWMtMC44LTEuMS0xLjQtMi4zLTEuOS0zLjVjMS4yLTQuNSwwLjctOC42LTEuOC0xMS45Yy0yLjktMy44LTguMi02LTE0LjUtNi4xCgljLTQuNS0wLjEtOC44LDEuNy0xMiw0LjhjLTMsMy00LjYsNy4yLTQuNSwxMS41Yy0wLjEsMi45LDAuOSw1LjgsMi43LDguMWMwLjgsMC44LDEuMywxLjksMS40LDN2NC41Yy0wLjgsMC41LTEuNCwxLjMtMS40LDIuMwoJYzAuMiwxLjUsMS41LDIuNiwzLDIuNGMxLjItMC4yLDIuMi0xLjEsMi40LTIuNGMwLTEtMC41LTEuOS0xLjQtMi4zdi00LjVjMC0yLTEtMy4zLTEuOS00LjZjLTEuNS0xLjktMi4yLTQuMi0yLjEtNi41CgljMC0zLjUsMS40LTYuOSwzLjgtOS40YzIuNy0yLjcsNi4zLTQuMSwxMC00LjFjNS41LDAsOS44LDEuOSwxMi4xLDVjMiwyLjgsMi41LDYuMywxLjQsOS42Yy0wLjQsMS4yLDAuNiwyLjcsMi4zLDUuNgoJYzAuNiwwLjksMS4yLDEuOSwxLjYsMi45Yy0wLjksMC43LTIsMS4yLTMuMSwxLjVjLTAuNSwwLjQtMC43LDAuOS0wLjgsMS41VjY1YzAsMC40LTAuMSwwLjgtMC40LDEuMWMtMC4zLDAuMi0wLjcsMC4zLTEuMSwwLjMKCWMtMS42LTAuMy0zLjQtMC43LTUuMi0xLjF2LTQuOGMwLjgtMC41LDEuNC0xLjQsMS40LTIuM2MwLTEuNS0xLjItMi43LTIuNy0yLjdzLTIuNywxLjItMi43LDIuN2MwLDEsMC41LDEuOSwxLjQsMi4zdjQuMQoJYy0wLjQtMC4xLTAuNy0wLjEtMS4xLTAuM2MtNC41LTEuMS00LjUtMi42LTQuNS0zLjR2LTguM2MzLjItMC43LDUuNC0zLjUsNS41LTYuN2MtMC4xLTMuOC0zLjMtNi43LTcuMS02LjZjLTMuNiwwLjEtNi40LDMtNi42LDYuNgoJYzAsMy4yLDIuMyw2LDUuNSw2Ljd2OC4zYzAsMiwwLjcsNC42LDYuNiw2LjFjMywwLjgsNiwxLjUsOS4xLDEuOWMwLjMsMCwwLjYsMC4xLDAuOCwwLjFjMSwwLDEuOS0wLjMsMi42LTEKCWMwLjktMC44LDEuNC0xLjksMS40LTMuMXYtNC41YzItMC44LDQuMS0yLDQuMS0zLjdDNzkuNyw1NS45LDc5LDU0LjYsNzcuNSw1Mi41eiIvPgo8Y2lyY2xlIGNsYXNzPSJzdDEiIGN4PSI1Ni41IiBjeT0iNDUuOSIgcj0iNS40Ii8+CjxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjQ4LjMiIGN5PSI2NSIgcj0iMS42Ii8+CjxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjY0LjgiIGN5PSI1OC4yIiByPSIxLjYiLz4KPHRleHQgdHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgMSAxMDEuMDIgNTkuMzMpIiBjbGFzcz0ic3QzIHN0NCBzdDUiPkF1dG9NTDwvdGV4dD4KPHJlY3QgeD0iMjMxLjEiIHk9IjM0IiBjbGFzcz0ic3Q2IiB3aWR0aD0iMSIgaGVpZ2h0PSIzMiIvPgo8dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDI1Ni4yOSA1OS42NikiIGNsYXNzPSJzdDcgc3Q0IHN0NSI+UGFydCBvZiBSZWQgSGF0IE9wZW5TaGlmdCBBSTwvdGV4dD4KPGxpbmVhckdyYWRpZW50IGlkPSJTVkdJRF8yXyIgZ3JhZGllbnRVbml0cz0idXNlclNwYWNlT25Vc2UiIHgxPSI3NzMuOCIgeTE9IjUwIiB4Mj0iMTc5NiIgeTI9IjUwIj4KCTxzdG9wIG9mZnNldD0iMCIgc3R5bGU9InN0b3AtY29sb3I6IzE2MTYxNiIvPgoJPHN0b3Agb2Zmc2V0PSIwLjUyIiBzdHlsZT0ic3RvcC1jb2xvcjojRkY2QjZCIi8+Cgk8c3RvcCBvZmZzZXQ9IjAuNjIiIHN0eWxlPSJzdG9wLWNvbG9yOiNFRTAwMDAiLz4KCTxzdG9wIG9mZnNldD0iMC44OCIgc3R5bGU9InN0b3AtY29sb3I6I0NDMDAwMCIvPgoJPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb2xvcjojQUEwMDAwIi8+CjwvbGluZWFyR3JhZGllbnQ+CjxyZWN0IHg9Ijc3My44IiBjbGFzcz0ic3Q4IiB3aWR0aD0iMTAyMi4yIiBoZWlnaHQ9IjEwMCIvPgo8dGV4dCB0cmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAxIDE0NDguMTY0MSA1OS40NikiIGNsYXNzPSJzdDMgc3Q0IHN0NSBzdDkiPlByZWRpY3RvciBOb3RlYm9vazwvdGV4dD4KPC9zdmc+Cg==)"  # noqa: E501
                ],
            },
            {
                "cell_type": "markdown",
                "id": "0e9aa72f",
                "metadata": {},
                "source": [
                    "## Notebook content\n",
                    "\n",
                    "This notebook lets you review the experiment leaderboard for insights into trained model evaluation quality, load a chosen AutoGluon model from S3, and run predictions. \n",  # noqa: E501
                    "\n",
                    "\n",
                    "**Tips:**\n",
                    "- Ensure the S3 connection to pipeline run results is configured so the notebook can access run artifacts.\n",  # noqa: E501
                    "- The model name must match one of the models listed in the leaderboard (the **model** column).\n",
                    "\n",
                    "### Contents\n",
                    "This notebook contains the following parts:\n",
                    "\n",
                    "**[Setup](#setup)**  \n",
                    "**[Experiment run details](#experiment-run-details)**  \n",
                    "**[Experiment leaderboard](#experiment-leaderboard)**  \n",
                    "**[Download trained model](#download-trained-model)**  \n",
                    "**[Model insights](#model-insights)**  \n",
                    "**[Load the predictor](#load-the-predictor)**  \n",
                    "**[Predict the values](#predict-the-values)**  \n",
                    "**[Summary and next steps](#summary-and-next-steps)**",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "a7d9cf2b-18cc-4ac9-87af-a74f8bf60322",
                "metadata": {},
                "source": ['<a id="setup"></a>\n', "## Setup"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "5bacd972",
                "metadata": {},
                "outputs": [],
                "source": ["import warnings\n", "\n", 'warnings.filterwarnings("ignore")'],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cec84205-8ee9-4aaf-a97e-4ef576e7b9da",
                "metadata": {},
                "outputs": [],
                "source": ["%pip install autogluon.tabular[all]==1.5 | tail -n 1"],
            },
            {
                "cell_type": "markdown",
                "id": "e8ff506e-f1a3-4990-a979-7790a5105251",
                "metadata": {},
                "source": [
                    '<a id="experiment-run-details"></a>\n',
                    "## Experiment run details\n",
                    "\n",
                    "Set the pipeline name, run name, and run ID that identify the training run whose artifacts you want to load. These values are typically available from the pipeline run or workbench.",  # noqa: E501
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "fa7f736d-0b5c-4988-87a5-4d1a5cde0873",
                "metadata": {},
                "outputs": [],
                "source": ['pipeline_name = "<PIPELINE_NAMNE>"\n', 'run_id = "<RUN_ID>"'],
            },
            {
                "cell_type": "markdown",
                "id": "00cc5969-0f9b-406d-a6e9-dce42ed64331",
                "metadata": {},
                "source": [
                    '<a id="experiment-leaderboard"></a>\n',
                    "## Experiment leaderboard\n",
                    "\n",
                    "**Action:** Ensure the S3 connection is added to the workbench so the notebook can access the results.",  # noqa: E501
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "9be1f501-02e4-4107-906b-8f19448768bd",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import boto3\n",
                    "import os\n",
                    "from IPython.display import HTML\n",
                    "\n",
                    "s3 = boto3.resource('s3', endpoint_url=os.environ['AWS_S3_ENDPOINT'])\n",
                    "bucket = s3.Bucket(os.environ['AWS_S3_BUCKET'])\n",
                    "leaderboard_prefix = os.path.join(pipeline_name, run_id, 'leaderboard-evaluation')\n",
                    "leaderboard_artifact_name = 'html_artifact'\n",
                    "\n",
                    "for obj in bucket.objects.filter(Prefix=leaderboard_prefix):\n",
                    "    if leaderboard_artifact_name in obj.key:\n",
                    "        bucket.download_file(obj.key, leaderboard_artifact_name)\n",
                    "\n",
                    "HTML(leaderboard_artifact_name)",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "54525a94-7799-41cc-822e-91bae88b3b78",
                "metadata": {},
                "source": [
                    '<a id="download-trained-model"></a>\n',
                    "## Download trained model\n",
                    "\n",
                    "**Tip:** IF you want to download different model than the best one set `model_name` accordingly (must match a name from the leaderboard **model** column).",  # noqa: E501
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "55ce6aee-8c0c-445e-8b1b-62585ac7ddd6",
                "metadata": {},
                "outputs": [],
                "source": ['model_name = "<MODEL_NAME>"'],
            },
            {
                "cell_type": "markdown",
                "id": "fba16ca7-b15f-4d7a-95b4-d3cf73163440",
                "metadata": {},
                "source": ["Download model binaries and metrics."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "e88370df-ecda-453d-913a-9524088ccc36",
                "metadata": {},
                "outputs": [],
                "source": [
                    'full_refit_prefix = os.path.join(pipeline_name, run_id, "autogluon-models-full-refit")\n',
                    'best_model_subpath = os.path.join("model_artifact", model_name)\n',
                    "best_model_path = None\n",
                    "local_dir = None\n",
                    "\n",
                    "for obj in bucket.objects.filter(Prefix=full_refit_prefix):\n",
                    "    if best_model_subpath in obj.key:\n",
                    "        target = obj.key if local_dir is None else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))\n",  # noqa: E501
                    "        if not os.path.exists(os.path.dirname(target)):\n",
                    "            os.makedirs(os.path.dirname(target))\n",
                    "        if obj.key[-1] == '/':\n",
                    "            continue\n",
                    "        bucket.download_file(obj.key, target)\n",
                    "        best_model_path = os.path.join(obj.key.split(model_name)[0], model_name)\n",
                    "\n",
                    'print("Model artifact stored under", best_model_path)',
                ],
            },
            {
                "cell_type": "markdown",
                "id": "cf53ddb3-14af-44e2-9c5d-6636095cb2b5",
                "metadata": {},
                "source": [
                    '<a id="model-insights"></a>\n',
                    "## Model insights\n",
                    "\n",
                    "Display the confusion matrix and features importances for selected model.",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "72345abd-b419-4f63-8b0b-6d023ddae73b",
                "metadata": {},
                "source": ["### Confusion matrix"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "bd38da24-a764-48e8-9c0c-9285e5810fe1",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "\n",
                    'confusion_matrix = pd.read_json(os.path.join(best_model_path, "metrics", "confusion_matrix.json"))\n',  # noqa: E501
                    "confusion_matrix.head()",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "cc4f419b-2e28-406c-932b-de43182bef31",
                "metadata": {},
                "source": ["### Feature importance\n", "Top ten are displayed."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "0a7417fa-396f-4d83-ba20-1df01a3c0e2a",
                "metadata": {},
                "outputs": [],
                "source": [
                    'feature_importance = pd.read_json(os.path.join(best_model_path, "metrics", "feature_importance.json"))\n',  # noqa: E501
                    "feature_importance.head(10)",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "6686ef6f-3251-43fa-bc9d-a9e911c7908c",
                "metadata": {},
                "source": [
                    '<a id="load-the-predictor"></a>\n',
                    "## Load the predictor\n",
                    "\n",
                    "Load the trained model as a TabularPredictor object.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "9ebc576f-17eb-49a7-8dcb-6b237dcc2218",
                "metadata": {},
                "outputs": [],
                "source": [
                    "from autogluon.tabular import TabularPredictor\n",
                    "\n",
                    "predictor = TabularPredictor.load(best_model_path)",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "064c76e4-1b44-4bba-8f2b-3178633a326a",
                "metadata": {},
                "source": [
                    '<a id="predict-the-values"></a>\n',
                    "## Predict the values\n",
                    "\n",
                    "Use sample records to predict values. ",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "d6955253-1891-4ff7-8b3e-ffa338d928f8",
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "\n",
                    "score_data = <SAMPLE_ROW>\n",
                    "\n",
                    "score_df = pd.DataFrame(data=score_data)\n",
                    "score_df.head()",
                ],
            },
            {
                "cell_type": "markdown",
                "id": "f07e1d71-85e8-4484-877a-5af40547de4f",
                "metadata": {},
                "source": ["Predict the values using `predict_proba` method."],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "8441133b-2984-4ea9-92a1-4e427d25ee1b",
                "metadata": {},
                "outputs": [],
                "source": ["predictor.predict_proba(score_df)"],
            },
            {
                "cell_type": "markdown",
                "id": "7ee6d313-4612-4fb9-bee2-b2dcc83772ef",
                "metadata": {},
                "source": [
                    '<a id="summary-and-next-steps"></a>\n',
                    "## Summary and next steps\n",
                    "\n",
                    "**Summary:** This notebook loaded a trained AutoGluon model from S3, displayed the experiment leaderboard, and ran predictions on sample data using `predict_proba`.\n",  # noqa: E501
                    "\n",
                    "**Next steps:**\n",
                    "- Run predictions on your own data (ensure columns match the training schema).\n",
                    "- Try another model from the leaderboard by changing `model_name` and re-running the download and load cells.\n",  # noqa: E501
                    "- Optionally create the Predictor online deployment using Kserve custom runtime.",
                ],
            },
            {"cell_type": "markdown", "id": "44a650c8-e5cc-4a2e-bebd-becd73944489", "metadata": {}, "source": ["---"]},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3.12", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    match problem_type:
        case "regression":
            notebook = REGRESSION
        case "binary" | "multiclass":
            notebook = CLASSIFICATION
        case _:
            raise ValueError(f"Invalid problem type: {problem_type}")

    def retrieve_pipeline_name(pipeline_name: str) -> str:
        pipeline_name_elements = pipeline_name.split("-")
        return "-".join(pipeline_name_elements[:-1])

    pipeline_name = retrieve_pipeline_name(pipeline_name)

    notebook["cells"][6]["source"][1] = notebook["cells"][6]["source"][1].replace("<RUN_ID>", run_id)
    notebook["cells"][6]["source"][0] = notebook["cells"][6]["source"][0].replace("<PIPELINE_NAME>", pipeline_name)
    notebook["cells"][10]["source"][0] = notebook["cells"][10]["source"][0].replace("<MODEL_NAME>", model_name)

    sample_row = json.loads(sample_row)
    sample_row.pop(label_column, None)
    sample_row_idx = 20 + int((problem_type in {"binary", "multiclass"}))
    notebook["cells"][sample_row_idx]["source"][2] = notebook["cells"][sample_row_idx]["source"][2].replace(
        "<SAMPLE_ROW>", json.dumps(sample_row, indent=2)
    )
    path = Path(notebook_artifact.path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "automl_predictor_notebook.ipynb").open("w", encoding="utf-8") as f:
        json.dump(notebook, f)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        notebook_generation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
