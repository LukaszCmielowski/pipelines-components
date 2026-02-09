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
            Expected format: '[{"col1": "val1","col2":"val2"},{"col1":"val3","col2":"val4"}]'
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
                    """<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 1796 100" style="enable-background:new 0 0 1796 100;" xml:space="preserve" width="100%">
<style type="text/css">
	.st0{fill-rule:evenodd;clip-rule:evenodd;fill:url(#SVGID_1_);}
	.st1{fill:none;stroke:#FFFFFF;stroke-width:2;stroke-miterlimit:10;}
	.st2{fill:none;stroke:#FFFFFF;stroke-width:1.5;stroke-miterlimit:10;}
	.st3{fill:#FFFFFF;}
	.st4{font-family:'Helvetica Neue', Arial, sans-serif;}
	.st5{font-size:32px;}
	.st6{fill:#3D3D3D;}
	.st7{fill:#939598;}
	.st8{opacity:0.2;fill:url(#SVGID_2_);enable-background:new;}
	.st9{font-weight:500;}
</style>
<rect width="1796" height="100" fill="#161616"/>
<linearGradient id="SVGID_1_" gradientUnits="userSpaceOnUse" x1="42.86" y1="50" x2="79.71" y2="50">
	<stop offset="0" style="stop-color:#FF6B6B"/>
	<stop offset="0.21" style="stop-color:#EE0000"/>
	<stop offset="0.75" style="stop-color:#CC0000"/>
	<stop offset="1" style="stop-color:#AA0000"/>
</linearGradient>
<path class="st0" d="M52.4,45.9c0-2.3,1.8-4.1,4.1-4.1s4.1,1.8,4.1,4.1S58.8,50,56.5,50l0,0c-2.2,0.1-4-1.7-4.1-3.9
	C52.4,46,52.4,46,52.4,45.9z M77.5,52.5c-0.8-1.1-1.4-2.3-1.9-3.5c1.2-4.5,0.7-8.6-1.8-11.9c-2.9-3.8-8.2-6-14.5-6.1
	c-4.5-0.1-8.8,1.7-12,4.8c-3,3-4.6,7.2-4.5,11.5c-0.1,2.9,0.9,5.8,2.7,8.1c0.8,0.8,1.3,1.9,1.4,3v4.5c-0.8,0.5-1.4,1.3-1.4,2.3
	c0.2,1.5,1.5,2.6,3,2.4c1.2-0.2,2.2-1.1,2.4-2.4c0-1-0.5-1.9-1.4-2.3v-4.5c0-2-1-3.3-1.9-4.6c-1.5-1.9-2.2-4.2-2.1-6.5
	c0-3.5,1.4-6.9,3.8-9.4c2.7-2.7,6.3-4.1,10-4.1c5.5,0,9.8,1.9,12.1,5c2,2.8,2.5,6.3,1.4,9.6c-0.4,1.2,0.6,2.7,2.3,5.6
	c0.6,0.9,1.2,1.9,1.6,2.9c-0.9,0.7-2,1.2-3.1,1.5c-0.5,0.4-0.7,0.9-0.8,1.5V65c0,0.4-0.1,0.8-0.4,1.1c-0.3,0.2-0.7,0.3-1.1,0.3
	c-1.6-0.3-3.4-0.7-5.2-1.1v-4.8c0.8-0.5,1.4-1.4,1.4-2.3c0-1.5-1.2-2.7-2.7-2.7s-2.7,1.2-2.7,2.7c0,1,0.5,1.9,1.4,2.3v4.1
	c-0.4-0.1-0.7-0.1-1.1-0.3c-4.5-1.1-4.5-2.6-4.5-3.4v-8.3c3.2-0.7,5.4-3.5,5.5-6.7c-0.1-3.8-3.3-6.7-7.1-6.6c-3.6,0.1-6.4,3-6.6,6.6
	c0,3.2,2.3,6,5.5,6.7v8.3c0,2,0.7,4.6,6.6,6.1c3,0.8,6,1.5,9.1,1.9c0.3,0,0.6,0.1,0.8,0.1c1,0,1.9-0.3,2.6-1
	c0.9-0.8,1.4-1.9,1.4-3.1v-4.5c2-0.8,4.1-2,4.1-3.7C79.7,55.9,79,54.6,77.5,52.5z"/>
<circle class="st1" cx="56.5" cy="45.9" r="5.4"/>
<circle class="st2" cx="48.3" cy="65" r="1.6"/>
<circle class="st2" cx="64.8" cy="58.2" r="1.6"/>
<text transform="matrix(1 0 0 1 101.02 59.33)" class="st3 st4 st5">AutoML</text>
<rect x="231.1" y="34" class="st6" width="1" height="32"/>
<text transform="matrix(1 0 0 1 256.29 59.66)" class="st7 st4 st5">Part of Red Hat OpenShift AI</text>
<linearGradient id="SVGID_2_" gradientUnits="userSpaceOnUse" x1="773.8" y1="50" x2="1796" y2="50">
	<stop offset="0" style="stop-color:#161616"/>
	<stop offset="0.52" style="stop-color:#FF6B6B"/>
	<stop offset="0.62" style="stop-color:#EE0000"/>
	<stop offset="0.88" style="stop-color:#CC0000"/>
	<stop offset="1" style="stop-color:#AA0000"/>
</linearGradient>
<rect x="773.8" class="st8" width="1022.2" height="100"/>
<text transform="matrix(1 0 0 1 1448.1641 59.46)" class="st3 st4 st5 st9">Predictor Notebook</text>
</svg>"""  # noqa: E501
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
                    " \U0001f4a1 **Tips:**\n",
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
                    " \U0001f4cc **Action:** Ensure the S3 connection is added to the workbench so the notebook can access the results.",  # noqa: E501
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
                    " \U0001f4a1 **Tip:** IF you want to download different model than the best one set `model_name` accordingly (must match a name from the leaderboard **model** column).",  # noqa: E501
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
                    """<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" viewBox="0 0 1796 100" style="enable-background:new 0 0 1796 100;" xml:space="preserve" width="100%">
<style type="text/css">
	.st0{fill-rule:evenodd;clip-rule:evenodd;fill:url(#SVGID_1_);}
	.st1{fill:none;stroke:#FFFFFF;stroke-width:2;stroke-miterlimit:10;}
	.st2{fill:none;stroke:#FFFFFF;stroke-width:1.5;stroke-miterlimit:10;}
	.st3{fill:#FFFFFF;}
	.st4{font-family:'Helvetica Neue', Arial, sans-serif;}
	.st5{font-size:32px;}
	.st6{fill:#3D3D3D;}
	.st7{fill:#939598;}
	.st8{opacity:0.2;fill:url(#SVGID_2_);enable-background:new;}
	.st9{font-weight:500;}
</style>
<rect width="1796" height="100" fill="#161616"/>
<linearGradient id="SVGID_1_" gradientUnits="userSpaceOnUse" x1="42.86" y1="50" x2="79.71" y2="50">
	<stop offset="0" style="stop-color:#FF6B6B"/>
	<stop offset="0.21" style="stop-color:#EE0000"/>
	<stop offset="0.75" style="stop-color:#CC0000"/>
	<stop offset="1" style="stop-color:#AA0000"/>
</linearGradient>
<path class="st0" d="M52.4,45.9c0-2.3,1.8-4.1,4.1-4.1s4.1,1.8,4.1,4.1S58.8,50,56.5,50l0,0c-2.2,0.1-4-1.7-4.1-3.9
	C52.4,46,52.4,46,52.4,45.9z M77.5,52.5c-0.8-1.1-1.4-2.3-1.9-3.5c1.2-4.5,0.7-8.6-1.8-11.9c-2.9-3.8-8.2-6-14.5-6.1
	c-4.5-0.1-8.8,1.7-12,4.8c-3,3-4.6,7.2-4.5,11.5c-0.1,2.9,0.9,5.8,2.7,8.1c0.8,0.8,1.3,1.9,1.4,3v4.5c-0.8,0.5-1.4,1.3-1.4,2.3
	c0.2,1.5,1.5,2.6,3,2.4c1.2-0.2,2.2-1.1,2.4-2.4c0-1-0.5-1.9-1.4-2.3v-4.5c0-2-1-3.3-1.9-4.6c-1.5-1.9-2.2-4.2-2.1-6.5
	c0-3.5,1.4-6.9,3.8-9.4c2.7-2.7,6.3-4.1,10-4.1c5.5,0,9.8,1.9,12.1,5c2,2.8,2.5,6.3,1.4,9.6c-0.4,1.2,0.6,2.7,2.3,5.6
	c0.6,0.9,1.2,1.9,1.6,2.9c-0.9,0.7-2,1.2-3.1,1.5c-0.5,0.4-0.7,0.9-0.8,1.5V65c0,0.4-0.1,0.8-0.4,1.1c-0.3,0.2-0.7,0.3-1.1,0.3
	c-1.6-0.3-3.4-0.7-5.2-1.1v-4.8c0.8-0.5,1.4-1.4,1.4-2.3c0-1.5-1.2-2.7-2.7-2.7s-2.7,1.2-2.7,2.7c0,1,0.5,1.9,1.4,2.3v4.1
	c-0.4-0.1-0.7-0.1-1.1-0.3c-4.5-1.1-4.5-2.6-4.5-3.4v-8.3c3.2-0.7,5.4-3.5,5.5-6.7c-0.1-3.8-3.3-6.7-7.1-6.6c-3.6,0.1-6.4,3-6.6,6.6
	c0,3.2,2.3,6,5.5,6.7v8.3c0,2,0.7,4.6,6.6,6.1c3,0.8,6,1.5,9.1,1.9c0.3,0,0.6,0.1,0.8,0.1c1,0,1.9-0.3,2.6-1
	c0.9-0.8,1.4-1.9,1.4-3.1v-4.5c2-0.8,4.1-2,4.1-3.7C79.7,55.9,79,54.6,77.5,52.5z"/>
<circle class="st1" cx="56.5" cy="45.9" r="5.4"/>
<circle class="st2" cx="48.3" cy="65" r="1.6"/>
<circle class="st2" cx="64.8" cy="58.2" r="1.6"/>
<text transform="matrix(1 0 0 1 101.02 59.33)" class="st3 st4 st5">AutoML</text>
<rect x="231.1" y="34" class="st6" width="1" height="32"/>
<text transform="matrix(1 0 0 1 256.29 59.66)" class="st7 st4 st5">Part of Red Hat OpenShift AI</text>
<linearGradient id="SVGID_2_" gradientUnits="userSpaceOnUse" x1="773.8" y1="50" x2="1796" y2="50">
	<stop offset="0" style="stop-color:#161616"/>
	<stop offset="0.52" style="stop-color:#FF6B6B"/>
	<stop offset="0.62" style="stop-color:#EE0000"/>
	<stop offset="0.88" style="stop-color:#CC0000"/>
	<stop offset="1" style="stop-color:#AA0000"/>
</linearGradient>
<rect x="773.8" class="st8" width="1022.2" height="100"/>
<text transform="matrix(1 0 0 1 1448.1641 59.46)" class="st3 st4 st5 st9">Predictor Notebook</text>
</svg>"""  # noqa: E501
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
                    " \U0001f4a1 **Tips:**\n",
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
                    " \U0001f4cc **Action:** Ensure the S3 connection is added to the workbench so the notebook can access the results.",  # noqa: E501
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
                    " \U0001f4a1 **Tip:** IF you want to download different model than the best one set `model_name` accordingly (must match a name from the leaderboard **model** column).",  # noqa: E501
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
    sample_row = [{col: value for col, value in row.items() if col != label_column} for row in sample_row]

    sample_row_idx = 20 + int((problem_type in {"binary", "multiclass"}))
    notebook["cells"][sample_row_idx]["source"][2] = notebook["cells"][sample_row_idx]["source"][2].replace(
        "<SAMPLE_ROW>", str(sample_row)
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
