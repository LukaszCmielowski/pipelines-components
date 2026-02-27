from typing import List, NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc",  # noqa: E501
)
def leaderboard_evaluation(
    models: List[dsl.Model],
    eval_metric: str,
    html_artifact: dsl.Output[dsl.HTML],
) -> NamedTuple("outputs", best_model=str):
    """Evaluate multiple AutoGluon models and generate a leaderboard.

    This component aggregates evaluation results from a list of Model artifacts
    (reading pre-computed metrics from JSON) and generates an HTML-formatted
    leaderboard ranking the models by their performance metrics. Each model
    artifact is expected to contain metrics at
    model.path / model.metadata["display_name"] / metrics / metrics.json.

    Args:
        models: A list of Model artifacts. Each should have metadata containing
            a "display_name" field and metrics file at
            model.path / model_name / metrics / metrics.json.
        eval_metric: The name of the evaluation metric to use for ranking.
            Must match a key in the metrics JSON (e.g., "accuracy" for
            classification, "root_mean_squared_error" for regression).
            The leaderboard is sorted by this metric in descending order.
        html_artifact: Output artifact where the HTML-formatted leaderboard
            will be written. The leaderboard contains model names and their
            evaluation metrics.

    Raises:
        FileNotFoundError: If any model metrics path cannot be found.
        KeyError: If model metadata does not contain "display_name" or the
            metrics JSON does not contain the eval_metric key.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_leaderboard_evaluation import (
            leaderboard_evaluation
        )

        @dsl.pipeline(name="model-evaluation-pipeline")
        def evaluation_pipeline(trained_models):
            leaderboard = leaderboard_evaluation(
                models=trained_models,
                eval_metric="root_mean_squared_error",
            )
            return leaderboard
    """
    import json
    import re
    from pathlib import Path

    import pandas as pd

    def _build_leaderboard_html(table_html: str, eval_metric: str, best_model_name: str, num_models: int) -> str:
        """Build a styled HTML document for the leaderboard."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoML Leaderboard</title>
  <style>
    :root {{
      --bg: #0f1419;
      --surface: #1a2332;
      --surface-hover: #243044;
      --border: #2d3a4f;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-dim: #388bfd66;
      --gold: #f0b429;
      --silver: #a8b2c1;
      --bronze: #c9a227;
      --success: #3fb950;
      --radius: 12px;
      --font: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      padding: 2rem;
      font-family: var(--font);
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      min-height: 100vh;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
    }}
    header {{
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    h1 {{
      margin: 0 0 0.25rem 0;
      font-size: 1.75rem;
      font-weight: 600;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 0.9rem;
    }}
    .badge {{
      display: inline-block;
      margin-left: 0.5rem;
      padding: 0.2rem 0.5rem;
      font-size: 0.75rem;
      font-weight: 600;
      border-radius: 6px;
      background: var(--accent-dim);
      color: var(--accent);
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }}
    .leaderboard-wrap {{
      overflow-x: auto;
    }}
    .leaderboard-wrap table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    .leaderboard-wrap th {{
      text-align: left;
      padding: 1rem 1.25rem;
      background: var(--surface-hover);
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border);
    }}
    .leaderboard-wrap td {{
      padding: 1rem 1.25rem;
      border-bottom: 1px solid var(--border);
    }}
    .leaderboard-wrap tr:last-child td {{
      border-bottom: none;
    }}
    .leaderboard-wrap tbody tr:hover {{
      background: var(--surface-hover);
    }}
    .leaderboard-wrap tbody tr.rank-1 {{
      background: linear-gradient(90deg, rgba(240,180,41,0.12) 0%, transparent 100%);
    }}
    .leaderboard-wrap tbody tr.rank-1 td:first-child {{
      color: var(--gold);
      font-weight: 700;
    }}
    .leaderboard-wrap .rank-cell {{
      font-weight: 600;
      color: var(--text-muted);
      width: 4rem;
    }}
    .leaderboard-wrap .metric-cell {{
      font-variant-numeric: tabular-nums;
      color: var(--success);
    }}
    .best-model-footer {{
      margin-top: 1.5rem;
      padding: 1rem 1.25rem;
      background: var(--surface-hover);
      border-radius: 8px;
      font-size: 0.9rem;
      color: var(--text-muted);
    }}
    .best-model-footer strong {{
      color: var(--gold);
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>AutoML Leaderboard</h1>
      <p class="subtitle">
        Ranked by <span class="badge">{eval_metric}</span> · {num_models} model(s)
      </p>
    </header>
    <div class="card">
      <div class="leaderboard-wrap">
        {table_html}
      </div>
    </div>
    <div class="best-model-footer">
      Best model: <strong>{best_model_name}</strong>
    </div>
  </div>
</body>
</html>"""

    results = []
    for model in models:
        eval_results = json.load(
            (Path(model.path) / model.metadata["display_name"] / "metrics" / "metrics.json").open("r")
        )
        results.append({"model": model.metadata["display_name"]} | eval_results)

    leaderboard_df = pd.DataFrame(results).sort_values(by=eval_metric, ascending=False)
    leaderboard_df.index = range(1, len(leaderboard_df) + 1)
    leaderboard_df.index.name = "rank"

    html_table = leaderboard_df.to_html(classes=None, border=0, escape=True)
    # Highlight first (best) row: add class to first tbody tr
    html_table = re.sub(r"(<tbody>\s*)<tr>", r'\1<tr class="rank-1">', html_table, count=1)

    best_model_name = leaderboard_df.iloc[0]["model"]
    html_content = _build_leaderboard_html(
        table_html=html_table,
        eval_metric=eval_metric,
        best_model_name=best_model_name,
        num_models=len(leaderboard_df),
    )
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)

    html_artifact.metadata["data"] = leaderboard_df.to_dict()
    html_artifact.metadata["display_name"] = "automl_leaderboard"
    return NamedTuple("outputs", best_model=str)(best_model=best_model_name)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
