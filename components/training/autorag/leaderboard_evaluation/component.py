from kfp import dsl


@dsl.component(
    base_image="quay.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9:rhoai-3.2",
)
def leaderboard_evaluation(
    rag_patterns: dsl.InputPath(dsl.Artifact),
    html_artifact: dsl.Output[dsl.HTML],
):
    """
    Build an HTML leaderboard artifact from RAG pattern evaluation results.

    Reads pattern.json from each subdirectory of rag_patterns (produced by
    rag_templates_optimization) and generates a single HTML table with RAG
    pattern names, settings, and metric values. Writes the HTML directly
    to html_artifact.path (single file at artifact path, same as autogluon
    leaderboard_evaluation).

    Args:
        rag_patterns
            Path to the directory of RAG patterns; each subdir contains
            pattern.json with evaluation result (pattern_name, indexing_params,
            rag_params, scores, execution_time, final_score).
        html_artifact
            Output HTML artifact; the leaderboard table is written to
            html_artifact.path (single file).
    """
    import html
    import json
    from pathlib import Path

    rag_patterns_dir = Path(rag_patterns)
    if not rag_patterns_dir.is_dir():
        raise FileNotFoundError("rag_patterns path is not a directory: %s" % rag_patterns_dir)

    evaluations = []
    for subdir in sorted(rag_patterns_dir.iterdir()):
        if not subdir.is_dir():
            continue
        pattern_file = subdir / "pattern.json"
        if not pattern_file.is_file():
            continue
        with pattern_file.open("r", encoding="utf-8") as f:
            evaluations.append(json.load(f))

    # Sort by final_score descending (best first); missing final_score last
    def _final_score(e):
        v = e.get("final_score")
        return (v is None, -(v if v is not None else float("-inf")))

    evaluations.sort(key=_final_score)

    def _format_settings(params: dict) -> str:
        if not params:
            return ""
        return "<br />".join(
            "%s: %s" % (html.escape(str(k)), html.escape(str(v)))
            for k, v in sorted(params.items())
        )

    def _scores_columns(evals: list) -> list:
        cols = []
        for e in evals:
            scores = e.get("scores") or {}
            for metric in (scores.get("scores") or {}).keys():
                if metric not in cols:
                    cols.append(metric)
        return cols

    metric_columns = _scores_columns(evaluations)
    rows = []
    for rank, e in enumerate(evaluations, start=1):
        pattern_name = e.get("pattern_name", "—")
        scores = e.get("scores") or {}
        metric_cells = []
        for m in metric_columns:
            info = (scores.get("scores") or {}).get(m, {})
            mean = info.get("mean", "")
            ci_low = info.get("ci_low")
            ci_high = info.get("ci_high")
            if ci_low is not None and ci_high is not None:
                cell = "%s [%s, %s]" % (mean, ci_low, ci_high)
            else:
                cell = str(mean) if mean != "" else "—"
            metric_cells.append("<td>%s</td>" % html.escape(str(cell)))

        indexing_params = e.get("indexing_params") or {}
        rag_params = e.get("rag_params") or {}
        settings_str = _format_settings({**indexing_params, **rag_params})
        exec_time = e.get("execution_time")
        final_score = e.get("final_score")

        rows.append(
            "<tr><td>%d</td><td>%s</td><td>%s</td>%s<td>%s</td><td>%s</td></tr>"
            % (
                rank,
                html.escape(str(pattern_name)),
                settings_str or "—",
                "".join(metric_cells),
                str(exec_time) if exec_time is not None else "—",
                str(final_score) if final_score is not None else "—",
            )
        )

    metric_headers = "".join("<th>%s</th>" % html.escape(m) for m in metric_columns)
    table_body = "".join(rows)
    html_content = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RAG Patterns Leaderboard</title>
  <style>
    table { border-collapse: collapse; }
    th, td { border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }
    th { background: #f5f5f5; }
  </style>
</head>
<body>
  <h1>RAG Patterns Leaderboard</h1>
  <p>Patterns ordered by optimization metric (best first).</p>
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>RAG Pattern</th>
        <th>Settings</th>
        %s
        <th>Execution time</th>
        <th>Final score</th>
      </tr>
    </thead>
    <tbody>
      %s
    </tbody>
  </table>
</body>
</html>
""" % (
        metric_headers,
        table_body,
    )

    # Write HTML directly to artifact path (single file at path, same as autogluon)
    Path(html_artifact.path).parent.mkdir(parents=True, exist_ok=True)
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
