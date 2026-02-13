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
    rag_templates_optimization) and generates a single HTML table: Pattern_Name,
    mean_* metrics (e.g. mean_answer_correctness, mean_faithfulness), then
    config columns (chunking.*, embeddings.model_id, retrieval.*,
    generation.model_id). Writes the HTML to html_artifact.path.

    Args:
        rag_patterns
            Path to the directory of RAG patterns; each subdir contains
            pattern.json (pattern_name, indexing_params, rag_params, scores,
            execution_time, final_score).
        html_artifact
            Output HTML artifact; the leaderboard table is written to
            html_artifact.path (single file).
    """
    import html
    import json
    from pathlib import Path

    def _get_nested(params: dict, key: str):
        """Resolve dotted key from flat or nested dict (e.g. chunking.method)."""
        if not params:
            return None
        if key in params:
            return params[key]
        parts = key.split(".", 1)
        if len(parts) == 2:
            outer = params.get(parts[0])
            if isinstance(outer, dict):
                return outer.get(parts[1])
        return None

    def _merge_params(indexing_params: dict, rag_params: dict) -> dict:
        merged = dict(indexing_params or {})
        merged.update(rag_params or {})
        return merged

    def _metric_to_mean_key(metric: str) -> str:
        return "mean_" + metric

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

    # Default column order: metrics first, then RAG config (chunking, embeddings, retrieval, generation).
    leaderboard_metric_columns = [
        "mean_answer_correctness",
        "mean_faithfulness",
        "mean_context_correctness",
    ]
    leaderboard_config_columns = [
        "chunking.method",
        "chunking.chunk_size",
        "chunking.chunk_overlap",
        "embeddings.model_id",
        "retrieval.method",
        "retrieval.number_of_chunks",
        "generation.model_id",
    ]
    # Build metric columns present in data (preferred order above)
    all_metric_names = []
    for e in evaluations:
        for m in (e.get("scores") or {}).get("scores") or {}:
            if m not in all_metric_names:
                all_metric_names.append(m)
    metric_columns = [c for c in leaderboard_metric_columns if c.replace("mean_", "", 1) in all_metric_names]
    for m in all_metric_names:
        col = _metric_to_mean_key(m)
        if col not in metric_columns:
            metric_columns.append(col)

    config_columns = list(leaderboard_config_columns)
    headers = ["Pattern_Name"] + metric_columns + config_columns
    header_row = "".join("<th>%s</th>" % html.escape(h) for h in headers)

    rows = []
    for e in evaluations:
        pattern_name = e.get("pattern_name", "—")
        scores = (e.get("scores") or {}).get("scores") or {}
        merged = e.get("settings") or _merge_params(
            e.get("indexing_params") or {}, e.get("rag_params") or {}
        )

        cells = [html.escape(str(pattern_name))]

        for col in metric_columns:
            metric_name = col.replace("mean_", "", 1)
            info = scores.get(metric_name) or {}
            mean = info.get("mean")
            if mean is not None:
                cell = "%.4f" % mean if isinstance(mean, (int, float)) else str(mean)
            else:
                cell = ""
            cells.append(cell)

        for col in config_columns:
            val = _get_nested(merged, col)
            if val is not None:
                cells.append(str(val))
            else:
                cells.append("")
        rows.append("<tr>" + "".join("<td>%s</td>" % html.escape(c) for c in cells) + "</tr>")

    table_body = "".join(rows)
    html_content = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>RAG Patterns Leaderboard</title>
  <style>
    table { border-collapse: collapse; width: 100%%; }
    th, td { border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }
    th { background: #f5f5f5; font-weight: 600; }
    .numeric { text-align: right; }
  </style>
</head>
<body>
  <h1>RAG Patterns Leaderboard</h1>
  <p>Patterns ordered by optimization metric (best first).</p>
  <table>
    <thead>
      <tr>%s</tr>
    </thead>
    <tbody>
      %s
    </tbody>
  </table>
</body>
</html>
""" % (
        header_row,
        table_body,
    )

    Path(html_artifact.path).parent.mkdir(parents=True, exist_ok=True)
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
