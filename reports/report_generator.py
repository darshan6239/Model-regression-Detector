"""
Renders the Jinja2 HTML template into a static report file for one run.
"""
from pathlib import Path
from datetime import datetime, timezone
from jinja2 import Environment, FileSystemLoader

TEMPLATE_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent / "output"


def generate_report(run_metadata: dict, stats: dict, results: list[dict], diff: dict) -> str:
    """
    Renders the HTML report and writes it to reports/output/<run_id>.html.
    Returns the path to the generated file.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("template.html")

    html = template.render(
        run=run_metadata,
        stats=stats,
        results=results,
        diff=diff,
    )

    output_path = OUTPUT_DIR / f"{run_metadata['run_id']}.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Also keep a stable "latest.html" for convenience
    latest_path = OUTPUT_DIR / "latest.html"
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return str(output_path)
