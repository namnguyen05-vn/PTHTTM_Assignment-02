"""Execute a trusted notebook in-process when the sandbox blocks Jupyter TCP kernels."""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import traceback
from pathlib import Path

def json_safe(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def execute(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    execution_count = 0

    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        execution_count += 1
        cell["execution_count"] = execution_count
        cell["outputs"] = []
        stdout = io.StringIO()
        stderr = io.StringIO()

        def flush_streams():
            out = stdout.getvalue()
            err = stderr.getvalue()
            if out:
                cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": out})
                stdout.seek(0)
                stdout.truncate(0)
            if err:
                cell["outputs"].append({"output_type": "stream", "name": "stderr", "text": err})
                stderr.seek(0)
                stderr.truncate(0)

        def add_display(value):
            flush_streams()
            data = {"text/plain": repr(value)}
            if hasattr(value, "_repr_html_"):
                try:
                    html = value._repr_html_()
                    if html:
                        data["text/html"] = html
                except Exception:
                    pass
            cell["outputs"].append({
                "output_type": "display_data",
                "metadata": {},
                "data": {k: json_safe(v) for k, v in data.items()},
            })

        def capture_show(*_args, **_kwargs):
            import matplotlib.pyplot as plt

            flush_streams()
            for number in plt.get_fignums():
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                cell["outputs"].append({
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {"image/png": encoded},
                })
            plt.close("all")

        namespace["display"] = add_display
        namespace["__capture_show__"] = capture_show
        source = cell["source"].replace("from IPython.display import display", "# display injected by in-process runner")
        source = source.replace("plt.show()", "__capture_show__()")

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{path.name}:cell-{execution_count}", "exec"), namespace)
        except Exception as exc:
            flush_streams()
            cell["outputs"].append({
                "output_type": "error",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": traceback.format_exc().splitlines(),
            })
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            raise
        else:
            flush_streams()

    notebook["metadata"]["execution"] = {
        "engine": "trusted in-process runner (sandbox-compatible)",
        "status": "completed",
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    execute(args.notebook.resolve())
    print(f"Executed {args.notebook}")
