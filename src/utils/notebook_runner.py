"""Execute Jupyter notebooks cleanly via jupyter_client + ipykernel without external dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jupyter_client.manager import start_new_kernel
import nbformat


def execute_notebook(nb_path: str | Path, timeout_sec: int = 300) -> dict[str, Any]:
    """Execute every code cell of a notebook in a live kernel and persist outputs in-place."""
    path = Path(nb_path).resolve()
    nb = nbformat.read(path, as_version=4)

    km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent))
    code_count = 0
    errors: list[dict[str, Any]] = []
    exec_num = 1
    try:
        for cell_idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            code_count += 1
            src = cell.source
            if not src.strip():
                cell.execution_count = exec_num
                cell.outputs = []
                exec_num += 1
                continue

            msg_id = kc.execute(src, allow_stdin=False, stop_on_error=True)
            outputs: list[Any] = []
            while True:
                msg = kc.get_iopub_msg(timeout=timeout_sec)
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    continue
                mtype = msg["header"]["msg_type"]
                content = msg["content"]
                if mtype == "status" and content.get("execution_state") == "idle":
                    break
                if mtype == "stream":
                    outputs.append(nbformat.v4.new_output("stream", name=content["name"], text=content["text"]))
                elif mtype in ("display_data", "execute_result"):
                    out_kwargs: dict[str, Any] = {
                        "data": content.get("data", {}),
                        "metadata": content.get("metadata", {}),
                    }
                    if mtype == "execute_result":
                        out_kwargs["execution_count"] = exec_num
                    outputs.append(nbformat.v4.new_output(mtype, **out_kwargs))
                elif mtype == "error":
                    outputs.append(
                        nbformat.v4.new_output(
                            "error",
                            ename=content.get("ename", "Error"),
                            evalue=content.get("evalue", ""),
                            traceback=content.get("traceback", []),
                        )
                    )
                    errors.append({"cell_idx": cell_idx, "ename": content.get("ename"), "evalue": content.get("evalue")})
            cell.execution_count = exec_num
            cell.outputs = outputs
            exec_num += 1
            if errors:
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)

    nbformat.write(nb, path)
    if errors:
        raise RuntimeError(f"Notebook {path.name} failed at cell {errors[0]['cell_idx']}: {errors[0]['ename']}: {errors[0]['evalue']}")
    return {"notebook": str(path), "code_cells_executed": code_count, "errors": 0}

