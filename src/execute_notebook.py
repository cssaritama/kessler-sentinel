"""Headless Notebook Execution Engine for KESSLER-SENTINEL.

Executes code cells in notebook.ipynb, captures stdout and dataframe outputs,
and serializes a 100% executed, populated notebook with zero external dependencies.
"""

import ast
import contextlib
import io
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_cell_code(code_str: str, global_scope: dict):
    """Execute code string and capture printed stdout as well as last expression."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    outputs = []

    try:
        parsed = ast.parse(code_str)
        # If last statement is an expression, evaluate it to display its value
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            last_expr = parsed.body.pop()
            exec_code = compile(parsed, filename="<cell>", mode="exec")
            eval_code = compile(ast.Expression(last_expr.value), filename="<cell>", mode="eval")

            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(exec_code, global_scope)
                result = eval(eval_code, global_scope)
        else:
            exec_code = compile(parsed, filename="<cell>", mode="exec")
            result = None
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                exec(exec_code, global_scope)

        out_text = stdout_buf.getvalue()
        err_text = stderr_buf.getvalue()

        if out_text:
            outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in out_text.rstrip("\n").split("\n")],
            })
        if err_text:
            outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": [line + "\n" for line in err_text.rstrip("\n").split("\n")],
            })

        if result is not None:
            # Handle pandas dataframe or generic display
            if hasattr(result, "to_string"):
                res_str = result.to_string()
            else:
                res_str = repr(result)
            outputs.append({
                "output_type": "execute_result",
                "execution_count": 1,
                "data": {"text/plain": [line + "\n" for line in res_str.split("\n")]},
                "metadata": {},
            })

    except Exception as e:
        tb = traceback.format_exc()
        outputs.append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": [line + "\n" for line in tb.split("\n")],
        })
        print(f"[CELL ERROR] {e}")

    return outputs


def execute_notebook_file(nb_path: str = "notebook.ipynb"):
    print(f"[NOTEBOOK] Executing all cells in {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Disable interactive plot showing during batch execution to prevent blocking
    scope = {"__name__": "__main__"}

    exec_count = 1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            # Patch plt.show to avoid GUI window popup during headless execution
            patched_source = source.replace("plt.show()", "# plt.show()")
            print(f"  Executing code cell {exec_count} (index {i})...")
            cell_outputs = run_cell_code(patched_source, scope)
            cell["outputs"] = cell_outputs
            cell["execution_count"] = exec_count
            exec_count += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print(f"[NOTEBOOK] Successfully executed and populated {nb_path}!")


if __name__ == "__main__":
    execute_notebook_file()
