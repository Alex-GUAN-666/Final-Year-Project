"""Parse, isolate and grade generated programs. Never execute model code on the host."""
import ast
import math
import os
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from decimal import Decimal, InvalidOperation
from pathlib import Path

GRADER_PROTOCOL = "python-first-fence_ast-last-expr_decimal-finite-float_isclose-rel1e-3-abs0_v1"


def extract_first_code(response):
    """Use only the first fenced block, accepting Python or an unlabeled fence."""
    match = re.search(r"```([^\n`]*)\n(.*?)```", response, re.DOTALL)
    if not match or match.group(1).strip().lower() not in ("", "python", "py"):
        return None
    return match.group(2).strip()


def wrap_last_expression(code):
    """Historical AST behavior: print the final expression; never invent a solve() call."""
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        value = tree.body[-1].value
        is_print = (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "print")
        if not is_print:
            tree.body[-1] = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [value], []))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def parse_scalar(value):
    """Strict scalar stdout: Decimal validation followed by finite float comparison."""
    token = str(value).strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
        raise ValueError("Expected one numeric scalar, without prose or extra lines")
    try:
        decimal = Decimal(token)
        number = float(decimal)
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise ValueError("Invalid numeric scalar") from exc
    if not decimal.is_finite() or not math.isfinite(number) or (decimal != 0 and number == 0):
        raise ValueError("Scalar is nonfinite or outside the finite nonzero float range")
    return number


def grade_execution(execution, reference):
    if execution.get("status") in ("not_executed", "execution_error"):
        return {"status": "ungraded", "correct": None}
    try:
        expected = parse_scalar(reference)
    except ValueError as exc:
        return {"status": "invalid_reference", "correct": None, "reason": str(exc)}
    if execution.get("status") != "ok":
        return {"status": "incorrect", "correct": False, "reason": execution.get("status")}
    try:
        observed = parse_scalar(execution.get("stdout", ""))
    except ValueError as exc:
        return {"status": "incorrect", "correct": False, "reason": str(exc)}
    correct = math.isclose(observed, expected, rel_tol=1e-3, abs_tol=0.0)
    return {"status": "correct" if correct else "incorrect", "correct": correct}


def ensure_docker(image):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Docker scoring requires Linux or WSL2. Native Windows pipe selectors are unsupported; run this command inside WSL2 with Docker integration enabled.")
    if not shutil.which("docker"):
        raise RuntimeError("Docker is unavailable. Install/start Docker or use --execution none (ungraded).")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@+\-]*", image):
        raise ValueError("Invalid Docker image reference")
    try:
        subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        result = subprocess.run(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                                check=True, capture_output=True, text=True, timeout=15)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError("Docker daemon or local image is unavailable. Start Docker and explicitly pull/build the image; no host execution fallback is permitted.") from exc
    return result.stdout.strip()


def execute_docker(code, image="python:3.11-slim", timeout=10, output_limit=65536):
    """Run a single isolated container; cap output while reading, then remove it."""
    if timeout <= 0 or output_limit <= 0:
        raise ValueError("Timeout and output limit must be positive")
    ensure_docker(image)
    if code is None:
        return {"status": "no_code", "stdout": "", "stderr": "No first Python fence"}
    if len(code.encode("utf-8")) > 262144:
        return {"status": "code_error", "stdout": "", "stderr": "Code exceeds 256 KiB limit"}
    try:
        wrapped = wrap_last_expression(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return {"status": "code_error", "stdout": "", "stderr": str(exc)}
    name = "fyp-exec-" + uuid.uuid4().hex
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    status, process = None, None
    with tempfile.TemporaryDirectory(prefix="fyp-code-") as directory:
        program = Path(directory) / "main.py"
        program.write_text(wrapped, encoding="utf-8")
        program.chmod(0o444)
        command = ["docker", "run", "--rm", "--pull", "never", "--name", name,
                   "--network", "none", "--read-only", "--user", "65534:65534",
                   "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                   "--memory", "256m", "--memory-swap", "256m", "--pids-limit", "32",
                   "--cpus", "0.5", "--ulimit", "nofile=64:64",
                   "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--entrypoint", "python",
                   "--mount", "type=bind,src=" + str(program) + ",dst=/work/main.py,readonly",
                   image, "-I", "-B", "-u", "/work/main.py"]
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                for label, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, label)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        status = "timeout"
                        break
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        room = output_limit - sum(map(len, captured.values()))
                        captured[key.data].extend(chunk[:room])
                        if len(chunk) > room:
                            status = "output_limit"
                            break
                    if status:
                        break
            if status is None:
                try:
                    returncode = process.wait(timeout=max(0.01, deadline - time.monotonic()))
                    status = "ok" if returncode == 0 else ("execution_error" if returncode in (125, 126, 127) else "code_error")
                except subprocess.TimeoutExpired:
                    status = "timeout"
        except OSError as exc:
            status = "execution_error"
            captured["stderr"].extend(str(exc).encode()[:output_limit])
        finally:
            try:
                if process:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                    process.stdout.close()
                    process.stderr.close()
            finally:
                try:
                    cleanup = subprocess.run(["docker", "rm", "-f", name],
                                             capture_output=True, text=True, timeout=10)
                    if cleanup.returncode and "No such container" not in cleanup.stderr:
                        raise RuntimeError("Failed to confirm Docker container cleanup: " + name)
                except (OSError, subprocess.SubprocessError):
                    raise RuntimeError("Failed to confirm Docker container cleanup: " + name)
    return {"status": status, **{key: value.decode("utf-8", errors="replace") for key, value in captured.items()}}
