import os
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox

_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path), override=True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
DATASETS_DIR = os.path.join(WORKSPACE_DIR, "datasets")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(WORKSPACE_DIR, "logs.json")


def interpreter(code: str) -> dict:
    execution_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    sbx = Sandbox.create()
    try:
        # ── Upload local dataset files into sandbox ──────────────
        # e2b runs in a remote Docker container — it has NO access
        # to the local filesystem. We must explicitly upload files.
        if os.path.exists(DATASETS_DIR):
            for fname in os.listdir(DATASETS_DIR):
                fpath = os.path.join(DATASETS_DIR, fname)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        sbx.files.write(f"/home/user/{fname}", f.read())
                    print(f"[SANDBOX] Uploaded: {fname}")

        # ── Execute the code ─────────────────────────────────────
        execution = sbx.run_code(code)
        logs = execution.logs

        # e2b returns stdout/stderr as lists of strings — join them
        raw_stdout = logs.stdout if hasattr(logs, "stdout") else []
        raw_stderr = logs.stderr if hasattr(logs, "stderr") else []
        stdout = "\n".join(raw_stdout) if isinstance(raw_stdout, list) else str(raw_stdout)
        stderr = "\n".join(raw_stderr) if isinstance(raw_stderr, list) else str(raw_stderr)

        # ── Download output files from /home/user ────────────────
        # Only scan /home/user to avoid iterating system directories
        saved_files = []
        try:
            files = sbx.files.list("/home/user")
            for file in files:
                if hasattr(file, "type") and file.type == "file":
                    try:
                        content = sbx.files.read(file.path)
                        filename = os.path.basename(file.path)
                        # Don't re-download files we uploaded
                        if filename in os.listdir(DATASETS_DIR):
                            continue
                        local_path = os.path.join(WORKSPACE_DIR, filename)
                        with open(local_path, "wb") as f:
                            f.write(content)
                        saved_files.append(local_path)
                        print(f"[SANDBOX] Downloaded: {filename}")
                    except Exception:
                        pass
        except Exception:
            pass

    except Exception as e:
        stdout = ""
        stderr = str(e)
        saved_files = []
    finally:
        try:
            sbx.kill()
        except Exception:
            pass

    log_data = {
        "execution_id": execution_id,
        "timestamp": timestamp,
        "input_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "saved_files": saved_files,
    }
    if os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, "r") as f:
                existing_logs = json.load(f)
        except json.JSONDecodeError:
            existing_logs = []
    else:
        existing_logs = []
    existing_logs.append(log_data)
    with open(LOG_FILE_PATH, "w") as f:
        json.dump(existing_logs, f, indent=4)
    return log_data
