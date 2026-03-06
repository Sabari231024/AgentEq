import os
import shutil

WORKSPACE_DIR = os.path.join(os.path.dirname(__file__), "workspace")

# Subdirectories created during execution
WORKSPACE_SUBDIRS = ["datasets", "images", "executions"]

# Log/result files created at workspace root
WORKSPACE_FILES = [
    "logs.json",
    "experiment_log.json",
]


def cleanup(verbose: bool = True) -> None:
    """
    Resets the workspace to its original empty state after execution.

    Removes:
    - workspace/datasets/    (uploaded CSV files)
    - workspace/images/      (generated plots)
    - workspace/executions/  (artifact directories)
    - workspace/logs.json    (sandbox execution logs)
    - workspace/experiment_log.json  (planner + tool call logs)
    - Any leftover pysr_result_*.json files in workspace root

    Recreates the empty subdirectory structure so the next run
    can start fresh without errors.
    """

    if verbose:
        print("\n[CLEANUP] Starting workspace cleanup...")

    # ---------------------------------------------------------
    # 1. Remove and recreate subdirectories
    # ---------------------------------------------------------
    for subdir in WORKSPACE_SUBDIRS:
        path = os.path.join(WORKSPACE_DIR, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
            if verbose:
                print(f"[CLEANUP] Removed directory: workspace/{subdir}/")
        os.makedirs(path, exist_ok=True)
        if verbose:
            print(f"[CLEANUP] Recreated empty: workspace/{subdir}/")

    # ---------------------------------------------------------
    # 2. Remove known log files
    # ---------------------------------------------------------
    for fname in WORKSPACE_FILES:
        fpath = os.path.join(WORKSPACE_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            if verbose:
                print(f"[CLEANUP] Removed file: workspace/{fname}")

    # ---------------------------------------------------------
    # 3. Remove any leftover pysr_result_*.json files
    # ---------------------------------------------------------
    if os.path.exists(WORKSPACE_DIR):
        for fname in os.listdir(WORKSPACE_DIR):
            if fname.startswith("pysr_result_") and fname.endswith(".json"):
                os.remove(os.path.join(WORKSPACE_DIR, fname))
                if verbose:
                    print(f"[CLEANUP] Removed file: workspace/{fname}")

    if verbose:
        print("[CLEANUP] Workspace reset to original state.\n")


if __name__ == "__main__":
    cleanup()
