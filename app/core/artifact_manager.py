import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any


class ArtifactManager:
    """
    Manages scientific experiment artifacts.
    Handles:
    - Execution directories
    - Artifact saving
    - Metadata tracking
    - Workspace summaries
    """

    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self.executions_dir = os.path.join(workspace_dir, "executions")
        os.makedirs(self.executions_dir, exist_ok=True)

    # ======================================================
    # EXECUTION MANAGEMENT
    # ======================================================

    def create_execution(self) -> str:
        execution_id = str(uuid.uuid4())
        execution_path = os.path.join(self.executions_dir, execution_id)
        os.makedirs(execution_path, exist_ok=True)
        return execution_id

    def get_execution_path(self, execution_id: str) -> str:
        return os.path.join(self.executions_dir, execution_id)

    # ======================================================
    # ARTIFACT SAVING
    # ======================================================

    def save_artifact(
        self,
        execution_id: str,
        filename: str,
        content: bytes,
        metadata: Dict[str, Any] = None,
    ) -> str:

        execution_path = self.get_execution_path(execution_id)
        os.makedirs(execution_path, exist_ok=True)

        file_path = os.path.join(execution_path, filename)

        with open(file_path, "wb") as f:
            f.write(content)

        artifact_metadata = {
            "filename": filename,
            "file_type": self._detect_file_type(filename),
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        self._append_metadata(execution_id, artifact_metadata)

        return file_path

    # ======================================================
    # METADATA TRACKING
    # ======================================================

    def _append_metadata(self, execution_id: str, artifact_metadata: Dict[str, Any]):
        metadata_file = os.path.join(
            self.get_execution_path(execution_id),
            "artifacts.json"
        )

        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    print(f"[ArtifactManager] Warning: artifacts.json had unexpected format, resetting.")
                    data = []
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[ArtifactManager] Warning: failed to parse artifacts.json: {e}. Resetting.")
                data = []
        else:
            data = []

        data.append(artifact_metadata)

        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=4)

    # ======================================================
    # FILE TYPE DETECTION
    # ======================================================

    def _detect_file_type(self, filename: str) -> str:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        type_map = {
            "png": "image", "jpg": "image", "jpeg": "image",
            "csv": "dataset",
            "json": "json",
            "py": "code",
            "txt": "text",
            "pdf": "document",
        }
        return type_map.get(ext, "unknown")

    # ======================================================
    # ARTIFACT LISTING
    # ======================================================

    def list_all_executions(self) -> List[str]:
        if not os.path.exists(self.executions_dir):
            return []
        return os.listdir(self.executions_dir)

    def list_execution_artifacts(self, execution_id: str) -> List[Dict[str, Any]]:
        metadata_file = os.path.join(
            self.get_execution_path(execution_id),
            "artifacts.json"
        )

        if not os.path.exists(metadata_file):
            return []

        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[ArtifactManager] Warning: failed to read artifacts.json: {e}")
            return []

    # ======================================================
    # WORKSPACE SUMMARY FOR LLM
    # ======================================================

    def summarize_workspace(self) -> str:
        """
        Generates a summary of all artifacts for LLM context.
        """

        summary = "## Workspace Artifact Summary\n"

        executions = self.list_all_executions()

        if not executions:
            summary += "No executions found.\n"
            return summary

        for execution_id in executions:
            summary += f"\nExecution ID: {execution_id}\n"
            artifacts = self.list_execution_artifacts(execution_id)

            if not artifacts:
                summary += "  No artifacts.\n"
                continue

            for art in artifacts:
                summary += (
                    f"  - {art.get('filename', '?')} "
                    f"({art.get('file_type', '?')}) "
                    f"Created: {art.get('created_at', '?')}\n"
                )

        return summary