import os
import shutil
from datetime import datetime
from typing import List, Dict


class WorkspaceManager:
    """
    Manages global workspace state.
    Handles datasets, images, and environment summary.
    """

    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir

        self.datasets_dir = os.path.join(workspace_dir, "datasets")
        self.images_dir = os.path.join(workspace_dir, "images")
        self.executions_dir = os.path.join(workspace_dir, "executions")

        self._initialize_structure()

    # ======================================================
    # INITIALIZATION
    # ======================================================

    def _initialize_structure(self):
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.executions_dir, exist_ok=True)

    # ======================================================
    # DATASET MANAGEMENT
    # ======================================================

    def register_dataset(self, file_path: str) -> str:
        """
        Copy dataset into workspace/datasets
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.datasets_dir, filename)

        shutil.copy(file_path, dest_path)
        return dest_path

    def list_datasets(self) -> List[str]:
        return os.listdir(self.datasets_dir)

    # ======================================================
    # IMAGE MANAGEMENT
    # ======================================================

    def register_image(self, file_path: str) -> str:
        """
        Copy image into workspace/images
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.images_dir, filename)

        shutil.copy(file_path, dest_path)
        return dest_path

    def list_images(self) -> List[str]:
        return os.listdir(self.images_dir)

    # ======================================================
    # EXECUTION MANAGEMENT
    # ======================================================

    def list_executions(self) -> List[str]:
        if not os.path.exists(self.executions_dir):
            return []
        return os.listdir(self.executions_dir)

    # ======================================================
    # CLEANUP
    # ======================================================

    def reset_workspace(self):
        """
        WARNING: Deletes entire workspace.
        """
        if os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)

        self._initialize_structure()

    # ======================================================
    # SUMMARY FOR LLM CONTEXT
    # ======================================================

    def summarize(self) -> str:
        """
        Structured summary for injecting into LLM.
        """

        summary = "## Workspace Overview\n"

        # Datasets
        datasets = self.list_datasets()
        summary += "\n### Datasets:\n"
        if datasets:
            for d in datasets:
                summary += f"- {d}\n"
        else:
            summary += "None\n"

        # Images
        images = self.list_images()
        summary += "\n### Images:\n"
        if images:
            for img in images:
                summary += f"- {img}\n"
        else:
            summary += "None\n"

        # Executions
        executions = self.list_executions()
        summary += "\n### Executions:\n"
        if executions:
            for e in executions:
                summary += f"- {e}\n"
        else:
            summary += "None\n"

        return summary

    # ======================================================
    # STRUCTURED SUMMARY (FOR PROGRAMMATIC USE)
    # ======================================================

    def structured_summary(self) -> Dict:
        return {
            "datasets": self.list_datasets(),
            "images": self.list_images(),
            "executions": self.list_executions(),
            "last_updated": datetime.utcnow().isoformat()
        }