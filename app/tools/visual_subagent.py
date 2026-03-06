import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from prompts.ToolPrompt import VISUAL_SUBAGENT_SYSTEM_PROMPT

_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path), override=True)

# Max image size we'll send to the vision model (5 MB).
_MAX_IMAGE_BYTES = 5 * 1024 * 1024

# The workspace images dir — image_path must resolve inside this.
_WORKSPACE_IMAGES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "workspace", "images")
)


class VisualSubagent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )

    def _validate_image_path(self, image_path: str) -> str:
        """
        Resolve and validate that image_path is inside the workspace images dir.
        Raises ValueError if the path escapes the allowed directory (path traversal guard).
        """
        resolved = os.path.abspath(image_path)

        # Also check workspace root for images saved directly there (e.g. by sandbox)
        workspace_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "workspace")
        )

        if not (resolved.startswith(_WORKSPACE_IMAGES_DIR) or resolved.startswith(workspace_root)):
            raise ValueError(
                f"Security: image_path '{image_path}' resolves outside the workspace directory. "
                f"Allowed prefix: '{workspace_root}'"
            )

        return resolved

    def analyze(self, image_path: str) -> Dict[str, Any]:
        # Validate path before touching the filesystem
        try:
            safe_path = self._validate_image_path(image_path)
        except ValueError as e:
            return {"error": str(e)}

        if not os.path.exists(safe_path):
            return {"error": f"Image not found: {safe_path}"}

        # Size check — warn if image is too large for the API
        image_size = os.path.getsize(safe_path)
        if image_size > _MAX_IMAGE_BYTES:
            print(
                f"[VISUAL] Warning: image is {image_size / 1024 / 1024:.1f} MB "
                f"(limit {_MAX_IMAGE_BYTES // 1024 // 1024} MB). May hit API size limits."
            )

        with open(safe_path, "rb") as f:
            image_bytes = f.read()

        messages = [
            SystemMessage(content=VISUAL_SUBAGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "image",
                        "data": image_bytes,
                        "mime_type": "image/png"
                    },
                    {
                        "type": "text",
                        "text": "Analyze this visualization."
                    }
                ]
            )
        ]
        response = self.llm.invoke(messages)
        if isinstance(response.content, list):
            content = "".join(
                part.get("text", "") for part in response.content
                if isinstance(part, dict)
            ).strip()
        else:
            content = response.content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON returned by visual subagent",
                "raw_output": content
            }