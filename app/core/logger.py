import os
import json
from datetime import datetime
from typing import Dict, Any, List


class ScientificLogger:
    """
    Experiment logger using JSONL format (one JSON object per line).
    This avoids the read-all / rewrite-all overhead of a JSON array
    and is safe for concurrent writes (each write is a single append).
    """

    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self.log_file = os.path.join(workspace_dir, "experiment_log.jsonl")
        os.makedirs(workspace_dir, exist_ok=True)

    def _append(self, entry: Dict[str, Any]):
        """Append a single JSON line — O(1), no read-all needed."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # =====================================================
    # LOG PLANNER DECISION
    # =====================================================

    def log_planner(
        self,
        input_messages: List[Dict[str, Any]],
        llm_output: str,
        parsed_output: Dict[str, Any] = None
    ):
        entry = {
            "type": "planner",
            "timestamp": datetime.utcnow().isoformat(),
            "input_messages": input_messages,
            "llm_raw_output": llm_output,
            "parsed_output": parsed_output
        }
        self._append(entry)

    # =====================================================
    # LOG TOOL CALL
    # =====================================================

    def log_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result: Any
    ):
        entry = {
            "type": "tool_call",
            "timestamp": datetime.utcnow().isoformat(),
            "tool_name": tool_name,
            "arguments": args,
            "result": result
        }
        self._append(entry)

    # =====================================================
    # LOG FINAL RESULT
    # =====================================================

    def log_final_result(self, final_answer: Any):
        entry = {
            "type": "final_result",
            "timestamp": datetime.utcnow().isoformat(),
            "final_answer": final_answer
        }
        self._append(entry)

    # =====================================================
    # LOAD FULL LOG
    # =====================================================

    def load_logs(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_file):
            return []

        logs = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    logs.append(json.loads(line))
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"[LOGGER] Warning: skipping malformed log line: {e}")
        return logs

    # =====================================================
    # SUMMARIZE LOGS FOR LLM CONTEXT
    # =====================================================

    def summarize_logs(self) -> str:
        """
        Returns compact textual summary to inject into LLM context.
        Only includes the last 10 entries to avoid context bloat.
        """
        logs = self.load_logs()

        if not logs:
            return "No prior experiment logs."

        summary = "## Experiment Log Summary\n"

        for entry in logs[-10:]:  # only recent logs
            if entry["type"] == "planner":
                summary += (
                    f"\n[Planner] {entry['timestamp']}\n"
                    f"Parsed Output: {entry.get('parsed_output')}\n"
                )

            elif entry["type"] == "tool_call":
                result = entry.get("result", {})
                result_keys = list(result.keys()) if isinstance(result, dict) else repr(result)[:100]
                summary += (
                    f"\n[Tool] {entry['tool_name']} at {entry['timestamp']}\n"
                    f"Args: {entry['arguments']}\n"
                    f"Result keys: {result_keys}\n"
                )

            elif entry["type"] == "final_result":
                summary += (
                    f"\n[Final Result] {entry['timestamp']}\n"
                    f"{entry['final_answer']}\n"
                )

        return summary