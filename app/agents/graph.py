import json
import os
import re
import sys
import time
from datetime import datetime

# Ensure app/ is importable regardless of CWD (fixes fragile sys.path.append("../.."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts.SystemPrompt import SYSTEM_PROMPT
from tools.Sandbox import interpreter
from tools.visual_subagent import VisualSubagent
from tools.pySRTool import PySRTool

from core.workspace_manager import WorkspaceManager
from core.artifact_manager import ArtifactManager
from core.logger import ScientificLogger

from dotenv import load_dotenv

# Always resolve .env relative to this file's location (D:\KeplerAgent\.env),
# so it works regardless of which directory the script is launched from.
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path), override=True)


# =========================================================
# STATE
# =========================================================

class AgentState(TypedDict):
    messages: List[dict]
    tool_call: Optional[Dict[str, Any]]
    tool_output: Optional[Dict[str, Any]]
    final_answer: Optional[str]
    iteration_count: Optional[int]


# =========================================================
# LLM CONFIG
# =========================================================

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-3-flash-preview",  # highest free-tier quota
    temperature=0
)


def invoke_llm_with_retry(messages, max_retries=5):
    """Invoke LLM with exponential backoff on resource exhausted (429) errors."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err = str(e).lower()
            if "resource_exhausted" in err or "429" in err or "quota" in err:
                wait = 2 ** attempt  # 1, 2, 4, 8, 16 seconds
                print(f"[RETRY]   Rate limited. Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise  # non-rate-limit errors bubble up immediately
    raise RuntimeError(f"LLM failed after {max_retries} retries due to rate limits.")


# =========================================================
# INFRASTRUCTURE INITIALIZATION
# =========================================================

WORKSPACE_DIR = "workspace"

workspace_manager = WorkspaceManager(workspace_dir=WORKSPACE_DIR)
artifact_manager = ArtifactManager(workspace_dir=WORKSPACE_DIR)
logger = ScientificLogger(workspace_dir=WORKSPACE_DIR)


def load_workspace_context() -> str:
    """
    Inject structured workspace + artifact + log summary
    into LLM context.
    """
    workspace_summary = workspace_manager.summarize()
    artifact_summary = artifact_manager.summarize_workspace()
    log_summary = logger.summarize_logs()

    return (
        workspace_summary
        + "\n"
        + artifact_summary
        + "\n"
        + log_summary
    )


# =========================================================
# LOOP / REPETITION DETECTION
# =========================================================

def is_looping_response(content: str, chunk_size: int = 200, threshold: int = 3) -> bool:
    """
    Detect when the LLM has looped inside a single response.
    A response is considered looping if any chunk_size-character
    sliding window appears >= threshold times in the content.
    """
    if len(content) < chunk_size * threshold:
        return False
    seen: dict = {}
    for i in range(0, len(content) - chunk_size, chunk_size // 2):
        chunk = content[i:i + chunk_size]
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= threshold:
            return True
    return False


# =========================================================
# JSON EXTRACTION HELPER
# =========================================================

def extract_json_block(content: str) -> str | None:
    """
    Scans the LLM response left-to-right and returns the first
    balanced JSON object that contains 'tool_call' or 'final_result'.

    WHY: rfind('{') is wrong — it finds the deepest/innermost '{'
    (e.g. the '{' before `"code": "..."`), not the outermost object.
    We need the outermost JSON object that carries the agent's decision.
    """
    start = 0
    while True:
        pos = content.find("{", start)
        if pos == -1:
            break

        # Walk from this '{' and balance braces
        depth = 0
        end_idx = None
        for i, ch in enumerate(content[pos:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = pos + i + 1
                    break

        if end_idx:
            candidate = content[pos:end_idx]
            try:
                parsed = json.loads(candidate)
                # Accept only if it's the decision object
                if "tool_call" in parsed or "final_result" in parsed or "tool_calls" in parsed:
                    return candidate
            except json.JSONDecodeError:
                pass  # not valid JSON, keep searching

        start = pos + 1  # try next '{'

    return None  # no valid decision JSON found


# =========================================================
# PLANNER NODE
# =========================================================

def planner(state: AgentState):
    iteration = state.get("iteration_count", 0) + 1

    print(f"\n{'='*60}")
    print(f"[PLANNER] Iteration #{iteration}  |  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"{'='*60}")

    # ── Guard: max iterations ─────────────────────────────────
    MAX_ITERATIONS = 20
    if iteration > MAX_ITERATIONS:
        print(f"[STOP]    Max iterations ({MAX_ITERATIONS}) reached. Terminating.")
        return {"final_answer": "Max iterations reached without convergence.", "iteration_count": iteration}

    # ── Auto-stop on MAPE target ──────────────────────────────
    if state.get("tool_output") and isinstance(state["tool_output"], dict):
        mape = state["tool_output"].get("mape")
        equation = state["tool_output"].get("equation")

        if mape is not None:
            print(f"[METRIC]  Last tool output -> MAPE = {mape:.4f}%  |  Equation: {equation}")

        if mape is not None and mape < 0.1:
            print(f"[STOP]    MAPE < 0.1% target achieved. Auto-stopping.")
            logger.log_final_result(equation)
            return {"final_answer": equation, "iteration_count": iteration}

    # ── Build context & call LLM ──────────────────────────────
    print(f"[PLANNER] Loading workspace context...")
    workspace_context = load_workspace_context()

    # Sliding-window truncation: keep only the last MAX_HISTORY messages
    # from state to prevent context window overflow over long runs.
    # Always keep the very first user message (the original task).
    MAX_HISTORY = 12
    history = state["messages"]
    if len(history) > MAX_HISTORY:
        history = history[:1] + history[-(MAX_HISTORY - 1):]
        print(f"[PLANNER] Context truncated: keeping first message + last {MAX_HISTORY - 1} messages.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": workspace_context},
    ] + history

    print(f"[PLANNER] Sending {len(messages)} messages to LLM...")

    # ── LLM call with empty-response retry ───────────────────
    # Gemini can return empty content due to: safety filters, thinking-mode
    # models, rate-limit degradation, or context overflow. Retry up to 3 times.
    content = ""
    for empty_attempt in range(3):
        response = invoke_llm_with_retry(messages)

        # Extract text — handle both list (multimodal) and str responses
        if isinstance(response.content, list):
            content = "".join(
                part.get("text", "")
                for part in response.content
                if isinstance(part, dict)
            ).strip()
        else:
            content = (response.content or "").strip()

        # Gemini 'thinking' models put reasoning in response_metadata, not content.
        # Fall back to that if content is empty.
        if not content:
            meta = getattr(response, "response_metadata", {}) or {}
            content = (
                meta.get("candidates", [{}])[0]
                .get("content", {}).get("parts", [{}])[0]
                .get("text", "")
            ).strip() if meta.get("candidates") else ""

        if content:
            break  # got a real response

        print(f"[WARN]    LLM returned empty response (attempt {empty_attempt + 1}/3). "
              f"Possible causes: safety filter, rate-limit degradation, context overflow. Retrying...")
        time.sleep(10)

    if not content:
        print(f"[ERROR]   LLM returned empty response 3 times in a row. Skipping iteration.")
        logger.log_planner(messages[-3:], "", None)
        return {"iteration_count": iteration}  # skip — don't end as empty final_answer

    # ── Loop/repetition detection ────────────────────────────
    # LLMs sometimes get stuck repeating the same text block endlessly
    # within a single generation ("Wait, I already saw it fail..." x5).
    # Detect and skip rather than treating as a final answer.
    if is_looping_response(content):
        print(
            f"[WARN]    LLM response appears to be looping/repeating ({len(content)} chars). "
            f"Skipping iteration and retrying."
        )
        print(f"          First 300 chars: {content[:300].replace(chr(10), ' ')}")
        logger.log_planner(messages[-3:], content, None)
        return {"iteration_count": iteration}  # skip — don't set final_answer

    # ── Show LLM reasoning preview ────────────────────────────
    preview = content[:400].replace("\n", " ")
    print(f"[LLM]     Response preview:\n    {preview}..." if len(content) > 400 else f"[LLM]     Response:\n    {content}")

    # ── Extract JSON block from response ─────────────────────
    json_str = extract_json_block(content)

    # Also check markdown code fences: ```json { ... } ```
    if not json_str:
        md_match = re.search(r"```(?:json)?\s*({.*?})\s*```", content, re.DOTALL)
        if md_match:
            json_str = md_match.group(1)

    if not json_str:
        # Short non-JSON content = likely a genuine final answer in plain text.
        # Long non-JSON content = confused model that didn't follow format — skip and retry.
        if len(content) > 500:
            print(
                f"[WARN]    No JSON found in long response ({len(content)} chars). "
                f"Model likely confused. Skipping and retrying."
            )
            logger.log_planner(messages[-3:], content, None)
            return {"iteration_count": iteration}  # retry, don't end as garbage final_answer

        print(f"[WARN]    No JSON block found in LLM response. Treating as final answer.")
        print(f"          Content: {content[:200]}")
        logger.log_planner(messages[-3:], content, None)
        return {"final_answer": content, "iteration_count": iteration}

    print(f"[PARSE]   Extracted JSON: {json_str[:200]}..." if len(json_str) > 200 else f"[PARSE]   Extracted JSON: {json_str}")

    try:
        parsed = json.loads(json_str)
        logger.log_planner(messages[-3:], content, parsed)

        # ── Handle singular tool_call ──────────────────────────
        if "tool_call" in parsed:
            tool_name = parsed["tool_call"].get("tool_name", "unknown")
            tool_args = parsed["tool_call"].get("args", {})
            print(f"[PLANNER] Dispatching tool: '{tool_name}'")
            print(f"          Args: {json.dumps(tool_args, indent=6)[:300]}")
            return {"tool_call": parsed["tool_call"], "iteration_count": iteration}

        # ── Handle plural tool_calls — dispatch first one ──────
        # System prompt allows up to 3 tool_calls at once.
        # We dispatch the first; the rest will be handled in future iterations.
        if "tool_calls" in parsed and isinstance(parsed["tool_calls"], list) and parsed["tool_calls"]:
            first_call = parsed["tool_calls"][0]
            tool_name = first_call.get("tool_name", "unknown")
            tool_args = first_call.get("args", {})
            remaining = len(parsed["tool_calls"]) - 1
            print(f"[PLANNER] Received {len(parsed['tool_calls'])} tool_calls. Dispatching first: '{tool_name}'")
            if remaining > 0:
                print(f"          {remaining} remaining call(s) deferred to next iteration")
            print(f"          Args: {json.dumps(tool_args, indent=6)[:300]}")
            return {"tool_call": first_call, "iteration_count": iteration}

        if "final_result" in parsed:
            print(f"[RESULT]  Planner returned final result: {parsed['final_result']}")
            logger.log_final_result(parsed["final_result"])
            return {"final_answer": parsed["final_result"], "iteration_count": iteration}

    except Exception as e:
        print(f"[ERROR]   Failed to parse extracted JSON: {e}")
        print(f"          Extracted snippet: {json_str[:300]}")
        logger.log_planner(messages[-3:], content, None)
        return {"final_answer": content, "iteration_count": iteration}

    print(f"[WARN]    No tool_call or final_result in parsed output. Ending.")
    return {"final_answer": content, "iteration_count": iteration}


# =========================================================
# TOOL INITIALIZATION
# =========================================================

visual_tool = VisualSubagent()
pysr_tool = PySRTool(workspace_dir=WORKSPACE_DIR)


# =========================================================
# TOOL EXECUTOR NODE
# =========================================================

def tool_executor(state: AgentState):
    tool_spec = state["tool_call"]

    print(f"\n{'-'*60}")
    print(f"[TOOL]    TOOL EXECUTOR")
    print(f"{'-'*60}")

    if not isinstance(tool_spec, dict):
        print(f"[ERROR]   Invalid tool specification received: {tool_spec}")
        return {
            "messages": state["messages"] + [{
                "role": "assistant",
                "content": "Invalid tool specification."
            }],
            "tool_call": None
        }

    tool_name = tool_spec.get("tool_name")
    args = tool_spec.get("args", {})

    print(f"[TOOL]    Running: '{tool_name}'")
    print(f"          Args: {json.dumps(args, indent=6)[:300]}")

    try:

        # ---------------------------------------------------
        # Python Interpreter
        # ---------------------------------------------------
        if tool_name == "python_interpreter":
            code = args.get("code", "")
            print(f"[SANDBOX] Executing Python code ({len(code)} chars)...")
            result = interpreter(code)
            stdout_preview = (result.get("stdout") or "")[:300]
            stderr_preview = (result.get("stderr") or "")[:200]
            print(f"          stdout: {stdout_preview}")
            if stderr_preview:
                print(f"          stderr: {stderr_preview}")
            print(f"          saved_files: {result.get('saved_files', [])}")

        # ---------------------------------------------------
        # Visual Subagent
        # ---------------------------------------------------
        elif tool_name == "visual_subagent":
            image_path = args.get("image_path")
            print(f"[VISUAL]  Analyzing image: {image_path}")
            result = visual_tool.analyze(image_path)
            print(f"          Output keys: {list(result.keys()) if isinstance(result, dict) else 'raw text'}")

        # ---------------------------------------------------
        # PySR Tool
        # ---------------------------------------------------
        elif tool_name == "pysr":
            print(f"[PYSR]    Running symbolic regression...")
            print(f"          input_file: {args.get('input_file')}  |  target: {args.get('target_column')}")
            print(f"          niterations: {args.get('niterations', 40)}  |  population_size: {args.get('population_size', 1000)}")
            result = pysr_tool.run(**args)
            print(f"          Best equation: {result.get('equation')}")
            print(f"          MAPE: {result.get('mape', 'N/A')}%")

        else:
            print(f"[WARN]    Unknown tool name: '{tool_name}'")
            result = {"error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        print(f"[ERROR]   Tool '{tool_name}' raised exception: {e}")
        result = {"error": str(e)}

    logger.log_tool_call(tool_name, args, result)
    print(f"[LOG]     Tool result written to experiment_log.json")

    tool_message = (
        f"Tool '{tool_name}' result:\n"
        f"{json.dumps(result, indent=2, default=str)}"
    )

    updated_messages = state["messages"] + [
        {
            "role": "assistant",
            "content": tool_message
        }
    ]

    return {
        "messages": updated_messages,
        "tool_call": None,
        "tool_output": result
    }


# =========================================================
# GRAPH BUILD
# =========================================================

def build_graph():
    print("\n[GRAPH]   Building KeplerAgent LangGraph...")

    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("tool_executor", tool_executor)

    graph.set_entry_point("planner")

    def route(state: AgentState):
        if state.get("tool_call"):
            print(f"\n[ROUTE]   planner -> tool_executor")
            return "tool_executor"
        print(f"\n[ROUTE]   planner -> END")
        return END

    graph.add_conditional_edges(
        "planner",
        route,
        {
            "tool_executor": "tool_executor",
            END: END
        }
    )

    graph.add_edge("tool_executor", "planner")

    print("[GRAPH]   Compiled: planner <-> tool_executor (loop until final_answer or max iterations)")
    print("          Flow: [planner] -> [tool_executor] -> [planner] -> ... -> END")
    print(f"          Max iterations: 20\n")
    return graph.compile()