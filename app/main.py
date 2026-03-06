import os
import sys

# Ensure app/ is on the path so prepare_instance and cleanup can be found
# regardless of which directory the script is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.graph import build_graph
from prepare_instance import prepare_instance
from cleanup import cleanup

instance_id = "lsr_synth_matsci_matsci0"
dataset_file, target_column = prepare_instance(instance_id)
print("Dataset prepared:", dataset_file)

graph = build_graph()

query = f"""
Dataset file: {instance_id}.csv
Target column: {target_column}

Perform symbolic regression to discover the equation.
Return only the RHS expression.
"""

final_result = None

try:
    result = graph.invoke({
        "messages": [
            {"role": "user", "content": query}
        ],
        "tool_call": None,
        "tool_output": None,
        "final_answer": None,
        "iteration_count": 0
    })

    final_result = result.get("final_answer")

    print("\n========== FINAL RESULT ==========\n")
    print(final_result)

except KeyboardInterrupt:
    print("\n[MAIN] Run interrupted by user.")

except Exception as e:
    print(f"\n[MAIN] Unhandled error: {e}")
    raise

finally:
    # Cleanup always runs, but result is already captured above.
    cleanup(verbose=True)
