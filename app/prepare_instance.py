import pandas as pd
import os
import numpy as np
import ast

# DATA_PATH: configurable via env var or auto-resolved relative to this file.
# This avoids hard-coded absolute paths that break on other machines.
DATA_PATH = os.environ.get(
    "KEPLER_DATA_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lsr_synth_matsci_train.csv")
)

# Always resolve workspace relative to this file so CWD doesn't matter.
WORKSPACE_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "workspace", "datasets"
)

os.makedirs(WORKSPACE_DATASET_DIR, exist_ok=True)


def parse_matrix(s: str, n_cols: int):
    s = s.replace("\n", " ")
    s = s.replace("[", "")
    s = s.replace("]", "")
    numbers = np.fromstring(s, sep=" ")
    return numbers.reshape(-1, n_cols).tolist()


def parse_vector(s: str):
    s = s.replace("\n", " ")
    s = s.replace("[", "")
    s = s.replace("]", "")
    return np.fromstring(s, sep=" ").tolist()


def prepare_instance(instance_id: str):

    df = pd.read_csv(DATA_PATH)

    matches = df[df["instance_id"].str.strip() == instance_id.strip()]
    if matches.empty:
        raise ValueError(f"Instance '{instance_id}' not found.")

    row = matches.iloc[0]

    input_vars = ast.literal_eval(row["input_vars"])
    output_vars = ast.literal_eval(row["output_vars"])

    target_var = output_vars[0]

    train_input = parse_matrix(row["train_input"], len(input_vars))
    train_output = parse_vector(row["train_output"])

    data = pd.DataFrame(train_input, columns=input_vars)
    data[target_var] = train_output

    output_file = os.path.join(
        WORKSPACE_DATASET_DIR,
        f"{instance_id}.csv"
    )

    data.to_csv(output_file, index=False)

    print(f"Prepared dataset saved to: {output_file}")
    return output_file, target_var