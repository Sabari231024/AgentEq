import os
import json
import uuid
from typing import Dict, Any, List

import pandas as pd
from pysr import PySRRegressor


class PySRTool:
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = workspace_dir
        self.datasets_dir = os.path.join(workspace_dir, "datasets")
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.datasets_dir, exist_ok=True)

    def run(
        self,
        input_file: str,
        target_column: str,
        binary_operators: List[str] = None,
        unary_operators: List[str] = None,
        niterations: int = 40,
        population_size: int = 1000,
        **kwargs,  # absorb unknown args the LLM may hallucinate (max_size, iterations, etc.)
    ) -> Dict[str, Any]:

        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]

        if unary_operators is None:
            unary_operators = []

        # Search in datasets/ first, then workspace/ root as fallback
        file_path = os.path.join(self.datasets_dir, input_file)
        if not os.path.exists(file_path):
            file_path = os.path.join(self.workspace_dir, input_file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Dataset not found. Searched:\n"
                f"  1. {os.path.join(self.datasets_dir, input_file)}\n"
                f"  2. {os.path.join(self.workspace_dir, input_file)}"
            )

        if kwargs:
            print(f"    ⚠️  PySRTool ignoring unknown kwargs: {list(kwargs.keys())}")

        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Drop rows with NaN in input features or target — PySR will produce
        # nonsensical results if NaN values are silently included in X or y.
        n_before = len(df)
        df = df.dropna(subset=[target_column] + [c for c in df.columns if c != target_column])
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            print(f"    ⚠️  PySRTool dropped {n_dropped} row(s) with NaN values before fitting.")

        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        model = PySRRegressor(
            niterations=niterations,
            population_size=population_size,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            model_selection="best",
            verbosity=0,
        )

        model.fit(X, y)

        # get_best() returns a pandas Series/Row — extract the equation string
        best_row = model.get_best()
        best_equation = str(best_row["equation"]) if hasattr(best_row, "__getitem__") else str(best_row)

        y_pred = model.predict(X)
        mape = float(
            (abs((y - y_pred) / (abs(y) + 1e-8))).mean() * 100
        )

        result_data = {
            "equation": best_equation,
            "mape": mape,
            "binary_operators": binary_operators,
            "unary_operators": unary_operators,
            "ignored_kwargs": list(kwargs.keys()) if kwargs else [],
        }

        return result_data
