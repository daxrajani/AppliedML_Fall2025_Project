import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    np.random.seed(42)
    source = Path("Prototype.csv")
    target = Path("Prototype_augmented.csv")

    if not source.exists():
        raise FileNotFoundError("Prototype.csv not found. Run this script from project root.")

    df = pd.read_csv(source)
    if "prognosis" not in df.columns:
        raise ValueError("Dataset must contain prognosis column.")

    symptom_columns = [column for column in df.columns if column != "prognosis"]
    for col in symptom_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    augmented_rows = []
    for _, row in df.iterrows():
        base = row.copy()
        augmented_rows.append(base)

        for _ in range(2):
            noisy = base.copy()
            active = [sym for sym in symptom_columns if noisy[sym] == 1]
            inactive = [sym for sym in symptom_columns if noisy[sym] == 0]

            # Drop at most one active symptom to simulate imperfect reporting.
            if active and np.random.rand() < 0.35:
                noisy[np.random.choice(active)] = 0

            # Add at most one inactive symptom to simulate symptom overlap.
            if inactive and np.random.rand() < 0.20:
                noisy[np.random.choice(inactive)] = 1

            augmented_rows.append(noisy)

    out_df = pd.DataFrame(augmented_rows)
    out_df.to_csv(target, index=False)
    print(f"Created {target} with {len(out_df)} rows (from {len(df)} base rows).")


if __name__ == "__main__":
    main()
