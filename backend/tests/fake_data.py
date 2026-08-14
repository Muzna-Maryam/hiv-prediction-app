"""
Synthetic data shaped like AIDS_Classification.csv, used to produce a
throwaway trained model for tests - locally and in CI - without ever
needing the real dataset checked in anywhere.

DELIBERATELY HAS NO app.* IMPORTS. conftest.py needs to generate this
data and train a model BEFORE any app module gets imported, because
app.config reads its Settings (data paths, etc.) once, at import time -
if app.pipeline or app.config were imported first, the env vars this
fixture sets afterward would be too late to matter. Keeping this file
import-clean is what makes that ordering possible.
"""

import numpy as np
import pandas as pd


def make_fake_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "time": rng.integers(1, 1200, n),
            "trt": rng.integers(0, 4, n),
            "age": rng.integers(18, 70, n),
            "wtkg": rng.uniform(40, 120, n),
            "hemo": rng.integers(0, 2, n),
            "homo": rng.integers(0, 2, n),
            "drugs": rng.integers(0, 2, n),
            "karnof": rng.choice([70, 80, 90, 100], n),
            "oprior": rng.integers(0, 2, n),
            "z30": rng.integers(0, 2, n),
            "preanti": rng.integers(0, 2000, n),
            "gender": rng.integers(0, 2, n),
            "str2": rng.integers(0, 2, n),
            "strat": rng.integers(1, 4, n),
            "symptom": rng.integers(0, 2, n),
            "treat": rng.integers(0, 2, n),
            "offtrt": rng.integers(0, 2, n),
            "cd40": rng.uniform(0, 500, n),
            "cd420": rng.uniform(0, 500, n),
            "cd80": rng.uniform(0, 2000, n),
            "cd820": rng.uniform(0, 2000, n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n), name="infected")
    return X, y