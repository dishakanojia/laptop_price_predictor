"""Cleaning + feature engineering for the laptop dataset.

Shared by the notebook (training) and app.py (prediction) so both
always turn raw specs into features the exact same way.
"""
import re

import numpy as np
import pandas as pd

NUMERIC_FEATURES = ["Ram", "Weight", "PPI", "Cpu_GHz", "SSD", "HDD", "Touchscreen", "IPS"]
CATEGORICAL_FEATURES = ["Company", "TypeName", "Cpu_brand", "Gpu_brand", "OS"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Price"


def ppi(x_res, y_res, inches):
    """Pixels per inch = diagonal pixel count / diagonal screen size."""
    return np.sqrt(x_res**2 + y_res**2) / inches


def cpu_brand(cpu: str) -> str:
    """Collapse ~118 CPU strings into 5 groups the model can learn from."""
    name = " ".join(cpu.split()[:3])
    if name in ("Intel Core i7", "Intel Core i5", "Intel Core i3"):
        return name
    if cpu.startswith("Intel"):
        return "Other Intel"
    if cpu.startswith("AMD"):
        return "AMD"
    return "Other"


def os_group(os_name: str) -> str:
    if os_name.startswith("Windows"):
        return "Windows"
    if os_name in ("macOS", "Mac OS X"):
        return "Mac"
    return "Other/Linux/No OS"


def _storage_gb(memory: str, kind: str) -> float:
    """Sum the GB of one storage type, e.g. '128GB SSD +  1TB HDD' -> SSD=128, HDD=1000."""
    total = 0.0
    for part in memory.split("+"):
        part = part.strip()
        if kind in part:
            size = re.search(r"([\d.]+)\s*(GB|TB)", part)
            if size:
                value = float(size.group(1))
                total += value * 1000 if size.group(2) == "TB" else value
    return total


def clean_and_engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw CSV columns -> clean, model-ready feature table."""
    df = raw.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.drop_duplicates()

    # --- cleaning: text units -> numbers ---
    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

    # --- feature engineering: screen ---
    df["Touchscreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS").astype(int)
    res = df["ScreenResolution"].str.extract(r"(\d+)x(\d+)").astype(int)
    df["PPI"] = ppi(res[0], res[1], df["Inches"]).round(2)

    # --- feature engineering: processor, storage, GPU, OS ---
    df["Cpu_brand"] = df["Cpu"].apply(cpu_brand)
    df["Cpu_GHz"] = df["Cpu"].str.extract(r"([\d.]+)GHz").astype(float)
    df["SSD"] = df["Memory"].apply(lambda m: _storage_gb(m, "SSD") + _storage_gb(m, "Flash"))
    df["HDD"] = df["Memory"].apply(lambda m: _storage_gb(m, "HDD") + _storage_gb(m, "Hybrid"))
    df["Gpu_brand"] = df["Gpu"].str.split().str[0]
    df["OS"] = df["OpSys"].apply(os_group)

    # rare one-off rows the model can't learn anything from
    df = df[(df["Gpu_brand"] != "ARM") & (df["Cpu_brand"] != "Other")]

    return df[FEATURES + [TARGET]].reset_index(drop=True)
