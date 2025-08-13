# utils/features.py
import numpy as np
import pandas as pd

REF_YEAR = 2025

def safe_div(a, b):
    b = np.where((b == 0) | (pd.isna(b)), np.nan, b)
    return a / b

def parse_last_service_date(df, col='last_service_date'):
    if col in df.columns:
        return pd.to_datetime(df[col], errors='coerce')
    else:
        return pd.Series(pd.NaT, index=df.index)

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Days since last service (reference end-of-2025)
    d = parse_last_service_date(df, 'last_service_date')
    ds = (pd.Timestamp('2025-12-31') - d).dt.days
    df['days_since_service'] = ds.fillna(ds.median() if ds.notna().any() else 0)

    # Car age
    df['age'] = REF_YEAR - df.get('year', np.nan)

    # Power/weight & performance index
    if {'horsepower','weight_kg'}.issubset(df.columns):
        df['hp_per_kg'] = safe_div(df['horsepower'], df['weight_kg'])
    if {'torque','weight_kg'}.issubset(df.columns):
        df['torque_per_kg'] = safe_div(df['torque'], df['weight_kg'])
    if {'zero_to_60_s','top_speed_mph'}.issubset(df.columns):
        df['perf_index'] = safe_div(df['top_speed_mph'], df['zero_to_60_s'])

    # Mileage normalization
    if 'mileage' in df.columns:
        df['mileage_per_year'] = safe_div(df['mileage'], np.maximum(df['age'], 1))
    else:
        df['mileage_per_year'] = 0

    # Damage signals
    df['has_damage'] = df.get('damage', 0).fillna(0).astype(int)
    df['damage_cost_log1p'] = np.log1p(df.get('damage_cost', 0).fillna(0))
    if 'damage_type' in df.columns:
        sev_map = {'none':0, 'minor':1, 'moderate':2, 'major':3}
        df['damage_severity'] = df['damage_type'].astype(str).str.lower().map(sev_map).fillna(1)
    else:
        df['damage_severity'] = 0

    # Flags -> int
    for c in ['has_warranty','limited_edition','aero_package','carbon_fiber_body','non_original_parts']:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    # Final numeric NA guard
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df 