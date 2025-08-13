# utils/infer.py
import json
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf
from .features import engineer

def load_artifacts(pre_path='preprocessor.joblib',
                   model_path='supercar_mlp.keras',
                   cols_path='feature_columns.json'):
    pre = load(pre_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(cols_path, 'r') as f:
        cols = json.load(f)
    num_cols, cat_cols = cols['num_cols'], cols['cat_cols']
    return pre, model, num_cols, cat_cols

def predict_df(df: pd.DataFrame,
               pre=None, model=None,
               num_cols=None, cat_cols=None) -> pd.DataFrame:
    if pre is None or model is None or num_cols is None or cat_cols is None:
        pre, model, num_cols, cat_cols = load_artifacts()
    dfe = engineer(df)
    X = pre.transform(dfe[num_cols + cat_cols])
    y_log = model.predict(X, verbose=0).squeeze()
    y = np.expm1(y_log)
    y = np.clip(y, 1000.0, None)
    return pd.DataFrame({'price': y}, index=df.index)

def predict_single_row(row_dict: dict,
                       pre=None, model=None,
                       num_cols=None, cat_cols=None) -> float:
    df = pd.DataFrame([row_dict], index=[0])
    return float(predict_df(df, pre, model, num_cols, cat_cols)['price'].iloc[0]) 