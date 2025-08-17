# utils/infer.py
import json
import numpy as np
import pandas as pd
from joblib import load
import tensorflow as tf
import os
from .features import engineer

def load_artifacts(pre_path='preprocessor.joblib',
                   model_path='supercar_mlp.keras',
                   cols_path='feature_columns.json',
                   blend_config_path=None):
    """
    Load model artifacts. Supports both single MLP model and blended XGBoost+MLP models.
    
    Args:
        pre_path: Path to preprocessor
        model_path: Path to MLP model
        cols_path: Path to feature columns config
        blend_config_path: Path to blend configuration (optional)
    
    Returns:
        pre, model, num_cols, cat_cols, blend_config
    """
    pre = load(pre_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open(cols_path, 'r') as f:
        cols = json.load(f)
    num_cols, cat_cols = cols['num_cols'], cols['cat_cols']
    
    blend_config = None
    if blend_config_path and os.path.exists(blend_config_path):
        try:
            with open(blend_config_path, 'r') as f:
                blend_config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load blend config: {e}")
    
    return pre, model, num_cols, cat_cols, blend_config

def load_xgb_model(xgb_path):
    """Load XGBoost model if it exists."""
    try:
        import xgboost as xgb
        if os.path.exists(xgb_path):
            return xgb.Booster()
        else:
            print(f"Warning: XGBoost model not found at {xgb_path}")
            return None
    except ImportError:
        print("Warning: XGBoost not installed. Install with: pip install xgboost")
        return None

def predict_df(df: pd.DataFrame,
               pre=None, model=None,
               num_cols=None, cat_cols=None,
               blend_config=None, xgb_model=None) -> pd.DataFrame:
    """
    Make predictions using either single MLP model or blended XGBoost+MLP models.
    """
    if pre is None or model is None or num_cols is None or cat_cols is None:
        pre, model, num_cols, cat_cols, blend_config = load_artifacts()
    
    dfe = engineer(df)
    X = pre.transform(dfe[num_cols + cat_cols])
    
    # Get MLP predictions
    y_log_mlp = model.predict(X, verbose=0).squeeze()
    
    if blend_config and blend_config.get('has_xgb', False) and xgb_model:
        try:
            # Get XGBoost predictions
            import xgboost as xgb
            X_dmatrix = xgb.DMatrix(X)
            y_log_xgb = xgb_model.predict(X_dmatrix)
            
            # Blend predictions in log space
            w_mlp = blend_config['weights']['W_MLP']
            w_xgb = blend_config['weights']['W_XGB']
            y_log_blend = w_mlp * y_log_mlp + w_xgb * y_log_xgb
            
            # Apply calibration
            a = blend_config['calibration']['a']
            b = blend_config['calibration']['b']
            y = a * np.expm1(y_log_blend) + b
            
        except Exception as e:
            print(f"Warning: XGBoost prediction failed, falling back to MLP: {e}")
            y = np.expm1(y_log_mlp)
    else:
        # Use only MLP model
        y = np.expm1(y_log_mlp)
    
    # Apply clipping
    clip_floor = blend_config.get('clip_floor', 1000.0) if blend_config else 1000.0
    y = np.clip(y, clip_floor, None)
    
    return pd.DataFrame({'price': y}, index=df.index)

def predict_single_row(row_dict: dict,
                       pre=None, model=None,
                       num_cols=None, cat_cols=None,
                       blend_config=None, xgb_model=None) -> float:
    df = pd.DataFrame([row_dict], index=[0])
    return float(predict_df(df, pre, model, num_cols, cat_cols, blend_config, xgb_model)['price'].iloc[0]) 