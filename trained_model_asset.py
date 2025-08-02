import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dagster import asset, AssetExecutionContext
import pickle
import io
from typing import Dict

@asset
def trained_model(context: AssetExecutionContext, feature_engineered_data: Dict[str, pd.DataFrame]) -> Dict[str, bytes]:
    """
    Train a Linear Regression model for each stock in the portfolio.
    
    This asset takes feature-engineered data for multiple stocks and trains a model for each one.
    Returns a dictionary of trained models (serialized).
    """

    trained_models = {}

    for symbol, data in feature_engineered_data.items():
        try:
            context.log.info(f"🚀 Training model for {symbol} with {len(data)} rows...")

            feature_columns = [
                'sma_5', 'sma_20', 'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'volume_sma_20'
            ]

            # Prepare data
            data = data.sort_index()
            data['target'] = data['close'].shift(-1)
            clean_data = data.dropna()

            if len(clean_data) < 50:
                raise ValueError(f"Not enough data for {symbol}. Need at least 50 rows.")

            X = clean_data[feature_columns]
            y = clean_data['target']

            # Time-based train-test split
            split_point = int(len(clean_data) * 0.8)
            X_train = X.iloc[:split_point]
            y_train = y.iloc[:split_point]
            X_val = X.iloc[split_point:]
            y_val = y.iloc[split_point:]

            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Model training
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            train_score = model.score(X_train_scaled, y_train)
            val_score = model.score(X_val_scaled, y_val)

            context.log.info(f"✅ {symbol} — Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")

            # Package model
            model_package = {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'train_score': train_score,
                'val_score': val_score,
                'training_samples': len(X_train)
            }

            # Serialize model package
            buffer = io.BytesIO()
            pickle.dump(model_package, buffer)
            trained_models[symbol] = buffer.getvalue()

        except Exception as e:
            context.log.warning(f"❌ Failed to train model for {symbol}: {e}")
            continue

    return trained_models
