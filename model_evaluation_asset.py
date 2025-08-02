import io
import pickle
import pandas as pd
from dagster import asset, AssetExecutionContext
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Dict

@asset(deps=["trained_model", "feature_engineered_data"])
def model_evaluation(
    context: AssetExecutionContext,
    trained_model: Dict[str, bytes],
    feature_engineered_data: Dict[str, pd.DataFrame]
) -> Dict[str, dict]:
    """
    Evaluate trained models for multiple stocks and return performance metrics,
    recent predictions, and trading signals for each stock.
    """
    
    evaluation_results = {}

    for symbol in trained_model:
        try:
            if symbol not in feature_engineered_data:
                context.log.warning(f"⚠️ Skipping {symbol}: No feature data available.")
                continue

            # Load model
            buffer = io.BytesIO(trained_model[symbol])
            model_package = pickle.load(buffer)

            model = model_package['model']
            scaler = model_package['scaler']
            feature_columns = model_package['feature_columns']

            data = feature_engineered_data[symbol].copy()
            data = data.sort_index()
            # ✅ FIXED: Use lowercase 'close' instead of 'Close'
            data['target'] = data['close'].shift(-1)
            clean_data = data.dropna()

            if len(clean_data) < 50:
                context.log.warning(f"⚠️ Skipping {symbol}: Not enough data for evaluation.")
                continue

            X = clean_data[feature_columns]
            y = clean_data['target']

            # Time series split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Scale features
            X_train_scaled = scaler.transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Predictions
            train_preds = model.predict(X_train_scaled)
            test_preds = model.predict(X_test_scaled)

            # Metrics
            train_mse = mean_squared_error(y_train, train_preds)
            test_mse = mean_squared_error(y_test, test_preds)
            train_r2 = r2_score(y_train, train_preds)
            test_r2 = r2_score(y_test, test_preds)
            train_rmse = np.sqrt(train_mse)
            test_rmse = np.sqrt(test_mse)
            train_mae = mean_absolute_error(y_train, train_preds)
            test_mae = mean_absolute_error(y_test, test_preds)

            # Trading signal
            # ✅ FIXED: Use lowercase 'close' instead of 'Close'
            current_price = clean_data['close'].iloc[-1]
            predicted_price = test_preds[-1]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100

            if price_change_pct > 2:
                signal = "BUY (STRONG)"
            elif price_change_pct > 0.5:
                signal = "BUY (WEAK)"
            elif price_change_pct < -2:
                signal = "SELL (STRONG)"
            elif price_change_pct < -0.5:
                signal = "SELL (WEAK)"
            else:
                signal = "HOLD (WEAK)"

            latest_date = clean_data.index[-1]
            if hasattr(latest_date, 'strftime'):
                latest_date = latest_date.strftime('%Y-%m-%d')

            # Logs
            context.log.info(f"📈 {symbol} — Signal: {signal} | Test R²: {test_r2:.4f}")

            evaluation_results[symbol] = {
                "training_metrics": {
                    'mse': train_mse,
                    'mae': train_mae,
                    'r2': train_r2,
                    'rmse': train_rmse
                },
                "test_metrics": {
                    'mse': test_mse,
                    'mae': test_mae,
                    'r2': test_r2,
                    'rmse': test_rmse
                },
                "model_analysis": {
                    'overfitting_check': {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'r2_difference': train_r2 - test_r2
                    },
                    'prediction_accuracy': {
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'rmse_ratio': test_rmse / train_rmse if train_rmse > 0 else float('inf')
                    }
                },
                "recent_predictions": {
                    'actual_values': y_test.tail(10).tolist(),
                    'predicted_values': test_preds[-10:].tolist(),
                    'dates': X_test.index[-10:].strftime('%Y-%m-%d').tolist()
                },
                "trading_prediction": {
                    'current_price': float(current_price),
                    'predicted_price': float(predicted_price),
                    'price_change': float(price_change),
                    'price_change_pct': float(price_change_pct),
                    'signal': signal,
                    'date': latest_date
                },
                "model": model,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "model_confidence": test_r2  # useful for prediction_alert
            }

        except Exception as e:
            context.log.warning(f"❌ Failed to evaluate model for {symbol}: {str(e)}")
            evaluation_results[symbol] = {"error": str(e)}

    return evaluation_results
