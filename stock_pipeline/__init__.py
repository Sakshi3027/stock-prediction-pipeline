import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List
import concurrent.futures
import numpy as np
from dagster import (
    asset, 
    AssetCheckResult, 
    asset_check, 
    Definitions,
    ScheduleDefinition,
    DefaultScheduleStatus,
    define_asset_job,
    Config
)
from trained_model_asset import trained_model
from model_evaluation_asset import model_evaluation

# Define the tickers we want to analyze
TICKERS = ["AAPL", "TSLA", "NVDA", "GOOGL", "MSFT", "AMZN", "META"]

def standardize_column_names(df):
    """Standardize column names to lowercase and handle multi-index"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Convert to lowercase and strip whitespace
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # Map common variations to standard names
    column_mapping = {
        'adj close': 'close',
        'adjclose': 'close',
        'adj_close': 'close'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df

def calculate_technical_indicators(data, ticker=None):
    """Calculate technical indicators with extensive debugging"""
    try:
        # Ensure we have a copy to avoid modifying original data
        df = data.copy()
        
        if ticker:
            print(f"🔍 [{ticker}] Input data shape: {df.shape}")
            print(f"🔍 [{ticker}] Date range: {df.index[0]} to {df.index[-1]}")
            print(f"🔍 [{ticker}] Close price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            print(f"🔍 [{ticker}] Latest close: ${df['close'].iloc[-1]:.2f}")
        
        # Basic validation
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in basic price data
        initial_len = len(df)
        df = df.dropna(subset=required_cols)
        
        if ticker and len(df) < initial_len:
            print(f"🔧 [{ticker}] Removed {initial_len - len(df)} rows with NaN values")
        
        if len(df) < 50:
            raise ValueError(f"Insufficient data: {len(df)} rows")
        
        # Calculate features step by step with debugging
        if ticker:
            print(f"📊 [{ticker}] Calculating technical indicators...")
        
        # 1. Basic price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # 2. Moving averages
        df['sma_5'] = df['close'].rolling(window=5, min_periods=5).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=20).mean()
        df['sma_50'] = df['close'].rolling(window=50, min_periods=50).mean()
        
        # 3. Volume features
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, np.nan)
        df['volume_ratio'] = df['volume_ratio'].clip(upper=10)
        
        # 4. Volatility and Bollinger Bands
        df['volatility'] = df['close'].rolling(window=20, min_periods=20).std()
        df['bb_upper'] = df['sma_20'] + (df['volatility'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['volatility'] * 2)
        
        # 5. RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].clip(0, 100)
        
        # 6. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # 7. Clean infinite and extreme values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # 8. Forward fill NaN values for technical indicators
        technical_cols = ['sma_5', 'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 
                         'volume_ratio', 'volatility', 'bb_upper', 'bb_lower', 'macd_histogram']
        
        for col in technical_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # 9. Final cleanup
        critical_cols = ['close', 'volume', 'price_change', 'sma_20', 'rsi']
        df = df.dropna(subset=critical_cols)
        
        # DEBUG: Show final feature values for this ticker
        if ticker and len(df) > 0:
            latest_row = df.iloc[-1]
            print(f"🔍 [{ticker}] FINAL FEATURES:")
            print(f"   Close: ${latest_row['close']:.2f}")
            print(f"   SMA_5: ${latest_row.get('sma_5', 0):.2f}")
            print(f"   SMA_20: ${latest_row.get('sma_20', 0):.2f}")
            print(f"   RSI: {latest_row.get('rsi', 0):.2f}")
            print(f"   Volume Ratio: {latest_row.get('volume_ratio', 0):.2f}")
            print(f"   MACD: {latest_row.get('macd', 0):.4f}")
        
        if ticker:
            print(f"✅ [{ticker}] Technical indicators complete. Final shape: {df.shape}")
        
        return df
        
    except Exception as e:
        if ticker:
            print(f"❌ [{ticker}] Error calculating technical indicators: {str(e)}")
        raise

def download_single_ticker(ticker: str, start_date: str, end_date: str) -> tuple:
    """Download data for a single ticker with debugging"""
    try:
        print(f"📊 Downloading {ticker} data from {start_date} to {end_date}")
        
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print(f"❌ No data found for {ticker}")
            return ticker, None
        
        print(f"🔍 [{ticker}] Raw download shape: {data.shape}")
        print(f"🔍 [{ticker}] Raw columns: {data.columns.tolist()}")
        
        # Show sample of raw data
        if len(data) > 0:
            latest = data.iloc[-1]
            print(f"🔍 [{ticker}] Raw latest close: ${latest.iloc[3] if len(latest) > 3 else 'N/A'}")  # Close is usually 4th column
        
        # Standardize column names
        data = standardize_column_names(data)
        
        print(f"🔍 [{ticker}] After standardization: {data.columns.tolist()}")
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"❌ {ticker}: Missing columns: {missing_columns}")
            return ticker, None
        
        # Remove empty rows
        data = data.dropna(how='all')
        
        if data.empty:
            print(f"❌ {ticker}: No data after cleaning")
            return ticker, None
        
        # Show final raw data info
        print(f"✅ {ticker}: Downloaded {len(data)} days")
        print(f"🔍 [{ticker}] Final close price: ${data['close'].iloc[-1]:.2f}")
        print(f"🔍 [{ticker}] Close price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
        
        return ticker, data
        
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return ticker, None

@asset
def raw_stock_data():
    """Download recent stock data for all tickers with debugging"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"🚀 Starting download for {len(TICKERS)} tickers...")
    
    all_data = {}
    
    # Process sequentially for better debugging
    for ticker in TICKERS:
        ticker_result, data_frame = download_single_ticker(ticker, start_str, end_str)
        if data_frame is not None:
            all_data[ticker] = data_frame
            print(f"✅ {ticker}: Added to dataset")
        else:
            print(f"⚠️ {ticker}: Skipped due to download failure")
        print("-" * 50)
    
    print(f"✅ Successfully downloaded data for {len(all_data)} out of {len(TICKERS)} tickers")
    
    # DEBUG: Show summary of all downloaded data
    print("\n📊 DOWNLOAD SUMMARY:")
    for ticker, data in all_data.items():
        if not data.empty:
            print(f"   {ticker}: {len(data)} rows, latest close: ${data['close'].iloc[-1]:.2f}")
    
    return all_data

def engineer_features_single_ticker(ticker: str, data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for a single ticker with debugging"""
    try:
        if data is None or data.empty:
            print(f"❌ {ticker}: No input data")
            return pd.DataFrame()
            
        print(f"\n🔧 FEATURE ENGINEERING FOR {ticker}")
        print("=" * 60)
        
        # Standardize columns (should already be done, but just in case)
        data = standardize_column_names(data)
        
        # Calculate technical indicators with debugging
        engineered_data = calculate_technical_indicators(data, ticker=ticker)
        
        if engineered_data.empty:
            print(f"❌ {ticker}: No data after feature engineering")
            return pd.DataFrame()
        
        print(f"✅ {ticker}: Feature engineering complete - {len(engineered_data)} rows")
        print("=" * 60)
        return engineered_data
        
    except Exception as e:
        print(f"❌ {ticker}: Feature engineering failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

@asset
def feature_engineered_data(raw_stock_data):
    """Create engineered features with debugging"""
    engineered_data = {}
    
    print(f"\n🔧 Starting feature engineering for {len(raw_stock_data)} tickers...")
    
    # Process sequentially for better debugging
    for ticker, data in raw_stock_data.items():
        result = engineer_features_single_ticker(ticker, data)
        engineered_data[ticker] = result
        
        if not result.empty:
            print(f"✅ {ticker}: Processed successfully")
        else:
            print(f"⚠️ {ticker}: No data after processing")
    
    print(f"\n✅ Feature engineering complete for {len(engineered_data)} tickers")
    
    # DEBUG: Show feature engineering summary
    print("\n📊 FEATURE ENGINEERING SUMMARY:")
    for ticker, data in engineered_data.items():
        if not data.empty:
            latest = data.iloc[-1]
            print(f"   {ticker}: Close=${latest['close']:.2f}, SMA20=${latest.get('sma_20', 0):.2f}, RSI={latest.get('rsi', 0):.1f}")
    
    return engineered_data

@asset_check(asset=feature_engineered_data)
def check_feature_quality(feature_engineered_data):
    """Check feature quality with realistic expectations for time series data"""
    all_passed = True
    results = []
    
    print("\n🔍 FEATURE QUALITY CHECK:")
    print("=" * 50)
    
    for ticker, data in feature_engineered_data.items():
        if data.empty:
            results.append(f"{ticker}: No data")
            all_passed = False
            print(f"❌ {ticker}: No data available")
            continue
        
        # Check for missing values in CRITICAL columns only
        critical_columns = ['close', 'volume', 'price_change']
        critical_missing = data[critical_columns].isnull().sum().sum()
        
        # Allow some missing values in technical indicators (first 50 rows are expected)
        total_missing = data.isnull().sum().sum()
        acceptable_missing_threshold = len(data) * 0.1  # Allow up to 10% missing
        
        # Check for infinite values
        inf_count = data.isin([float('inf'), float('-inf')]).sum().sum()
        
        # Check data sufficiency
        has_sufficient_data = len(data) > 50  # Need at least 50 days
        
        # Check feature ranges for non-NaN values
        try:
            volume_ratio_valid = True
            rsi_valid = True
            
            if 'volume_ratio' in data.columns:
                vr_clean = data['volume_ratio'].dropna()
                if len(vr_clean) > 0:
                    volume_ratio_valid = (vr_clean >= 0).all() and (vr_clean <= 10).all()
            
            if 'rsi' in data.columns:
                rsi_clean = data['rsi'].dropna()
                if len(rsi_clean) > 0:
                    rsi_valid = (rsi_clean >= 0).all() and (rsi_clean <= 100).all()
                    
        except Exception as e:
            print(f"⚠️ {ticker}: Error checking feature ranges: {e}")
            volume_ratio_valid = rsi_valid = False
        
        # Updated pass criteria
        ticker_passed = (
            critical_missing == 0 and  # No missing in critical columns
            inf_count == 0 and  # No infinite values
            has_sufficient_data and  # Sufficient data points
            total_missing <= acceptable_missing_threshold and  # Reasonable missing values
            volume_ratio_valid and 
            rsi_valid
        )
        
        status = "✅" if ticker_passed else "❌"
        missing_pct = (total_missing / (len(data) * len(data.columns))) * 100
        
        print(f"{status} {ticker}: {len(data)} rows, {critical_missing} critical missing, "
              f"{total_missing} total missing ({missing_pct:.1f}%), {inf_count} infinite")
        
        if not ticker_passed:
            all_passed = False
            if critical_missing > 0:
                print(f"   ⚠️ Critical missing values in: {critical_columns}")
            if total_missing > acceptable_missing_threshold:
                print(f"   ⚠️ Too many missing values: {total_missing} > {acceptable_missing_threshold:.0f}")
            if inf_count > 0:
                print(f"   ⚠️ Infinite values detected")
            
        results.append(f"{ticker}: {len(data)} rows, {missing_pct:.1f}% missing, {inf_count} infinite")
    
    print("=" * 50)
    return AssetCheckResult(
        passed=all_passed,
        description=f"Feature quality check: {'; '.join(results)}"
    )

def generate_single_prediction(ticker: str, model_data: dict, raw_data: pd.DataFrame) -> dict:
    """Generate prediction with extensive debugging"""
    try:
        print(f"\n🎯 GENERATING PREDICTION FOR {ticker}")
        print("=" * 70)
        
        # Validate inputs
        if not model_data or "model" not in model_data:
            raise ValueError("No trained model available")
        
        if raw_data is None or raw_data.empty:
            raise ValueError("No raw data available")
        
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_columns = model_data["feature_columns"]
        
        print(f"🔍 [{ticker}] Model info:")
        print(f"   Required features: {feature_columns}")
        print(f"   Scaler type: {type(scaler).__name__}")
        print(f"   Model type: {type(model).__name__}")
        
        # Create a completely independent copy to avoid data contamination
        prediction_data = raw_data.copy(deep=True)
        
        print(f"🔍 [{ticker}] Input data:")
        print(f"   Shape: {prediction_data.shape}")
        print(f"   Date range: {prediction_data.index[0]} to {prediction_data.index[-1]}")
        print(f"   Columns: {prediction_data.columns.tolist()}")
        
        # Standardize column names
        prediction_data = standardize_column_names(prediction_data)
        
        # Calculate technical indicators
        prediction_data = calculate_technical_indicators(prediction_data, ticker=ticker)
        
        if prediction_data.empty:
            raise ValueError("No data after technical indicator calculation")
        
        # Verify we have all required features
        missing_features = [col for col in feature_columns if col not in prediction_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Get the most recent complete data point
        feature_data = prediction_data[feature_columns + ['close']].dropna()
        
        if len(feature_data) == 0:
            raise ValueError("No complete feature data available")
        
        # Use the most recent data point
        latest_features = feature_data[feature_columns].iloc[-1]
        current_price = feature_data['close'].iloc[-1]
        prediction_date = feature_data.index[-1].strftime('%Y-%m-%d')
        
        print(f"🔍 [{ticker}] Prediction input:")
        print(f"   Current price: ${current_price:.2f}")
        print(f"   Date: {prediction_date}")
        print(f"   Feature values:")
        for i, col in enumerate(feature_columns):
            print(f"      {col}: {latest_features[col]:.6f}")
        
        # Convert to numpy array and reshape
        feature_array = latest_features.values.reshape(1, -1)
        
        print(f"🔍 [{ticker}] Feature array shape: {feature_array.shape}")
        print(f"🔍 [{ticker}] Feature array values: {feature_array[0]}")
        
        # Scale features
        try:
            scaled_features = scaler.transform(feature_array)
            print(f"🔍 [{ticker}] Scaled features: {scaled_features[0]}")
        except Exception as e:
            raise ValueError(f"Feature scaling failed: {str(e)}")
        
        # Make prediction
        try:
            predicted_price = model.predict(scaled_features)[0]
            print(f"🔍 [{ticker}] Raw prediction: {predicted_price}")
        except Exception as e:
            raise ValueError(f"Model prediction failed: {str(e)}")
        
        # Calculate price change
        price_change_percent = ((predicted_price - current_price) / current_price) * 100
        
        # Generate trading signal
        if price_change_percent > 2.0:
            signal = "BUY"
            signal_strength = "STRONG" if price_change_percent > 4.0 else "MODERATE"
        elif price_change_percent < -2.0:
            signal = "SELL"
            signal_strength = "STRONG" if price_change_percent < -4.0 else "MODERATE"
        else:
            signal = "HOLD"
            signal_strength = "WEAK"
        
        # Get model confidence
        confidence = model_data.get('model_confidence', model_data.get('val_score', 0.5))
        
        result = {
            "ticker": ticker,
            "date": prediction_date,
            "current_price": float(current_price),
            "predicted_price": float(predicted_price),
            "price_change_percent": float(price_change_percent),
            "signal": signal,
            "signal_strength": signal_strength,
            "model_confidence": float(confidence) if confidence is not None else 0.5,
            "data_points_used": len(feature_data),
            "success": True
        }
        
        print(f"🎯 [{ticker}] FINAL RESULT:")
        print(f"   Prediction: ${predicted_price:.2f} ({price_change_percent:+.2f}%)")
        print(f"   Signal: {signal} ({signal_strength})")
        print(f"   Confidence: {confidence:.1%}")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        print(f"❌ [{ticker}] Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return {
            "ticker": ticker,
            "error": str(e),
            "success": False
        }

@asset(deps=[model_evaluation, raw_stock_data])
def multi_ticker_predictions(
    model_evaluation: Dict[str, dict], 
    raw_stock_data: Dict[str, pd.DataFrame]
) -> dict:
    """Generate predictions with debugging"""
    
    print("\n🚀 STARTING MULTI-TICKER PREDICTION GENERATION")
    print("=" * 80)
    
    predictions = {}
    
    # Process sequentially for better debugging
    for ticker in TICKERS:
        print(f"\nProcessing {ticker}...")
        
        if ticker not in model_evaluation:
            print(f"⚠️ {ticker}: No model evaluation data")
            predictions[ticker] = {"ticker": ticker, "error": "No model data", "success": False}
            continue
            
        if ticker not in raw_stock_data:
            print(f"⚠️ {ticker}: No raw stock data")
            predictions[ticker] = {"ticker": ticker, "error": "No raw data", "success": False}
            continue
        
        if not model_evaluation[ticker] or not isinstance(raw_stock_data[ticker], pd.DataFrame):
            print(f"⚠️ {ticker}: Invalid model or data")
            predictions[ticker] = {"ticker": ticker, "error": "Invalid data", "success": False}
            continue
        
        try:
            result = generate_single_prediction(
                ticker, 
                model_evaluation[ticker], 
                raw_stock_data[ticker]
            )
            predictions[ticker] = result
            
            if result.get("success", False):
                print(f"✅ {ticker}: Success - {result['signal']} ({result['price_change_percent']:+.2f}%)")
            else:
                print(f"❌ {ticker}: Failed - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error processing {ticker}: {str(e)}")
            predictions[ticker] = {"ticker": ticker, "error": str(e), "success": False}
    
    print("\n🎯 PREDICTION GENERATION COMPLETE")
    print("=" * 80)
    
    # Show summary
    successful = sum(1 for p in predictions.values() if p.get("success", False))
    print(f"📊 Results: {successful}/{len(predictions)} successful predictions")
    
    return predictions

@asset(deps=[multi_ticker_predictions])
def prediction_summary_report(multi_ticker_predictions: dict) -> dict:
    """Generate summary with detailed debugging"""
    
    successful_predictions = {k: v for k, v in multi_ticker_predictions.items() if v.get("success", False)}
    failed_predictions = {k: v for k, v in multi_ticker_predictions.items() if not v.get("success", False)}
    
    # Analyze signals
    buy_signals = {k: v for k, v in successful_predictions.items() if v["signal"] == "BUY"}
    sell_signals = {k: v for k, v in successful_predictions.items() if v["signal"] == "SELL"}
    hold_signals = {k: v for k, v in successful_predictions.items() if v["signal"] == "HOLD"}
    
    print("\n" + "=" * 80)
    print("🚨 MULTI-TICKER DAILY TRADING ALERT 🚨")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"📊 Total Analyzed: {len(multi_ticker_predictions)}")
    print(f"✅ Successful: {len(successful_predictions)}")
    print(f"❌ Failed: {len(failed_predictions)}")
    print("=" * 80)
    
    print("📈 SIGNAL SUMMARY:")
    print(f"   🟢 BUY: {len(buy_signals)}")
    print(f"   🔴 SELL: {len(sell_signals)}")
    print(f"   🟡 HOLD: {len(hold_signals)}")
    print("-" * 80)
    
    # Show successful predictions with debugging info
    if successful_predictions:
        print("📊 DETAILED PREDICTIONS:")
        for ticker, pred in successful_predictions.items():
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[pred["signal"]]
            print(f"{signal_emoji} {ticker}:")
            print(f"   Current: ${pred['current_price']:.2f}")
            print(f"   Predicted: ${pred['predicted_price']:.2f}")
            print(f"   Change: {pred['price_change_percent']:+.2f}%")
            print(f"   Signal: {pred['signal']} ({pred['signal_strength']})")
            print(f"   Confidence: {pred['model_confidence']:.1%}")
            print(f"   Data Points: {pred['data_points_used']}")
            print()
    
    # Show failures
    if failed_predictions:
        print("❌ FAILED PREDICTIONS:")
        for ticker, pred in failed_predictions.items():
            print(f"   {ticker}: {pred.get('error', 'Unknown error')}")
        print()
    
    print("=" * 80)
    
    # Summary statistics
    summary = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "total_tickers": len(multi_ticker_predictions),
        "successful_predictions": len(successful_predictions),
        "failed_predictions": len(failed_predictions),
        "signal_breakdown": {
            "buy": len(buy_signals),
            "sell": len(sell_signals),
            "hold": len(hold_signals)
        },
        "predictions": successful_predictions,
        "errors": failed_predictions
    }
    
    return summary

@asset
def market_status():
    """Check market status"""
    from datetime import datetime, timezone
    import pytz
    
    et = pytz.timezone('US/Eastern')
    now_et = datetime.now(et)
    
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = now_et.weekday() < 5
    is_market_hours = market_open <= now_et <= market_close
    is_market_open = is_weekday and is_market_hours
    
    status = {
        "is_market_open": is_market_open,
        "current_time_et": now_et.strftime('%Y-%m-%d %H:%M:%S ET'),
        "market_open_time": market_open.strftime('%H:%M ET'),
        "market_close_time": market_close.strftime('%H:%M ET')
    }
    
    print(f"📈 Market Status: {'OPEN' if is_market_open else 'CLOSED'}")
    print(f"   Current Time: {status['current_time_et']}")
    
    return status

@asset(deps=[prediction_summary_report, market_status])
def enhanced_multi_ticker_alert(
    prediction_summary_report: dict,
    market_status: dict
) -> dict:
    """Generate enhanced alert"""
    
    print("\n" + "=" * 90)
    print("🚨 ENHANCED MULTI-TICKER TRADING ALERT 🚨")
    print("=" * 90)
    print(f"📅 Date: {prediction_summary_report['date']}")
    print(f"⏰ Time: {market_status['current_time_et']}")
    print(f"📈 Market: {'OPEN' if market_status['is_market_open'] else 'CLOSED'}")
    print("-" * 90)
    
    print("💼 PORTFOLIO OVERVIEW:")
    print(f"   📊 Total Analyzed: {prediction_summary_report['total_tickers']}")
    print(f"   ✅ Successful: {prediction_summary_report['successful_predictions']}")
    print(f"   🟢 BUY Signals: {prediction_summary_report['signal_breakdown']['buy']}")
    print(f"   🔴 SELL Signals: {prediction_summary_report['signal_breakdown']['sell']}")
    print(f"   🟡 HOLD Signals: {prediction_summary_report['signal_breakdown']['hold']}")
    print("=" * 90)
    
    enhanced_result = {
        **prediction_summary_report,
        "market_status": market_status,
        "enhanced_insights": {
            "total_signals": prediction_summary_report['signal_breakdown']['buy'] + 
                           prediction_summary_report['signal_breakdown']['sell'],
            "action_recommended": prediction_summary_report['signal_breakdown']['buy'] > 0 or 
                                prediction_summary_report['signal_breakdown']['sell'] > 0,
            "market_timing": "OPEN" if market_status['is_market_open'] else "CLOSED"
        }
    }
    
    return enhanced_result

# Job and Schedule Definitions
daily_multi_ticker_job = define_asset_job(
    name="daily_multi_ticker_job",
    selection=[
        "raw_stock_data",
        "feature_engineered_data", 
        "trained_model",
        "model_evaluation",
        "multi_ticker_predictions",
        "prediction_summary_report",
        "market_status",
        "enhanced_multi_ticker_alert"
    ],
    description="Daily multi-ticker stock prediction pipeline with debugging"
)

daily_multi_ticker_schedule = ScheduleDefinition(
    name="daily_multi_ticker_schedule",
    cron_schedule="30 8 * * 1-5",  # 8:30 AM ET weekdays
    job=daily_multi_ticker_job,
    default_status=DefaultScheduleStatus.RUNNING,
    description="Daily multi-ticker predictions at 8:30 AM ET"
)

defs = Definitions(
    assets=[
        raw_stock_data, 
        feature_engineered_data, 
        trained_model, 
        model_evaluation, 
        multi_ticker_predictions,
        prediction_summary_report,
        market_status,
        enhanced_multi_ticker_alert
    ],
    asset_checks=[check_feature_quality],
    jobs=[daily_multi_ticker_job],
    schedules=[daily_multi_ticker_schedule]
)