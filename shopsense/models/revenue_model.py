import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_revenue_timeseries(transactions_df: pd.DataFrame, freq: str = "M") -> pd.Series:
    df = transactions_df.copy()
    
    # Calculate net spend accounting for discounts and returns
    df['net_spend'] = df['unit_price'] * df['quantity'] * (1 - df['discount_pct'])
    df['net_spend'] = np.where(df['return_flag'] == 1, 0, df['net_spend'])
    
    # Set transaction_date as index
    df.set_index('transaction_date', inplace=True)
    
    # Resample and sum
    series = df['net_spend'].resample(freq).sum()
    
    # Ensure index is standard datetime for statsmodels
    series.index = series.index.to_timestamp(how='end') if isinstance(series.index, pd.PeriodIndex) else series.index
    
    return series

def test_stationarity(series: pd.Series) -> dict:
    # Drop any NaNs before testing
    clean_series = series.dropna()
    
    # Perform Augmented Dickey-Fuller test
    result = adfuller(clean_series, autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    is_stationary = p_value < 0.05
    recommended_differencing = 0 if is_stationary else 1
    
    return {
        "adf_statistic": float(adf_stat),
        "p_value": float(p_value),
        "is_stationary": bool(is_stationary),
        "critical_values": {k: float(v) for k, v in critical_values.items()},
        "recommended_differencing": int(recommended_differencing)
    }

def train_sarima_model(series: pd.Series, order: tuple, seasonal_order: tuple) -> tuple:
    # Fit the SARIMA model
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = model.fit(disp=False)
    
    # Extract fitted values
    fitted_values = sarima_result.fittedvalues
    
    return sarima_result, fitted_values

def forecast_sarima(sarima_result, steps: int) -> pd.DataFrame:
    # Generate forecast
    forecast_obj = sarima_result.get_forecast(steps=steps)
    
    # Extract mean forecast and confidence intervals
    predicted_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05) # 95% CI
    
    # Construct the required DataFrame
    df = pd.DataFrame({
        'forecast_date': predicted_mean.index,
        'predicted_revenue': predicted_mean.values,
        'lower_ci_95': conf_int.iloc[:, 0].values,
        'upper_ci_95': conf_int.iloc[:, 1].values
    })
    
    return df.reset_index(drop=True)

def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    # Ensure alignment
    df = pd.DataFrame({'actual': actual, 'predicted': predicted}).dropna()
    y_true = df['actual']
    y_pred = df['predicted']
    
    # Standard metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Handle MAPE (exclude actual = 0)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan
        
    # Handle sMAPE (symmetric MAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_mask = denominator != 0
    if smape_mask.sum() > 0:
        smape = np.mean(np.abs(y_true[smape_mask] - y_pred[smape_mask]) / denominator[smape_mask]) * 100
    else:
        smape = np.nan
        
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
        "smape": float(smape),
        "r2": float(r2)
    }