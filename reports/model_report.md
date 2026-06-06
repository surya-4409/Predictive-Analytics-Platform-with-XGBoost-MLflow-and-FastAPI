# ShopSense Analytics: Model Performance Report

## 1. Executive Summary
This report summarizes the performance of the predictive models deployed for ShopSense Analytics.

## 2. Churn Model Performance
| Metric | Value |
|--------|-------|
| ROC AUC | 0.8500 |
| PR AUC | 0.7000 |
| F1 Score | 0.6500 |
| Precision | 0.6000 |
| Recall | 0.7100 |

## 3. Revenue Forecast Accuracy
| Metric | Value |
|--------|-------|
| MAPE | 15.50% |
| sMAPE | 14.20% |
| MAE | $5000.00 |
| RMSE | $6000.00 |

## 4. Customer Segments Summary
```text
   cluster_size  churn_rate
0           100         0.1
1           200         0.8
## 5. Top 15 Most Important Features (SHAP)
recency      5.0
frequency    3.0

## 6. Recommendations
- Churn model performance is healthy.
- Revenue forecasting error is within acceptable limits.
