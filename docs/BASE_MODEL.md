\# AVA-AI Base Model v1



\## 1. Overview



AVA-AI Base Model v1 is the first end-to-end machine-learning forecasting system developed for the AVA-AI project.



The purpose of this version is to establish a reproducible baseline for next-day stock-return prediction using historical market data and technical indicators.



The system currently supports:



\- Apple (AAPL)

\- Amazon (AMZN)

\- Alphabet (GOOGL)

\- Microsoft (MSFT)

\- NVIDIA (NVDA)

\- Tesla (TSLA)



The base model is intended as a research and engineering baseline, not as a validated trading strategy or investment recommendation.



\---



\## 2. Dataset



The base model uses daily historical stock-market data for six equities.



Each stock contains:



\- 500 daily observations

\- Date range: 2024-09-27 to 2026-09-25

\- Open price

\- High price

\- Low price

\- Close price

\- Volume



Data quality checks confirmed:



\- Missing values: 0

\- Duplicate dates: 0

\- Rows per stock: 500



\---



\## 3. Feature Engineering



The model uses the following technical features:



\- Close price

\- 20-day moving average (MA20)

\- 50-day moving average (MA50)

\- 20-day return volatility

\- Volume percentage change

\- 1-day lagged return

\- 5-day lagged return

\- 14-period RSI

\- MACD

\- MACD signal



The prediction target is the next trading day's percentage return.



The target is calculated as:



`next\_day\_close / current\_close - 1`



\---



\## 4. Validation Method



The model was evaluated using walk-forward validation rather than a random train/test split.



The validation structure used sequential historical periods:



| Fold | Training End | Test Start | Test End |

|---|---:|---:|---:|

| 1 | 150 | 150 | 210 |

| 2 | 210 | 210 | 270 |

| 3 | 270 | 270 | 330 |

| 4 | 330 | 330 | 390 |

| 5 | 390 | 390 | 450 |



This approach preserves the chronological nature of financial data and avoids randomly mixing future observations into the training data.



\---



\## 5. Naive Baseline



A naive baseline was established before evaluating machine-learning models.



The baseline predicts the next return using the current return.



Overall baseline performance:



\- MAE: 0.01558

\- RMSE: 0.02105



The naive baseline is important because a machine-learning model should demonstrate improvement over a simple reference method before being considered useful.



\---



\## 6. Machine-Learning Experiments



Several models were evaluated using the same walk-forward framework.



\### Random Forest



\- MAE: 0.01706

\- RMSE: 0.02259

\- Directional Accuracy: 49.67%



\### Gradient Boosting



\- MAE: 0.01996

\- RMSE: 0.02611

\- Directional Accuracy: 50.94%



\### Linear Regression



\- MAE: 0.01790

\- RMSE: 0.02348

\- Directional Accuracy: 49.50%



\### Ridge Regression



\- MAE: 0.01753

\- RMSE: 0.02311

\- Directional Accuracy: 49.00%



\---



\## 7. Expanded Feature Experiments



Additional lagged returns, rolling returns, and price-to-moving-average features were also tested.



These experiments did not produce a meaningful improvement over the naive baseline.



Random Forest with expanded features:



\- MAE: 0.01693

\- RMSE: 0.02251

\- Directional Accuracy: 48.44%



Gradient Boosting with expanded features:



\- MAE: 0.01940

\- RMSE: 0.02538

\- Directional Accuracy: 48.72%



Scaled Ridge with expanded features:



\- MAE: 0.01770

\- RMSE: 0.02325

\- Directional Accuracy: 48.17%



\---



\## 8. Classification Experiments



Next-day direction classification was also tested.



The experiments included:



\- Random Forest classification

\- Logistic Regression

\- Majority-class baseline

\- Multiple-stock pooled classification

\- Confidence-threshold experiments



The results remained close to chance-level performance.



Logistic Regression:



\- Accuracy: 49.39%

\- Balanced Accuracy: 50.79%



Random Forest next-day direction:



\- Accuracy: 49.89%



Majority baseline:



\- Accuracy: 50.33%



These results did not provide evidence of a meaningful predictive signal in the current technical feature set.



\---



\## 9. Final Base Model



The production inference pipeline currently uses a Random Forest model for each supported stock.



Configuration:



\- Random Forest

\- 300 trees

\- Technical feature set described above

\- Final model fitted using the available historical observations

\- Saved using Joblib



Model files are stored locally under:



`models/\*.joblib`



The generated model files are intentionally excluded from Git version control because of their size.



\---



\## 10. Model Performance Finding



The most important finding from Base Model v1 is that the machine-learning models did not consistently outperform the naive baseline.



The naive baseline achieved:



\- MAE: 0.01558

\- RMSE: 0.02105



The original Random Forest achieved:



\- MAE: 0.01706

\- RMSE: 0.02259



Therefore, Base Model v1 should be treated as a technical and research baseline rather than evidence of a profitable forecasting system.



This result is retained intentionally rather than selecting a model solely because it produces attractive predictions.



\---



\## 11. API Integration



The trained models are integrated into a FastAPI backend.



The API supports:



`GET /predict/{ticker}`



Supported tickers:



\- AAPL

\- AMZN

\- GOOGL

\- MSFT

\- NVDA

\- TSLA



The API returns a structured response containing:



\- ticker

\- predicted next-day return



Example:



```json

{

&#x20; "ticker": "AAPL",

&#x20; "predicted\_next\_day\_return": -0.001156518179056177

}

