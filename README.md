# AVA-AI_MODEL-2
{Timeline: Model 1 (AVA-AI_model-1) built Sept-Dec 2025. Model 2 development began around April 2026 locally and was uploaded to GitHub in September 2026.}

The goal of this project is to build a machine learning based platform that retrieve data for publicly traded companies, analyze historical price behaviours and relevant financial indicators. And generate forecast for future stockprice
The initial version will focus on major publicly traded companies such as Apple, Microsoft, Alphabet, Amazon, NVIDIA, and Tesla. The architecture will be designed so that additional stocks and markets can be supported in the future.
This project is intended as a research and educational forecasting system. Stock-market predictions are inherently uncertain, and model outputs should not be interpreted as guaranteed future prices or financial advice.

Objectives 
1) The main objectives of the project are to:
2) Retrieve current and historical stock-market data.
3) Process and clean financial time-series data.
4) Engineer useful features from historical market information.
5) Develop machine-learning models for stock-price forecasting.
6) Evaluate models using historical backtesting.
7) Visualize historical prices and model predictions.
8) Provide forecast ranges and model evaluation metrics.
9) Build a scalable architecture that can support multiple stocks.
10)Continuously improve the forecasting methodology as the project develops.

Planned Features 

(A)Market Data
Current/latest available stock price
Historical price data
Trading volume
Open, high, low, and close prices
Support for multiple stocks
Expandable support for additional markets

(B)Technical Analysis
Potential features include:
Moving averages
Exponential moving averages
Relative Strength Index (RSI)
MACD
Volatility
Momentum
Trading volume indicators

(C)Forecasting
The platform is intended to provide forecasts for multiple horizons, such as:
Short-term forecasts
7-day forecasts
30-day forecasts
Where appropriate, the system will provide an estimated prediction range rather than presenting a single predicted price as certain.

(D)Dashboard
The planned dashboard will allow users to:
Search for a stock.
View its current/latest available market price.
Explore historical price data.
View technical indicators.
Generate forecasts.
Compare predicted and actual prices.
Review model performance.

Initial Stocks
The first development phase will use a small set of highly traded companies for testing:
Company
Ticker
Apple - AAPL
Microsoft - MSFT
Alphabet - GOOGL
Amazon - AMZN
NVIDIA - NVDA
Tesla - TSLA
The system will eventually be expanded to support additional stocks dynamically.


System Architecture
The initial architecture is planned around four major components:

─────────────────────┐
                 │     Frontend        │
                 │  React Web App      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      Backend        │
                 │    FastAPI API      │
                 └──────────┬──────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
        Market Data     Feature Engine   Database
              │             │
              └─────────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   ML Forecasting    │
                 │       Models        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │     Predictions     │
                 │   + Evaluation      │
                 └─────────────────────┘


Machine Learning Methodology
The forecasting pipeline will generally follow this process:
Market Data
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Train / Validation / Test Split
     ↓
Model Training
     ↓
Historical Backtesting
     ↓
Performance Evaluation
     ↓
Forecast Generation
     ↓
Dashboard Visualization
Special care will be taken to avoid data leakage, particularly because financial time-series data is time-dependent.
Models will be evaluated on data that occurs after the training period rather than randomly mixing historical observations between training and testing datasets.