from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import pytz
import os

app = FastAPI()

import os

@app.on_event("startup")
async def startup():
    print("Current directory:", os.getcwd())
    print("Files in directory:", os.listdir())
    if not os.path.exists("index.html"):
        print("ERROR: index.html not found!")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Thêm route cho trang chủ
from fastapi.responses import HTMLResponse

# @app.get("/", response_class=HTMLResponse)
# async def read_index():
#     with open("index.html", "r") as f:
#         return f.read()


@app.get("/", response_class=HTMLResponse)
async def read_index():
    try:
        # Mở file với encoding UTF-8
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model = load_model("model_stock.keras")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Get list of trained tickers from the encoder
trained_tickers = label_encoder.classes_.tolist()

# ---------- Utility Functions ----------
def get_recent_data(ticker, days=7, interval="1m"):
    """Get recent market data from yfinance"""
    stock = yf.Ticker(ticker)
    df = stock.history(period=f"{days}d", interval=interval)
    df = df.reset_index()
    df = df.rename(columns={"Close": "close", "Datetime": "datetime"})
    df["ticker"] = ticker
    return df

def is_market_open():
    """Check if current time is within market hours (9:30-16:00 ET)"""
    ny_time = datetime.now(pytz.timezone('America/New_York'))
    if ny_time.weekday() >= 5:  # Weekend
        return False
    market_open = ny_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = ny_time.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= ny_time <= market_close

def preprocess_data(df, ticker, window_size=60):
    """Prepare data for model prediction"""
    if ticker not in label_encoder.classes_:
        raise ValueError(f"Ticker {ticker} not in trained tickers")
    
    # Get last window_size+1 data points
    recent_data = df["close"].values[-window_size-1:]
    
    if len(recent_data) < window_size+1:
        raise ValueError(f"Need at least {window_size+1} data points, got {len(recent_data)}")
    
    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(recent_data[:-1].reshape(-1, 1))
    normalized = scaler.transform(recent_data[-window_size:].reshape(-1, 1))
    
    # Reshape for CNN input
    X = normalized.reshape(1, window_size, 1)
    return X, scaler

# ---------- API Endpoints ----------
@app.get("/tickers")
async def get_supported_tickers():
    """Get list of supported stock tickers"""
    return {"tickers": trained_tickers}

@app.get("/history/{ticker}")
async def get_history(ticker: str, days: int = 7):
    """Get historical data for a ticker"""
    ticker = ticker.upper()
    if ticker not in trained_tickers:
        return {"error": f"Ticker {ticker} is not supported"}
    
    try:
        df = get_recent_data(ticker, days)
        df = df.dropna(subset=["close"])
        history = df[["datetime", "close"]].to_dict('records')
        return {"history": history}
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict/{ticker}")
async def predict(ticker: str):
    """Predict next minute price for given ticker"""
    ticker = ticker.upper()
    if ticker not in trained_tickers:
        return {"error": f"Ticker {ticker} is not supported"}
    
    try:
        # Get recent data
        df = get_recent_data(ticker)
        df = df.dropna(subset=["close"])
        
        # Check if we have enough data
        if len(df) < 61:
            return {"error": "Not enough historical data available"}
        
        # Preprocess data
        X, scaler = preprocess_data(df, ticker)
        
        # Make prediction
        pred = model.predict(X)
        pred_price = scaler.inverse_transform(pred)[0][0]
        
        # Get actual last price
        actual_price = df["close"].iloc[-1]
        
        return {
            "ticker": ticker,
            "predicted_price": round(float(pred_price), 2),
            "actual_price": round(float(actual_price), 2),
            "timestamp": datetime.now().isoformat(),
            "market_open": is_market_open()
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/predict_multi/{ticker}")
async def predict_multiple_steps(ticker: str, steps: int = 3):
    """Predict multiple future steps (minutes)"""
    ticker = ticker.upper()
    if ticker not in trained_tickers:
        return {"error": f"Ticker {ticker} is not supported"}
    if steps < 1 or steps > 5:
        return {"error": "Steps must be between 1 and 5"}
    
    try:
        df = get_recent_data(ticker)
        df = df.dropna(subset=["close"])
        
        if len(df) < 61:
            return {"error": "Not enough historical data available"}
        
        predictions = []
        current_data = df.copy()
        
        for _ in range(steps):
            X, scaler = preprocess_data(current_data, ticker)
            pred = model.predict(X)
            pred_price = scaler.inverse_transform(pred)[0][0]
            predictions.append(round(float(pred_price), 2))
            
            # Append prediction to dataframe for next step
            new_row = pd.DataFrame({
                "datetime": [current_data["datetime"].iloc[-1] + timedelta(minutes=1)],
                "close": [pred_price],
                "ticker": [ticker]
            })
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return {
            "ticker": ticker,
            "predictions": predictions,
            "actual_price": round(float(df["close"].iloc[-1]), 2),
            "steps": steps
        }
    except Exception as e:
        return {"error": str(e)}