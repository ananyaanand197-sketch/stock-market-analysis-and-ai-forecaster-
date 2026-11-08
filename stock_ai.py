import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_stock_ai():
    print("--- Financial AI Stock Forecaster ---")
    
    # 1. Generate realistic simulated stock data (100 trading days)
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=100)
    prices = 150 + np.cumsum(np.random.randn(100) * 2)
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    
    # 2. Feature Engineering (Financial Indicators)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['Price_Change'] = df['Close'].diff()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int) # 1 if price goes up tomorrow, else 0
    
    df = df.dropna()
    
    # 3. Train ML Model
    X = df[['Close', 'SMA_10', 'Price_Change']]
    y = df['Target']
    
    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"Model Training Complete.")
    print(f"Prediction Accuracy on Test Data: {acc * 100:.1f}%")
    print("\nRecent Price Data & Signal Preview:")
    print(df[['Date', 'Close', 'SMA_10', 'Target']].tail(5).to_string(index=False))

if __name__ == "__main__":
    run_stock_ai()