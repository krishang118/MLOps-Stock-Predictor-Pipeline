import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from .utils import load_model
from .preprocess import DataPreprocessor
from sklearn.metrics import accuracy_score
TRANSACTION_COST = 0.0005  
POSITION_SIZE = 1.0      
STOP_LOSS = 0.02         
TAKE_PROFIT = 0.07       
preprocessor = DataPreprocessor()
data = pd.read_csv("data/processed/processed_stock_data.csv")
X_train, X_test, y_train, y_test, features = preprocessor.prepare_training_data(data)
model = load_model("models/xgboost_tuned_latest.pkl") if joblib.os.path.exists("models/xgboost_tuned_latest.pkl") else load_model(sorted([f for f in joblib.os.listdir("models") if f.startswith("xgboost_tuned") and f.endswith(".pkl")])[-1])
proba = model.predict_proba(X_test)[:, 1]
pred = np.full_like(proba, -1, dtype=int)  
pred[proba > 0.6] = 1 
pred[proba < 0.4] = 0  
test_idx = y_test.index
test_df = data.iloc[test_idx].copy().reset_index(drop=True)
test_df['prediction'] = pred
test_df['actual'] = y_test.values
capital = 100000  
cash = capital
positions = 0
pnl_curve = []
trade_log = []
for i, row in test_df.iterrows():
    signal = row['prediction']
    if signal not in [0, 1]:
        pnl_curve.append(cash)
        trade_log.append({
            'date': row['Date'],
            'symbol': row['Symbol'],
            'signal': signal,
            'actual': row['actual'],
            'entry': np.nan,
            'exit': np.nan,
            'trade_pnl': 0,
            'trade_cost': 0,
            'net_pnl': 0,
            'capital': cash})
        continue
    open_price = row['Open']
    close_price = row['Close']
    actual = row['actual']
    trade_pnl = 0
    trade_cost = 0
    if signal == 1:  
        entry = open_price
        stop = entry * (1 - STOP_LOSS)
        target = entry * (1 + TAKE_PROFIT)
        exit_price = close_price
        if row['Low'] <= stop:
            exit_price = stop
        elif row['High'] >= target:
            exit_price = target
        trade_pnl = (exit_price - entry) * POSITION_SIZE
        trade_cost = (entry + exit_price) * POSITION_SIZE * TRANSACTION_COST
    elif signal == 0:  
        entry = open_price
        stop = entry * (1 + STOP_LOSS)
        target = entry * (1 - TAKE_PROFIT)
        exit_price = close_price
        if row['High'] >= stop:
            exit_price = stop
        elif row['Low'] <= target:
            exit_price = target
        trade_pnl = (entry - exit_price) * POSITION_SIZE
        trade_cost = (entry + exit_price) * POSITION_SIZE * TRANSACTION_COST
    net_pnl = trade_pnl - trade_cost
    cash += net_pnl
    pnl_curve.append(cash)
    trade_log.append({
        'date': row['Date'],
        'symbol': row['Symbol'],
        'signal': signal,
        'actual': actual,
        'entry': open_price,
        'exit': exit_price,
        'trade_pnl': trade_pnl,
        'trade_cost': trade_cost,
        'net_pnl': net_pnl,
        'capital': cash})
pnl_curve = np.array(pnl_curve)
returns = np.diff(pnl_curve) / pnl_curve[:-1]
cum_return = (pnl_curve[-1] - capital) / capital
max_drawdown = np.max(np.maximum.accumulate(pnl_curve) - pnl_curve)
sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
print(f"Final capital: ${pnl_curve[-1]:.2f}")
print(f"Total return: {cum_return*100:.2f}%")
print(f"Max drawdown: ${max_drawdown:.2f}")
print(f"Sharpe ratio: {sharpe:.2f}")
trade_log_df = pd.DataFrame(trade_log)
trade_log_df.to_csv("models/trade_log.csv", index=False)
plt.figure(figsize=(10,6))
plt.plot(pnl_curve)
plt.title('Backtest P&L Curve')
plt.xlabel('Trade #')
plt.ylabel('Capital ($)')
plt.grid()
plt.tight_layout()
plt.savefig('models/pnl_curve.png')
plt.show() 