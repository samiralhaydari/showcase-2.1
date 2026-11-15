import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supplylens.data import load_exchange_rates, available_priority_currencies

rates = load_exchange_rates(force_download=True)
print("Shape:", rates.shape)
print("Date range:", rates.index.min(), "to", rates.index.max())
print("Columns:", list(rates.columns))
print("Priority currencies:", available_priority_currencies(rates))
print(rates.tail())
