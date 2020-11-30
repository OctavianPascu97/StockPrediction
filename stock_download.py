
import yfinance as yf

root_path = r"C:\Users\octav\PycharmProjects\Stockpred"

if __name__ == "__main__":
    tick = ['SPY','AAPL','BTC-USD','NVDA','RIOT','BA','TSLA','OSTK','IBM','MSFT','CCEP','LNVGY']
    for ticker in tick:
        data = yf.download(ticker, start="2015-01-01", end="2020-01-30")
        data.to_csv(f"{root_path}/stock_data/{ticker}")
