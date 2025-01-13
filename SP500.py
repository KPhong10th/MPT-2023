#PREDICT THE BEST STOCKS TO BUY IN 2023 FROM SPY500
import yfinance as yf

# Define function to fetch historical data
def fetch_sp500_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            stock_data[ticker] = hist
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return stock_data

# Scrape S&P 500 tickers
import pandas as pd
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_table = pd.read_html(url)[0]
tickers = sp500_table['Symbol'].tolist()
tickers = [ticker for ticker in tickers if "." not in ticker and "-" not in ticker]

# Fetch data for S&P 500 stocks
def fetch_sp500_data(tickers, start_date, end_date):
    stock_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:  # Check if data is empty
                print(f"Skipping {ticker}: No data found.")
                continue
            stock_data[ticker] = hist
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return stock_data

# Fetch data for S&P 500 stocks
START_DATE = "2021-01-01"
END_DATE = "2022-12-31"
sp500_data = fetch_sp500_data(tickers, START_DATE, END_DATE)

def preprocess_sp500_data(sp500_data):
    all_data = []
    for ticker, data in sp500_data.items():
        data = data.copy()  # Explicitly create a copy
        data['1M_Return'] = data['Close'].pct_change(periods=21)
        data['3M_Return'] = data['Close'].pct_change(periods=63)
        data['Volatility'] = data['Close'].rolling(window=21).std()
        data['1Y_Return'] = (data['Close'].shift(-252) - data['Close']) / data['Close']
        data['Target'] = data['1Y_Return'].apply(lambda x: 1 if x > 0.2 else 0)
        data = data.dropna()
        data['Ticker'] = ticker
        all_data.append(data)
    return pd.concat(all_data)

# Preprocess the data
processed_data = preprocess_sp500_data(sp500_data)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Features and labels
features = processed_data[['1M_Return', '3M_Return', 'Volatility']]
labels = processed_data['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict and evaluate
predictions = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Predict probabilities
processed_data['Prediction_Probability'] = classifier.predict_proba(features)[:, 1]

# Identify top stocks
top_stocks = processed_data.sort_values(by='Prediction_Probability', ascending=False).head(10)
print("Top Stocks for 2023:")
print(top_stocks[['Ticker', 'Prediction_Probability', '1Y_Return']])



