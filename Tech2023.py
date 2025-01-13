#FINDING THE OPTIMAL PORTOFLIO FOR MAJOR TECH STOCKS IN 2023
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# Constants
NUM_TRADING_DAYS = 252
NUM_PORTFOLIOS = 10000
RISK_FREE_RATE = 0.03  # Annualized risk-free rate (3%)

# Stocks and historical data range
stocks = ['AI', 'NVDA', 'TSLA', 'AMD', 'AAPL', 'SPY']
start_date = '2018-01-01'
end_date = '2023-01-01'


def download_data():
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(stock_data).dropna()


def show_data(data):
    data.plot(figsize=(10, 5))
    plt.title("Historical Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def calculate_return(data):
    log_return = np.log(data / data.shift(1))
    return log_return[1:]


def generate_portfolios(returns):
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(NUM_PORTFOLIOS):
        w = np.random.random(len(stocks))
        w /= np.sum(w)
        portfolio_weights.append(w)
        portfolio_means.append(np.sum(returns.mean() * w) * NUM_TRADING_DAYS)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * NUM_TRADING_DAYS, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)


def statistics(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * NUM_TRADING_DAYS, weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])


def min_function_sharpe(weights, returns):
    return -statistics(weights, returns)[2]


def optimize_portfolio(initial_weights, returns):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(initial_weights)))  # Adjust bounds based on weights
    return optimization.minimize(
        fun=min_function_sharpe,
        x0=initial_weights,
        args=returns,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )


def print_optimal_portfolio(optimum, returns):
    weights = optimum['x'].round(3)
    print("Optimal Portfolio Allocation:")
    for stock, weight in zip(stocks, weights):
        print(f"{stock}: {weight * 100:.2f}%")

    expected_return, volatility, sharpe_ratio = statistics(weights, returns)
    print("\nExpected Portfolio Metrics:")
    print(f"Return: {expected_return:.2f}")
    print(f"Volatility: {volatility:.2f}")
    print(f"Sharpe Ratio (Adjusted for Risk-Free Rate): {sharpe_ratio:.2f}")


def show_portfolios(returns, volatilities):
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=(returns - RISK_FREE_RATE) / volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()


def show_optimal_portfolio(opt, rets, portfolio_rets, portfolio_vols):
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c=(portfolio_rets - RISK_FREE_RATE) / portfolio_vols, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.plot(statistics(opt['x'], rets)[1], statistics(opt['x'], rets)[0], 'g*', markersize=20.0)
    plt.title("Optimal Portfolio vs Random Portfolios")
    plt.legend(["Optimal Portfolio"])
    plt.show()


def backtest_portfolio(optimal_weights, testing_data):
    testing_returns = calculate_return(testing_data)
    portfolio_daily_returns = testing_returns @ optimal_weights
    portfolio_cum_returns = (1 + portfolio_daily_returns).cumprod()

    portfolio_return = np.sum(testing_returns.mean() * optimal_weights) * NUM_TRADING_DAYS
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(testing_returns.cov() * NUM_TRADING_DAYS, optimal_weights)))
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility if portfolio_volatility > 0 else 0

    print("\nBacktest Results:")
    print(f"Testing Period Return: {portfolio_return:.3f}")
    print(f"Testing Period Volatility: {portfolio_volatility:.3f}")
    print(f"Testing Period Sharpe Ratio: {sharpe_ratio:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_cum_returns, label="Portfolio (Backtested)")
    plt.title("Cumulative Returns During Testing Period")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # Download and preprocess data
    dataset = download_data()
    show_data(dataset)

    # Split data into training and testing sets
    split_date = '2022-01-01'
    training_data = dataset[:split_date]
    testing_data = dataset[split_date:]

    # Calculate log daily returns for training data
    log_training_returns = calculate_return(training_data)

    # Optimize portfolio using training data
    initial_weights = np.ones(len(stocks)) / len(stocks)
    optimum = optimize_portfolio(initial_weights, log_training_returns)

    # Print optimal portfolio from training
    print_optimal_portfolio(optimum, log_training_returns)

    # Backtest the portfolio on testing data
    backtest_portfolio(optimum['x'], testing_data)
