import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
from numpy import linalg
import streamlit as st


yf.pdr_override() # <== that's all it takes :-)
start_date = st.text_input('Input start date here [YYYY-MM-DD]')#"2011-01-01"

end_date = st.text_input('Input end date here [YYYY-MM-DD]') #"2021-01-01"
frequency = st.selectbox('Select frequency',    ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')) #'1mo'
tickers = st.multiselect('Select tickers', ('^FTSE', '^GSPC', '^DJI', '^IXIC', '^GDAXI', '^FCHI', '^N225', '^HSI', '000001.SS', '399001.SZ', '^AXJO', '^GSPTSE', '^JN0U.JO', '^RUT', '^VIX', '^STOXX50E', '^N100', '^BFX', 'IMOEX.ME', '^NYA', '^XAX', '^STI', '^AORD', '^BSESN', '^JKSE', '^KLSE', '^NZ50', '^KS11', '^TWII', '^BVSP', '^MXX', '^IPSA', '^MERV', '^TA125.TA', '^CASE30', '^NSEI', 'FTSEMIB.MI'))
#['^FTSE', '^VIX', '^N100', 'IMOEX.ME', '^GDAXI']


def pull_yf_data_multiple_tickers(tickers, start_date, end_date, frequency):
    data2 = pd.DataFrame({})
    for ticker in tickers:
        data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date, interval=frequency)
        #data.rename(columns={'Adj Close': ticker}, inplace=True)
        data = pd.DataFrame({ticker: data['Adj Close']})
        data2 = pd.concat([data2, data], axis=1)
    return data2

def plot_prices(df):
    cols = df.columns
    plt.figure(figsize=(15,12))
    for col in cols:
        plt.plot(df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('Adjusted Close Price')
    plt.title('Adjusted Close Prices')
    plt.legend()
    plt.grid(True)
    return plt.show()

def plot_returns(df):
    cols = df.columns
    plt.figure(figsize=(15,12))
    for col in cols:
        plt.plot(df[col], label=col)
    plt.xlabel('Time')
    plt.ylabel('Returns')
    plt.title('Returns')
    plt.legend()
    plt.grid(True)
    return plt.show()
    
def calc_mvp(df):
    returns = df.pct_change()
    returns.drop(returns.head(1).index, inplace=True)
    
    covmat = returns.cov()
    inverse = pd.DataFrame(np.linalg.inv(covmat), columns = covmat.columns, index = covmat.index)
    
    unit = []
    for x in range(0, len(covmat)):
        unit.append(1)
    unit_transp = pd.DataFrame({'test': unit})

    c = np.dot(np.dot(unit, inverse), unit_transp)
    inverse_unit = np.dot(inverse,unit_transp)

    xmin = pd.DataFrame(((1/c)* inverse_unit)*100)
    xmin.columns = ['Optimal portfolio weights']
    xmin.index = covmat.index

    return xmin

def returns(df):
    returns = df.pct_change()
    returns.drop(returns.head(1).index, inplace=True)
    return returns

st.table(data=calc_mvp(pull_yf_data_multiple_tickers(tickers, start_date, end_date, frequency)))
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig=plot_prices(pull_yf_data_multiple_tickers(tickers, start_date, end_date, frequency)))
st.pyplot(fig=plot_returns(returns(pull_yf_data_multiple_tickers(tickers, start_date, end_date, frequency))))


