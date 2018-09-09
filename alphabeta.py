import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()

from sys import argv as args
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta

_AV_KEY_ = 'YOUR_KEY_HERE'

# To prove this works, check the beta roughly matches yahoo finance.
# It won't be exact due to methodology.
def calculate(benchmark, stock, event_date, beta_span=3):
    df_idx = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + benchmark + '&apikey=' + _AV_KEY_ + '&datatype=csv', index_col='timestamp')
    df_sym = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=' + stock + '&apikey=' + _AV_KEY_ + '&datatype=csv', index_col='timestamp')

    df_idx.drop(columns=['open', 'high', 'low', 'adjusted close', 'volume', 'dividend amount'], inplace=True)
    df_sym.drop(columns=['open', 'high', 'low', 'adjusted close', 'volume', 'dividend amount'], inplace=True)

    new = pd.merge(df_idx, df_sym, on='timestamp', how='outer')

    # flip data from AV
    new = new.iloc[::-1]

    # Remove n years from date given to given us start_date
    start_date = event_date - relativedelta(months=12*beta_span) # *2 as Saturday and Sunday are omitted

    # Return only rows with dates between our values fore regression
    new = new[(new.index >= str(start_date)) & (new.index <= str(event_date))]

    # Create percent change
    new = new.pct_change(1)

    # Drop NaN
    new = new.dropna(axis=0)  # drop first missing row

    # split dependent and independent variable
    benchmark = new['close_x'].values.tolist()
    study     = new['close_y'].values.tolist()

    # Add a constant to the independent value
    X1 = sm.add_constant(benchmark)

    # make regression model
    model = sm.OLS(study, X1)

    # fit model and print results
    results = model.fit()

    intercept = results.params[0]
    slope     = results.params[1]

    return [intercept, slope]

if __name__ == '__main__':
        benchmark  = args[1]
        stock      = args[2]
        event_date = args[3]
        beta_span  = args[4]
        main(benchmark, stock, event_date, beta_span)
