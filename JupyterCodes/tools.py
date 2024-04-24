import os
import pandas as pd
import time
import requests
import yfinance as yf
from yahoofinancials import YahooFinancials


def get_sp500_list():
    table=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    return df

def download_stock_price_data(symbol, start, end, option='yahoo_finance', interval='1d', verbose=False):
    try:
        if option == 'yahoo_finance':
            # https://github.com/ranaroussi/yfinance
            # 
            # print(f"Download {symbol} data with Yahoo Finance API")
            # Option 1
            # Interval: 1m, 5m, 15m, 30m, 60m, 1h, 1d, 1wk,1mo,
            # auto_adjust = True: so all the presented prices areadjusted for potential corporate actions, such as splits.
            data = yf.download(symbol, start=start, end=end, progress=False, interval=interval)

        elif option == "YahooFinancials":
            # Option 3
            yahoo_financials = YahooFinancials(symbol)
            data = yahoo_financials.get_historical_price_data(start_date=start, end_date=end, 
                                                              time_interval='daily')
            data = pd.DataFrame(data[symbol]['prices'])        


        elif option == "alpha_vintage":
            ALPHA_VANTAGE_API_KEY = 'I3Z5IQNTGCQ7IED4'  # J4RADOTW5DBN344U
            print(f"{i+1}/{len(symbols)}: Download {symbol} data with Alpha Vintage using Pandas data_reader API")
            data = web.DataReader(symbol, "av-daily-adjusted", # av-daily-adjusted / av-daily
                                  start=start, end=end,
                                  api_key=ALPHA_VANTAGE_API_KEY)

        return data
    
    except Exception as e:
        print(f"Got an error: {e}")


def download_stock_price_data_wrapper(symbols, 
                                      start_date, prev_end_date, new_end_date, fake_end_date,
                                      option='yahoo_finance', interval='1d',
                                      save_folder='../Data/Daily/US_Market/latest'):

    interval_dict = {'1d': 'daily', '1h':'hourly', '60min':'hourly', '60m':'hourly', '1mo':'Monthly'}
    incomplete_downloads = dict()
    for i, symbol in enumerate(symbols[:]):
        print(f"{i+1} : {symbol} \t", end='')

        time_period = f"{start_date.replace('-', '')}_to_{prev_end_date.replace('-', '')}"
        save_file_name_old = f'{save_folder}/{symbol}_USD_{time_period}_{interval_dict[interval]}_{option}.csv'

        time_period = f"{start_date.replace('-', '')}_to_{new_end_date.replace('-', '')}"
        save_file_name_new = f'{save_folder}/{symbol}_USD_{time_period}_{interval_dict[interval]}_{option}.csv'

        if os.path.exists(save_file_name_new):
            data = pd.read_csv(save_file_name_new).set_index("Date")
            last_row_date = data.iloc[-1, :].name
            if last_row_date == new_end_date:
                print('{:>2d}: {:<10s} Has already been downloaded until {:<12s}'.format(i+1, symbol, last_row_date))
                continue
            else:
                print('{:>2d}: {:<10s} Download from {:<12s} to {:^12s}'.format(i+1, symbol, last_row_date, new_end_date))
                new_data = download_stock_price_data(symbol, start=last_row_date, end=fake_end_date, interval=interval)
                new_data.index = new_data.index.astype(str)
                merged_data = pd.concat((data, new_data))
                merged_data = merged_data[~merged_data.index.duplicated(keep='first')]
                os.remove(save_file_name_new)
                merged_data.to_csv(save_file_name_new)
                if merged_data.index[-1] != new_end_date:
                    incomplete_downloads[symbol] = merged_data.index[-1]

        elif os.path.exists(save_file_name_old):
            data = pd.read_csv(save_file_name_old).set_index("Date")
            last_row_date = data.iloc[-1, :].name
            print('{:>2d}: {:<10s} Download from {:<12s} to {:^12s}'.format(i+1, symbol, last_row_date, new_end_date))
            new_data = download_stock_price_data(symbol, start=last_row_date, end=fake_end_date, interval=interval)
            new_data.index = new_data.index.astype(str)
            merged_data = pd.concat((data, new_data))
            merged_data = merged_data[~merged_data.index.duplicated(keep='first')]
            os.remove(save_file_name_old)
            merged_data.to_csv(save_file_name_new)
            if merged_data.index[-1] != new_end_date:
                incomplete_downloads[symbol] = merged_data.index[-1]

        else:
            print('{:>2d}: {:<10s} Download from scratch'.format(i+1, symbol))
            data = download_stock_price_data(symbol, start=start_date, end=fake_end_date, interval=interval)
            data.to_csv(save_file_name_new)
            if data.index[-1] != new_end_date:
                incomplete_downloads[symbol] = data.index[-1]

    print("Done")