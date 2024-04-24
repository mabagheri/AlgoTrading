from functools import wraps
import pandas as pd
import numpy as np
from pandas import DataFrame, Series


def inputvalidator(input_="ohlc"):
    def dfcheck(func):
        @wraps(func)
        def wrap(*args, **kwargs):

            args = list(args)
            i = 0 if isinstance(args[0], pd.DataFrame) else 1

            args[i] = args[i].rename(columns={c: c.lower() for c in args[i].columns})

            inputs = {
                "o": "open",
                "h": "high",
                "l": "low",
                "c": kwargs.get("column", "close").lower(),
                "v": "volume",
            }

            if inputs["c"] != "close":
                kwargs["column"] = inputs["c"]

            for l in input_:
                if inputs[l] not in args[i].columns:
                    raise LookupError(
                        'Must have a dataframe column named "{0}"'.format(inputs[l])
                    )

            return func(*args, **kwargs)

        return wrap

    return dfcheck


def apply(decorator):
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))

        return cls

    return decorate

# class Test:
#     print(2)

@apply(inputvalidator(input_="ohlc"))
class TA:

    __version__ = "1.3"

    @classmethod
    def Normalized_SMA(cls, ohlc: DataFrame, period: int = 20, column: str = "close") -> Series:
        "normalized SMA"
        return pd.Series(
            ohlc[column].rolling(window=period).mean() / ohlc[column],
            name=f"Normalized_SMA_{period}",
                        )

    @classmethod
    def Relative_SMA(cls, ohlc: DataFrame, period1: int = 20, period2: int = 50, column: str = "close") -> Series:
        "Relative Simple Moving Average"
        return pd.Series(
            ohlc[column].rolling(window=period1).mean() / ohlc[column].rolling(window=period2).mean(),
            name=f"RSMA_{period1}_{period2}")
   

    @classmethod
    def Normalized_EMA(cls, ohlc: DataFrame, period: int = 9, column: str = "close", adjust: bool = True, ) -> Series:
        return pd.Series(
            ohlc[column].ewm(span=period, adjust=adjust).mean()/ ohlc[column],
            name=f"Normalized_EMA_{period}",
        )

    @classmethod
    def Relative_EMA(cls, ohlc:DataFrame, period1: int=20, period2: int=50, column:str="close", adjust: bool = True) -> Series:
        return pd.Series(
            ohlc[column].ewm(span=period1, adjust=adjust).mean()/ ohlc[column].ewm(span=period2, adjust=adjust).mean(),
            name=f"REMA_{period1}_{period2}",
        )
    
    @classmethod
    def Pct_change(cls, ohlc: DataFrame, period: int = 20, column: str = "close") -> Series:
        "Percent of change in a specific period"
        return pd.Series(
            np.round(ohlc[column].pct_change(periods=period)*100, 1), name=f"Pct_change_{period}")

    
    @classmethod
    def Max_increase_pct(cls, ohlc: DataFrame, period: int = 20, column: str = "close") -> Series:
        results = [np.nan] * period
        for i in range(period, len(ohlc)):
            curr_price = ohlc[column].iloc[i]
            min_price_in_the_last_n_values = (ohlc['low'].iloc[i-period:i]).min()
            max_increase_pct = (curr_price - min_price_in_the_last_n_values)/min_price_in_the_last_n_values * 100
            results.append(max_increase_pct)

        return pd.Series(
            np.round(results, 2), name=f"Max_increase_pct_{period}")
    
        # def rolling_max_increase_in_the_last_n_values(df, n=3, col='close'):
        #     def max_inc_in_the_last_n_values(prev_and_cur_vals, n=3, y=0):
        #         curr_val = prev_and_cur_vals.iloc[-1]
        #         prev_vals = prev_and_cur_vals.iloc[-n - 1:-1]
        #         return np.round((curr_val - prev_vals.min()) / prev_vals.min() * 100, 2)

        #     result = df[col].rolling(window=n + 1, min_periods=n + 1).apply(max_inc_in_the_last_n_values, args=(n,))
        #     result.name = f'max_inc_{n}'
        #     return result

    @classmethod
    def Max_decrease_pct(cls, ohlc: DataFrame, period: int = 20, column: str = "close") -> Series:
        results = [np.nan] * period
        for i in range(period, len(ohlc)):
            curr_price = ohlc[column].iloc[i]
            max_price_in_the_last_n_values = (ohlc['high'].iloc[i-period:i]).max()
            max_decrease_pct = (curr_price - max_price_in_the_last_n_values)/max_price_in_the_last_n_values * 100
            results.append(max_decrease_pct)

        return pd.Series(
            np.round(results, 2), name=f"Max_decrease_pct_{period}")

    @classmethod
    def Std_daily_pct_change(cls, ohlc: DataFrame, period: int = 20, column: str = "close") -> Series:
        """ Standard Deviation of daily (one-period) percent change in the last month (20 days) """
        results = [np.nan] * period
        pct_change = np.round(ohlc[column].pct_change(periods=1)*100, 2)
        return pd.Series(
            pct_change.rolling(window=period).std(),
            name=f"Std_pct_change",
        )
    
    # ---------------------------------------------------------------------
    # ------------ Previously available functions -------------------------
    # ---------------------------------------------------------------------

    @classmethod
    def TRIX(
        cls,
        ohlc: DataFrame,
        period: int = 20,
        column: str = "close",
        adjust: bool = True,
    ) -> Series:
        """
        The TRIX indicator calculates the rate of change of a triple exponential moving average.
        The values oscillate around zero. Buy/sell signals are generated when the TRIX crosses above/below zero.
        A (typically) 9 period exponential moving average of the TRIX can be used as a signal line.
        A buy/sell signals are generated when the TRIX crosses above/below the signal line and is also above/below zero.

        The TRIX was developed by Jack K. Hutson, publisher of Technical Analysis of Stocks & Commodities magazine,
        and was introduced in Volume 1, Number 5 of that magazine.
        """

        data = ohlc[column]

        def _ema(data, period, adjust):
            return pd.Series(data.ewm(span=period, adjust=adjust).mean())

        m = _ema(_ema(_ema(data, period, adjust), period, adjust), period, adjust)

        return pd.Series(100 * (m.diff() / m), name="{0} period TRIX".format(period))

    @classmethod
    def ER(cls, ohlc: DataFrame, period: int = 10, column: str = "close") -> Series:
        """The Kaufman Efficiency indicator is an oscillator indicator that oscillates between +100 and -100, where zero is the center point.
         +100 is upward forex trending market and -100 is downwards trending markets."""

        change = ohlc[column].diff(period).abs()
        volatility = ohlc[column].diff().abs().rolling(window=period).sum()

        return pd.Series(change / volatility, name="{0} period ER".format(period))




    @classmethod
    def PPO(
        cls,
        ohlc: DataFrame,
        period_fast: int = 12,
        period_slow: int = 26,
        signal: int = 9,
        column: str = "close",
        adjust: bool = True,
    ) -> DataFrame:
        """
        Percentage Price Oscillator
        PPO, PPO Signal and PPO difference.
        As with MACD, the PPO reflects the convergence and divergence of two moving averages.
        While MACD measures the absolute difference between two moving averages, PPO makes this a relative value by dividing the difference by the slower moving average
        """

        EMA_fast = pd.Series(
            ohlc[column].ewm(ignore_na=False, span=period_fast, adjust=adjust).mean(),
            name="EMA_fast",
        )
        EMA_slow = pd.Series(
            ohlc[column].ewm(ignore_na=False, span=period_slow, adjust=adjust).mean(),
            name="EMA_slow",
        )
        PPO = pd.Series(((EMA_fast - EMA_slow) / EMA_slow) * 100, name=f"PPO_{period_fast}_{period_slow}")
        PPO_signal = pd.Series(
            PPO.ewm(ignore_na=False, span=signal, adjust=adjust).mean(), name=f"SIGNAL_{period_fast}_{period_slow}_{signal}"
        )
        PPO_histo = pd.Series(PPO - PPO_signal, name=f"HISTO_{period_fast}_{period_slow}_{signal}")

        return pd.concat([PPO, PPO_signal, PPO_histo], axis=1)


    @classmethod
    def ROC(cls, ohlc: DataFrame, period: int = 12, column: str = "close") -> Series:
        """The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum,
        is a pure momentum oscillator that measures the percent change in price from one period to the next.
        The ROC calculation compares the current price with the price “n” periods ago."""

        return pd.Series(
            (ohlc[column].diff(period) / ohlc[column].shift(period)) * 100, name=f"ROC_{period}_{column[:3]}"
        )


    @classmethod
    def RSI(
        cls,
        ohlc: DataFrame,
        period: int = 14,
        column: str = "close",
        adjust: bool = True,
    ) -> Series:
        """Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
        RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
        Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
        RSI can also be used to identify the general trend."""

        ## get the price diff
        delta = ohlc[column].diff()

        ## positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # EMAs of ups and downs
        _gain = up.ewm(alpha=1.0 / period, adjust=adjust).mean()
        _loss = down.abs().ewm(alpha=1.0 / period, adjust=adjust).mean()

        RS = _gain / _loss
        return pd.Series(100 - (100 / (1 + RS)), name="{0} period RSI".format(period))

    @classmethod
    def IFT_RSI(
        cls,
        ohlc: DataFrame,
        rsi_period: int = 5,
        wma_period: int = 9,
        column: str = "close",
    ) -> Series:
        """Modified Inverse Fisher Transform applied on RSI.
        Suggested method to use any IFT indicator is to buy when the indicator crosses over –0.5 or crosses over +0.5
        if it has not previously crossed over –0.5 and to sell short when the indicators crosses under +0.5 or crosses under –0.5
        if it has not previously crossed under +0.5."""

        # v1 = .1 * (rsi - 50)
        v1 = pd.Series(0.1 * (cls.RSI(ohlc, rsi_period) - 50), name="v1")

        # v2 = WMA(wma_period) of v1
        d = (wma_period * (wma_period + 1)) / 2  # denominator
        weights = np.arange(1, wma_period + 1)

        def linear(w):
            def _compute(x):
                return (w * x).sum() / d

            return _compute

        _wma = v1.rolling(wma_period, min_periods=wma_period)
        v2 = _wma.apply(linear(weights), raw=True)

        ift = pd.Series(((v2 ** 2 - 1) / (v2 ** 2 + 1)), name=f"IFT_RSI_{rsi_period}")
        return ift

