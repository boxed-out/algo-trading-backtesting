# The code is from Jing Wu and Shile Wen, posted at the following link
# https://www.quantconnect.com/forum/discussion/3260/etf-trend-following-algorithm/p1
# The strategy is to use Trend Following to make trades for ETFs


from math import ceil,floor
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from System.Drawing import Color


class TrendFollowingAlgorithm(QCAlgorithm):
    

    def Initialize(self):
        self.SetStartDate(2018, 10, 1)  
        self.SetEndDate(2018, 12, 24)
        self.SetCash(100000)            
        self.lookback = int(252/2)
        self.profittake = 1.96 # 95% bollinger band
        self.maxlever = 0.9 # always hold 10% Cash
        self.AddEquity("SPY", Resolution.Daily)
        etfs = [
            # Equity
            'DIA',    # Dow
            'SPY',    # S&P 500
            # Fixed income
            'IEF',    # Treasury Bond
            'HYG',    # High yield bond
            # Alternatives
            'USO',    # Oil
            'GLD',    # Gold
            'VNQ',    # US Real Estate
            'RWX',    # Dow Jones Global ex-U.S. Select Real Estate Securities Index
            'UNG',    # Natual gas
            'DBA',    # Agriculture
        ]
        
        for etf in etfs:
            self.AddEquity(etf, Resolution.Daily)
        
        self.symbol_data = {} # stores the Rolling open and close data, as well as weights and stop prices
        
        self.PctDailyVolatilityTarget = 0.025 # target daily vol target in %
        
        # setting benchmark for graph
        self.SetBenchmark("SPY")
        # Variable to hold the last calculated SPY value
        self.lastSPYValue = None
        # Inital benchmark value scaled to be the same as portfolio starting cash
        self.SPYPerformance = self.Portfolio.TotalPortfolioValue
        # Performance plot
        PerformancePlot = Chart("PerformancePlot")
        PerformancePlot.AddSeries(Series("Linear Regression Strategy", SeriesType.Line,"", Color.Red))
        PerformancePlot.AddSeries(Series("SPY", SeriesType.Line,"", Color.Blue))
        PerformancePlot.AddSeries(Series("Original", SeriesType.Line, "", Color.Black))
        self.AddChart(PerformancePlot)

        # trailing stop
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.AfterMarketOpen("SPY", 10), self.trail_stop)
        
        # perform calculations for asset weightings
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.AfterMarketOpen("SPY", 28), self.compute_regression_asset_weightings)
        
        # rebalance the portfolio according to calculations
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.AfterMarketOpen("SPY", 30), self.rebalance)

        self.curr_day = -1
        
        # update closing price data
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.BeforeMarketClose("SPY", 1), self.update_closes)

        # plot graph
        self.Schedule.On(self.DateRules.EveryDay("SPY"), self.TimeRules.BeforeMarketClose("SPY", 1), self.plotting_performance)
    
    def update_closes(self):
        # updates closing price data
        
        for symbol, sd in self.symbol_data.items():
            if self.CurrentSlice.Bars.ContainsKey(symbol):
                sd.UpdateClose(self.CurrentSlice.Bars[symbol].Close)
    
    def OnData(self, data):
        # updates the SymbolDatas with daily opening prices
        
        if self.curr_day == self.Time.day:
            return
        
        self.curr_day = self.Time.day
        
        for symbol, sd in self.symbol_data.items():
            if data.Bars.ContainsKey(symbol):
                sd.UpdateOpen(data.Bars[symbol].Open)
    
    def OnSecuritiesChanged(self, changed):
        # add and remove SymbolDatas to our dict
        for security in changed.AddedSecurities:
            self.symbol_data[security.Symbol] = SymbolData(self, security.Symbol, self.lookback)
        
        for security in changed.RemovedSecurities:
            self.symbol_data.pop(security.Symbol, None)
    
    def calc_vol_scalar(self):
        # calculate the volatility scale factors for each ticker
        
        processed_sd = {symbol.Value: list(sd.open_data)[::-1] for symbol, sd in self.symbol_data.items()}
        
        df_price = pd.DataFrame(processed_sd) 
        rets = np.log(df_price).diff().dropna()
        lock_value = df_price.iloc[-1]
        
        # Exponentially-weighted moving std
        price_vol = rets.ewm(halflife=20,ignore_na=True, min_periods=0, adjust=True).std(bias=False).dropna() 
        
        volatility_scalar = self.PctDailyVolatilityTarget / price_vol.iloc[-1]

        return volatility_scalar
  
    def compute_regression_asset_weightings(self):
        # compute asset weightings
        
        A = range( self.lookback + 1 )
        for symbol, sd in self.symbol_data.items():
            if not sd.IsReady:
                continue
            
            prices = list(sd.open_data)[::-1]  # undo reverse order of RWs
            
            # volatility
            std = np.std(prices)
            # Price points to run regression
            Y = prices
            # Add column of ones so we get intercept
            X = np.column_stack([np.ones(len(A)), A])
            if len(X) != len(Y):
                length = min(len(X), len(Y))
                X = X[-length:]
                Y = Y[-length:]
                A = A[-length:]
            # Creating Model
            reg = LinearRegression()
            # Fitting training data
            
            reg = reg.fit(X, Y)
            # run linear regression y = ax + b
            b = reg.intercept_
            a = reg.coef_[1]
            
            # Normalized slope
            slope = a / b *252.0
            # Currently how far away from regression line
            delta = Y - (np.dot(a, A) + b)
            # Don't trade if the slope is near flat (at least %7 growth per year to trade)
            slope_min = 0.252
            
            # Long but slope turns down, then exit
            if sd.weight > 0 and slope < 0:
                sd.weight = 0
                
            # short but slope turns upward, then exit
            if sd.weight < 0 and slope > 0:
                sd.weight = 0
                
            # Trend is up
            if slope > slope_min:
                
                # price crosses the regression line
                if delta[-1] > 0 and delta[-2] < 0 and sd.weight == 0:
                    sd.stopprice = None
                    sd.weight = slope
                # Profit take, reaches the top of 95% bollinger band
                if delta[-1] > self.profittake * std and sd.weight > 0:
                    sd.weight = 0
            
            # Trend is down
            if slope < -slope_min:
          
                # price crosses the regression line
                if delta[-1] < 0 and delta[-2] > 0 and sd.weight == 0:
                    sd.stopprice = None
                    sd.weight = slope
                # profit take, reaches the top of 95% bollinger band
                if delta[-1] < self.profittake * std and sd.weight < 0:
                    sd.weight = 0
                    
    
    def rebalance(self):
        # rebalance portfolio
        
        vol_mult = self.calc_vol_scalar()
        no_positions = len([1 for _, sd in self.symbol_data.items() if sd.weight != 0])

        for symbol, sd in self.symbol_data.items():
            if not sd.IsReady:
                continue
            if sd.weight == 0:
                self.Liquidate(symbol)
            elif sd.weight > 0:
                self.SetHoldings(symbol, (min(sd.weight, self.maxlever)/no_positions)*vol_mult[symbol.Value])
            elif sd.weight < 0:
                self.SetHoldings(symbol, (max(sd.weight, -self.maxlever)/no_positions)*vol_mult[symbol.Value])

    def trail_stop(self):
        for symbol, sd in self.symbol_data.items():
            if not sd.IsReady:
                continue
            mean_price = np.mean(list(sd.close_data))
            # Stop loss percentage is the return over the lookback period
            stoploss = abs(sd.weight * self.lookback / 252.0) + 1    # percent change per period
            if sd.weight > 0 and sd.stopprice is not None:
                if sd.stopprice is not None and sd.stopprice < 0:
                    sd.stopprice = mean_price / stoploss
                else:
                    sd.stopprice = max(mean_price / stoploss, sd.stopprice)
                    if mean_price < sd.stopprice:
                        sd.weight = 0
                        self.Liquidate(symbol)
            
            elif sd.weight < 0 and sd.stopprice is not None: 
                if sd.stopprice is not None and sd.stopprice < 0:
                    sd.stopprice = mean_price * stoploss
                else:
                    sd.stopprice = min(mean_price * stoploss, sd.stopprice)
                    if mean_price > sd.stopprice:
                       sd.weight = 0
                       self.Liquidate(symbol)
            
            else:
                sd.stopprice = None
                
    def plotting_performance(self):
        
        # benchmark = 0
        
        # for symbol in self.symbols:
        #     benchmark += self.Securities[symbol].Close
            
        # benchmark = benchmark / len(self.symbols)
        
        # Plot performance graph    
        benchmark = self.Securities["SPY"].Close
        
        # Update SPY performance if it's not the first periof performance
        if self.lastSPYValue is not  None and self.lastSPYValue != 0:
           self.SPYPerformance = self.SPYPerformance * (benchmark/self.lastSPYValue)
           
        # store today's benchmark close price for use tomorrow
        self.lastSPYValue = benchmark
        
        # make our plots
        self.Plot("PerformancePlot", "Linear Regression Strategy", self.Portfolio.TotalPortfolioValue)
        self.Plot("PerformancePlot", "SPY", self.SPYPerformance)
        self.Plot("PerformancePlot", "Original", 100000)
                
class SymbolData:
    def __init__(self, algorithm, symbol, lookback):
        self.open_data = RollingWindow[float](lookback)
        self.close_data = RollingWindow[float](3)
        
        hist = algorithm.History(symbol, lookback, Resolution.Daily).loc[symbol]
        for _, row in hist.iterrows():
            self.open_data.Add(row.open)
            self.close_data.Add(row.close)
            
        self.stopprice = 0
        self.weight = 0
    
    def UpdateOpen(self, value):
        self.open_data.Add(value)
    
    def UpdateClose(self, value):
        self.close_data.Add(value)
    
    @property
    def IsReady(self):
        return self.open_data.IsReady and self.close_data.IsReady