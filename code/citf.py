# The following code is our implemented improvement for Trend Following.
# We used the 200 day Simple Moving Average, 14 period RSI and 
# current prices of the assets

import numpy as np
from datetime import datetime
from System.Drawing import Color

class BasicTemplateAlgorithm(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2015, 1, 1)  
        self.SetEndDate(2020, 1, 1)    
        self.SetCash(100000)           
        self.data = {}
        self.macd = {}
        self.rsi_values = {}
        
        fastPeriod = 12
        slowPeriod = 26
        signalPeriod = 9
        period = 200
        
        self.SetWarmUp(period)
        # self.symbols = ["SPY", "EFA", "BND", "VNQ", "GSG"]
        self.symbols = [
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
        
        self.fast_sma = {}
        self.sma_history = {}
        self.price_history = {}
        
        for symbol in self.symbols:
            self.AddEquity(symbol, Resolution.Daily)
            self.data[symbol] = self.SMA(symbol, period, Resolution.Daily)
            self.rsi_values[symbol] = self.RSI(symbol, 14,  MovingAverageType.Simple, Resolution.Daily)
            self.fast_sma[symbol] = self.SMA(symbol, 50, Resolution.Daily)
            self.sma_history[symbol] = RollingWindow[float](3)
            self.price_history[symbol] = RollingWindow[float](3)
            self.price_history[symbol].Add(-1)
            
            self.macd[symbol] = self.MACD(symbol, fastPeriod, slowPeriod, signalPeriod, MovingAverageType.Exponential, Resolution.Daily)
            self.__previous = datetime.min
            
        # charts
        # plot sma, current price to see the crossovers
        # plot rsi to see how it compares with current price
        # plot the 70, 30 rsi lines to see the overbought, solds
        
        PricePlot = Chart("Price Plot")
        PricePlot.AddSeries(Series("200 Day SMA", SeriesType.Line,"", Color.Red))
        PricePlot.AddSeries(Series("Current Price", SeriesType.Line,"", Color.Blue))
        
        RSIPlot = Chart("RSI Plot")
        RSIPlot.AddSeries(Series("RSI", SeriesType.Line,"", Color.Green))
        RSIPlot.AddSeries(Series("Overbought", SeriesType.Line,"", Color.Black))
        RSIPlot.AddSeries(Series("Oversold", SeriesType.Line, "", Color.Black))
        
        # setting benchmark for graph
        self.SetBenchmark("SPY")
        # Variable to hold the last calculated SPY value
        self.lastSPYValue = None
        # Inital benchmark value scaled to be the same as portfolio starting cash
        self.SPYPerformance = self.Portfolio.TotalPortfolioValue
        # Performance plot
        PerformancePlot = Chart("PerformancePlot")
        PerformancePlot.AddSeries(Series("Combined Strategy", SeriesType.Line,"", Color.Red))
        PerformancePlot.AddSeries(Series("SPY", SeriesType.Line,"", Color.Blue))
        PerformancePlot.AddSeries(Series("Original", SeriesType.Line, "", Color.Black))
            
        self.AddChart(PricePlot)
        self.AddChart(RSIPlot)
        self.AddChart(PerformancePlot)
        

    def OnData(self, data):
        
        if self.IsWarmingUp: return
    
        tolerance = 0.0015
        isUptrend = []
        
        for symbol, sma in self.data.items():
            if not self.macd[symbol].IsReady: return
        
            # only once per day
            if self.__previous.date() == self.Time.date(): return
            
            holdings = self.Portfolio[symbol].Quantity
            
            self.sma_history[symbol].Add(sma.Current.Value)
            self.price_history[symbol].Add(self.Securities[symbol].Price)
            
            current_sma = sma.Current.Value
            current_price = self.Securities[symbol].Price
            
            if self.price_history[symbol][1] != -1:
                previous_sma = self.sma_history[symbol][1]
                previous_price = self.price_history[symbol][1]
                
            else:
                previous_sma = current_sma
                previous_price = current_price
            
            tolerance = 0.5
            
            # plot SPY
            if symbol == "SPY":
                self.plotting_info(symbol, sma)
                
            signalDeltaPercent = (self.macd[symbol].Current.Value - self.macd[symbol].Signal.Current.Value)/self.macd[symbol].Fast.Current.Value
            
            
            if (self.Securities[symbol].Price > sma.Current.Value 
            and current_price - current_sma <= tolerance
            and previous_price - previous_sma > tolerance
            and self.rsi_values[symbol].Current.Value < 30):
                self.SetHoldings(symbol, 1/len(self.symbols))
            else:
                self.Liquidate(symbol)
                
            self.__previous = self.Time
            
        # plot performance
        self.plotting_performance()
            
    def plotting_info(self, symbol, sma):
        self.Plot("Price Plot", "Current Price", float(self.Securities[symbol].Price))
        self.Plot("Price Plot", "200 Day SMA", float(sma.Current.Value))
        self.Plot("RSI Plot", "RSI", float(self.rsi_values[symbol].Current.Value))
        
    def plotting_performance(self):
        
        # Plot performance graph    
        benchmark = self.Securities["SPY"].Close
        
        # Update SPY performance if it's not the first periof performance
        if self.lastSPYValue is not  None:
           self.SPYPerformance = self.SPYPerformance * (benchmark/self.lastSPYValue)
           
        # store today's benchmark close price for use tomorrow
        self.lastSPYValue = benchmark
        
        # make our plots
        self.Plot("PerformancePlot", "Combined Strategy", self.Portfolio.TotalPortfolioValue)
        self.Plot("PerformancePlot", "SPY", self.SPYPerformance)
        self.Plot("PerformancePlot", "Original", 100000)