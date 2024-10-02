# %%
#Imports 

import sys
import os
import yaml

sys.path.append(os.getenv("CODE_PATH"))
sys.path.append(os.getenv("FIN_DATABASE_PATH"))


import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from Data.connect import engine, DailyStockData, HourlyStockData, OneMinuteStockData, FiveMinuteStockData,FifteenMinuteStockData, StockSplits, StockNews, CompanyFinancials
from Pre_Processing.pre_processing import PreProcessing
from data_fetcher import DataFetcher
from Feature_Engineering.feature_engineering import TechnicalIndicators
from pipeline import Pipeline
import json
from pandas import json_normalize


class CalculateMetrics:
    
    def __init__(self, data):
        """ data is a dataframe with tickers as index."""
        self.data = data
        
    def profitability_ratios(self):
        """Calculates profitability ratios."""
        # data = self.self.data.copy()
        
        # Gross Margin
        self.data['gross_margin'] = self.data[('income_statement', 'gross_profit')] / self.data[('income_statement', 'revenues')]
        
        # Operating Margin
        self.data['operating_margin'] = self.data[('income_statement', 'operating_income_loss')] / self.data[('income_statement', 'revenues')]
        
        # Net Profit Margin
        self.data['net_profit_margin'] = self.data[('income_statement', 'net_income_loss')] / self.data[('income_statement', 'revenues')]
        
        #ROA
        self.data['ROA'] = self.data[('income_statement', 'net_income_loss')] / self.data[('balance_sheet', 'assets')]
        
        #ROE
        self.data['ROE'] = self.data[('income_statement', 'net_income_loss')] / self.data[('balance_sheet', 'equity')]
        
        return self.data
    
    def liquidity_ratios(self):
        """ Calculates liquidity ratios."""
        # data = self.self.data.copy()

        #Current Ratio
        self.data['current_ratio'] = self.data[('balance_sheet', 'current_assets')] / self.data[('balance_sheet', 'current_liabilities')]
        
        #Quick Ratio
        self.data['quick_ratio'] = (self.data[('balance_sheet', 'current_assets')] - self.data[('balance_sheet', 'inventory')]) / self.data[('balance_sheet', 'current_liabilities')]
        
    
    def other_ratios(self):
        """Calculates other ratios."""
        # data = self.self.data.copy()
        
        #Debt to Equity
        self.data['debt_to_equity'] = self.data[('balance_sheet', 'liabilities')] / self.data[('balance_sheet', 'equity')]
        
        #Interest Coverage
        self.data['interest_coverage'] = self.data[('income_statement', 'operating_income_loss')] / self.data[('income_statement', 'interest_expense_operating')]
        self.data['R&D_ratio'] = self.data[('income_statement', 'research_and_development')] / self.data[('income_statement', 'revenues')]
        
        return self.data
    
    def calculate_metrics(self):
        """Orchestrates the calculation of financial metrics."""
        self.profitability_ratios()
        self.liquidity_ratios()
        self.other_ratios()
        
        return self.data
        


