# %%
import sys
import os

sys.path.append(os.getenv("CODE_PATH"))
sys.path.append(os.getenv("FIN_DATABASE_PATH"))

import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select, and_, or_
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from algo_trading.log_config import setup_logging
from Fin_Database.Data.connect import engine, DailyStockData, HourlyStockData, OneMinuteStockData, FiveMinuteStockData,FifteenMinuteStockData, StockSplits, StockNews, CompanyFinancials

# %%
load_dotenv()


# %%
username = os.getenv("DATABASE_USERNAME")
password = os.getenv("DATABASE_PASSWORD")
host = os.getenv("DATABASE_HOST")
port = os.getenv("DATABASE_PORT")
database = os.getenv("DATABASE_NAME")


# %%
engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database}')

# %%
Session = sessionmaker(bind = engine)
session = Session()

# %%
class DataFetcher:
    """ This class will fetch the data from Postgres SQL using SQL Alchemy. 
    It is flexible to allow the retrieve of data from any of the tables"""
    
    TABLE_MAPPING = {
        'minute': OneMinuteStockData,
        '5minutes': FiveMinuteStockData,
        '15minutes': FifteenMinuteStockData,
        'hour': HourlyStockData,
        'daily': DailyStockData
    }

    
    def __init__(self, tickers):
        if isinstance(tickers,str):
            self.tickers = [tickers]
        else:
            self.tickers = tickers
        
    def apply_date_filters(self, query, table, start_date, end_date):
        """Helper function to apply date filters to the query"""
        if start_date and end_date:
            logging.info(f"Applying date filters: start_date = {start_date}, end_date = {end_date}")
            query = query.filter(and_(table.date >= start_date, table.date <= end_date))
        elif start_date:
            logging.info(f"Applying start date filter: start_date = {start_date}")
            query = query.filter(table.date >= start_date)
        elif end_date:
            logging.info(f"Applying end date filter: end_date = {end_date}")
            query = query.filter(table.date <= end_date)
        
        return query



    def get_stock_data(self, timespan = 'daily', start_date = None, end_date = None, combine = False):
        logging.info(f"Fetching stock data for ticker(s) {self.tickers} with timespan '{timespan}'")
        
        table = self.TABLE_MAPPING.get(timespan)
        
        if not table:
            logging.error(f"Invalid timespan: {timespan}")
            raise ValueError(f"Invalid timespan: {timespan}")

        ticker_column = getattr(table, 'ticker_column')
        
        if combine:
            logging.info(f"Combining all tickers into a single dataframe")
        
            query = session.query(table).filter(getattr(table, ticker_column).in_(self.tickers))
            query = self.apply_date_filters(query, table, start_date, end_date)
            query = query.order_by(table.date)
            
            logging.info(f"Executing query for {self.tickers}")
            
            with engine.connect() as connection:
                result = pd.read_sql(query.statement, connection)
            
            logging.info(f"Succesfully fetched data for {self.tickers}")
        
        else:
            
            logging.info(f"Fetching data separately for each ticker into a dictionary of DataFrames.")
            dataframes = {}
            
            for ticker in self.tickers:
                logging.info(f"Fetching data for ticker: {ticker}")
                
                query = session.query(table).filter(getattr(table, ticker_column) == ticker)
                query = self.apply_date_filters(query, table, start_date, end_date)
                query = query.order_by(table.date)
                
                logging.info(f"Executing query for ticker: {ticker}")
                with engine.connect() as connection:
                    dataframes[ticker] = pd.read_sql(query.statement, connection)
                
                logging.info(f"Successfully fetched data for ticker: {ticker}")
            
            result = dataframes

        return result
            
    def get_company_data(self, combine=False):
        
        if combine:
            logging.info(f'Getting financial statement data for {self.tickers}')
            query = session.query(CompanyFinancials).filter(CompanyFinancials.tickers.in_(self.tickers))
            with engine.connect() as connection:
                result = pd.read_sql(query.statement, connection)
        else:
            dataframes = {}
            for ticker in self.tickers:
                logging.info(f'Getting financial statement data for {ticker}')
                query = session.query(CompanyFinancials).filter(CompanyFinancials.tickers.like(f'%{ticker}%'))
                with engine.connect() as connection:
                    dataframes[ticker] = pd.read_sql(query.statement, connection)
            result = dataframes

        return result
    



