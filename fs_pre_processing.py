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


class PreProcessingFinancials:
    def __init__(self, data, sections, tickers):
        """
        args: 
        data: dictionary with multiple tickers
        sections: list with sections from a Financial Statement
        tickers: list with tickers we want to analyse
        """
        if isinstance(data,pd.DataFrame): #If we pass only one ticker, which would be only one dataframe, we transform into a dictionary.
            self.data = {}
            self.data[tickers] = data
            self.tickers = [tickers]
        else:
            self.data = data
            self.tickers = tickers 
        self.sections = sections
        
    def adjust_data(self, data):
        """Adjusting filling date for TTM, Q4, FY, as they don't have any filing dates.
        Using as a proxy the end date of the period plus 37 days, which is the average time it takes to file the 10-K/10-Q."""
        conditions = (
            (data['filing_date'].isna()) &
            (data['fiscal_period'].isin(['TTM', 'Q4', 'FY']))
                        )
    
        data['filing_date'] = np.where(conditions, data['end_date'] + timedelta(days=37), data['filing_date'])
      
        return data
    
    def replacing_nan(self,data):
        """Replace NaN values in the fiscal year column with the correct date."""
        data['period'] = np.where(data['fiscal_'])
        
    
    def flatten_json_section(self):
        """Preprocess the financials column by flattening JSON fields and handling filing dates/fiscal periods."""
        for ticker in self.tickers:
            data = self.data[ticker]
            
            # Converting JSON strings into Python dictionaries if necessary
            data['financials'] = data['financials'].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )

            # Flattening each section and concatenate with the main dataframe
            for section in self.sections:
                flattened = self.flattening(data, 'financials', section)
                data = pd.concat([data, flattened], axis=1)

            # Handle filing dates and fiscal periods
            data['filing_date'] = pd.to_datetime(data['filing_date'])

            #Adjust filing dates before setting index
            data = self.adjust_data(data)

            #Sort by filing date 
            data.sort_values(by='filing_date', inplace=True, ascending=False)

            self.data[ticker] = data
    
    
    
    def flattening(self, data, json_col, section):
        """ Helper function to flatten a JSON section of the financials dataframe
        """
        section_data = data[json_col].apply(lambda x: x.get(section) if section in x else {})
        flattened_section = json_normalize(section_data)  # Flatten the section
        flattened_section.columns = [f"{section}_{col}" for col in flattened_section.columns]  # Add prefix to columns
                
        return flattened_section

    
    def removing_cols(self):
        """ This function cleans the dataframe by dropping columns with '.unit' in the name and '.order' in the name.
        If .unit columns are the same for each row, we will drop.
        Also dropping columns that have .order
        """
        
        for ticker in self.tickers:
            data = self.data[ticker]
            # print(f'Processing ticker {ticker}')
            for section in self.sections:
                section_columns = [col for col in data.columns if section in col]
                section_df = data[section_columns].copy()
            
                #Removing .order columns
                order_columns = [col for col in section_df.columns if '.order' in col]
                # if order_columns:
                #     print(f"Found '.order' columns for {ticker} in {section}: {order_columns}")  # Debugging
                # else:
                #     print(f"No '.order' columns found for {ticker} in {section}") 
                section_df.drop(columns=order_columns, inplace=True)
                
                #Removing .unit columns if they only have one unique value
                unit_columns = [col for col in section_df.columns if '.unit' in col]
                # print(f"Found '.unit' columns for {ticker} in {section}: {unit_columns}")  # Debugging
 
                for col in unit_columns:
                    if section_df[col].nunique() == 1:
                        section_df.drop(columns=col, inplace=True)
        
                
                label_columns = [col for col in section_df.columns if '.label' in col]
                section_df.drop(columns=label_columns, inplace=True)
                
                
                #Converting numeric values to millions
                value_columns = [col for col in section_df.columns if '.value' in col]
                for col in value_columns:
                    section_df[col] = pd.to_numeric(section_df[col])
                    section_df[col] = section_df[col]/1000000
                    # print(f"Converted {col} to millions for {ticker} in {section}")  # Debugging
                data.drop(columns=section_columns, inplace=True)  # Remove the original section columns
                data = pd.concat([data, section_df], axis=1)
               
            self.data[ticker] = data
    
    def transform_columns(self, data):
        """Transform data column data into required type"""
        
        # Replacing empty strings with np.nan in the entire dataframe at once
        data.replace('', np.nan, inplace=True)
        
        # Convert specific columns to required types
        data['fiscal_year'] = pd.to_numeric(data['fiscal_year'], errors='coerce')
        data['fiscal_period'] = data['fiscal_period'].astype('category')
        data['start_date'] = pd.to_datetime(data['start_date'], errors='coerce')
        data['end_date'] = pd.to_datetime(data['end_date'], errors='coerce')
        
    def create_period(self, row):
        if row['timeframe'] == 'quarterly':
            quarter = int(row['fiscal_period'].replace('Q', ''))
            year = int(row['fiscal_year'])
            return pd.Period(freq='Q', year=year, quarter=quarter)
        else:
            return np.nan
        
        
    def preprocess_financials(self, columns):
        """Orchestrates the entire pre-processing of financials."""
        self.flatten_json_section()  
        self.removing_cols() 
        
        # Step 3: Create multi-indexed DataFrame for each ticker
        processed_data = {}
        
        for ticker in self.tickers:
            data = self.data[ticker]
            
            # Extract general columns from the data
            general_df = data[columns].copy()

            # If 'filing_date' is not already an index, set it as the index in general_df
            if 'filing_date' in general_df.columns:
                general_df.set_index('filing_date', inplace=True)
            
            # Proccessing each section
            section_dataframes = []
            
           
            for section in self.sections:
                # Filter columns related to the current section
                section_columns = [col for col in data.columns if section in col]
                section_df = data[['filing_date'] + section_columns].copy()  # Ensure 'filing_date' is included

                # Set 'filing_date' as the index for the section to align it properly
                section_df.set_index('filing_date', inplace=True)

                # Removing '.value' suffix from the column names
                section_df.columns = section_df.columns.str.replace('.value', '', regex=False)

                # Removing the section name from the second-level column names
                section_df.columns = pd.MultiIndex.from_product(
                    [[section], section_df.columns.str.replace(f'{section}_', '', regex=False)]
                )

                # Add this section DataFrame to the list
                section_dataframes.append(section_df)
            
            # Concatenate all section DataFrames into one DataFrame (financial data)
            financial_data = pd.concat(section_dataframes, axis=1)

            # Concatenate general_df (with general columns) and financial_data (with sections)
            full_data = pd.concat([general_df, financial_data], axis=1)

            # Store the processed DataFrame
            processed_data[ticker] = full_data
            
        combined_data = pd.concat(processed_data.values(), keys=processed_data.keys(), names=['ticker'])
        self.transform_columns(combined_data)
        combined_data['period'] = combined_data.apply(self.create_period, axis=1)
        
        
        return processed_data, combined_data