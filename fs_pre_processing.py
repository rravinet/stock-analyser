# %%
import sys
import os

sys.path.append('/Users/raphaelravinet/Code')

import pandas as pd
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, select
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from algo_trading.log_config import setup_logging
import json
from datetime import datetime

# %%
setup_logging()

        
class FinancialDataPreProcessing:
    def __init__(self, df):
        self.df = df
    
    def preprocess_financials(self):
        """Preprocess the financials column by flattening JSON fields."""
        self.df['financials'] = self.df['financials'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        #That's the name of our column from the API
        
        def flatten_json_section(df, json_col, section):
            for idx, row in df.iterrows():
                if section in row[json_col]:
                    section_data = row[json_col][section]
                    for key, value in section_data.items():
                        col_name = f"{section}_{key}"
                        df.loc[idx, col_name] = value['value'] if 'value' in value else None
        
        sections = ['balance_sheet', 'income_statement', 'cash_flow_statement', 'comprehensive_income']
        for section in sections:
            flatten_json_section(self.df, 'financials', section)
        
        return self

    def pre_processor(self):
        """Run all preprocessing steps for financial data."""
        self.preprocess_financials()
        return self


    



