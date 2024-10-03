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
from fs_pre_processing import PreProcessingFinancials
from metrics import CalculateMetrics
import openai

# %%
openai.api_key = os.getenv("OPENAI_API_KEY")

# %%
GENERAL_COLUMNS = ['company_name', 'start_date', 'end_date', 'filing_date',
       'fiscal_period', 'fiscal_year', 'acceptance_datetime',
       'timeframe', 'tickers', 'sic']
SECTIONS = ['balance_sheet', 'income_statement', 'cash_flow_statement', 'comprehensive_income']


# %%
#Selecting some tickers to analyse
tickers = ['AAPL', 'MSFT']

# %%
#Fetching data from our SQL database

fetch_data = DataFetcher(tickers)
company_data = fetch_data.get_company_data()

# %%
#Pre Processing the data

prepocess = PreProcessingFinancials(company_data, SECTIONS, tickers)
data_dict, df = prepocess.preprocess_financials(GENERAL_COLUMNS)

# %%
#Calculating Financial ratios

metrics = CalculateMetrics(df)
final_data = metrics.calculate_metrics()

# %%
#List with values we will be using in the dashboard dropdowns

tickers = final_data.index.get_level_values(0).unique()
timeframes = final_data['timeframe'].unique()
metrics = ['gross_margin',
           'operating_margin',
           'net_profit_margin',
           'ROA',
           'ROE',
           'current_ratio',
           'quick_ratio',
           'debt_to_equity',
           'interest_coverage',
           'R&D_ratio']
graph_types = ['Line', 'Bar', 'Scatter']



# %%
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
from dash.exceptions import PreventUpdate
from flask_caching import Cache

# Initialising the Dash app
# app = dash.Dash(__name__)
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# caching to optimise
cache = Cache(app.server, config={'CACHE_TYPE': 'simple'})


# Layout
app.layout = html.Div([
    html.H1("Financial Dashboard"),

    # Dropdown to select the ticker
    dcc.Dropdown(
        id='select-ticker',
        options=[{'label': ticker, 'value': ticker} for ticker in tickers],
        value=tickers[0],  # default value set to the first ticker
        style={'width': '50%'}
    ),

    # Dropdown to select timespan
    dcc.Dropdown(
        id='select-timespan',
        options=[
            {'label': 'Minute', 'value': 'minute'},
            {'label': 'Hourly', 'value': 'hour'},
            {'label': 'Daily', 'value': 'daily'},
        ],
        value='daily',  # Default value
        style={'width': '50%'}
    ),
    


    # Tabs for organizing content
    dcc.Tabs(id='tabs', value='tab-financials', children=[
        dcc.Tab(label='Financial Ratios', value='tab-financials'),
        dcc.Tab(label='Stock Data', value='tab-stock')
    ]),
    html.Div(id='tabs-content')
])


# %%
@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def render_content(tab):
    if tab == 'tab-financials':
        return html.Div([
            # Financials tab content
            dcc.Dropdown(id='select-timeframe', options=[{'label': t, 'value': t} for t in timeframes],
                         value='quarterly', style={'width': '50%'}),
            dcc.Dropdown(id='select-metric', options=[{'label': m, 'value': m} for m in metrics],
                         value=metrics[0], style={'width': '50%'}),
            dcc.Dropdown(id='select-graph-type', options=[{'label': g, 'value': g} for g in graph_types],
                         value='Line', style={'width': '50%'}),
            dcc.Graph(id='financial-plot')
        ])
    elif tab == 'tab-stock':
        return html.Div([
            dcc.Dropdown(id='select-technical-indicator', options=[], value=None, multi = True, style={'width': '50%'}),
            dcc.Graph(id='stock-plot')
        ])
    return html.Div()  # Default empty content


# %%

@app.callback(
    Output('select-technical-indicator', 'options'),
    [Input('select-ticker', 'value'),
     Input('select-timespan', 'value'),
     Input('tabs', 'value')]  # Added Input to track the active tab
)
def update_indicator_options(selected_ticker, selected_timespan, tab):
    # Only update the technical indicators if the "Stock Data" tab is active
    if tab != 'tab-stock':
        raise PreventUpdate

    # Fetch and process data as usual
    data = get_cached_data(selected_ticker, selected_timespan)
    # non_indicator_columns = ['id', 'date', 'timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
    # indicator_options = [{'label': col, 'value': col} for col in data.columns if col not in non_indicator_columns]
    non_indicator_columns = ['id', 'date', 'timestamp', 'ticker']
    indicator_options = [{'label': col, 'value': col} for col in data.columns if col not in non_indicator_columns]
    return indicator_options

# Second Callback: Update Stock Plot Figure
@app.callback(
    Output('stock-plot', 'figure'),
    [Input('select-ticker', 'value'),
     Input('select-timespan', 'value'),
     Input('select-technical-indicator', 'value')]
)
def update_stock_graph(selected_ticker, selected_timespan, selected_indicators):
    data = get_cached_data(selected_ticker, selected_timespan)
    if not selected_indicators:
        # If no indicator is selected, use a default one or do not plot
        non_indicator_columns = ['id', 'date', 'timestamp', 'ticker']
        technical_indicators = [col for col in data.columns if col not in non_indicator_columns]
        selected_indicators = technical_indicators[:1] if technical_indicators else []
    
    fig = go.Figure()

    if selected_indicators:
        for indicator in selected_indicators:
            fig.add_trace(go.Scatter(
                x=data['date'],
                y=data[indicator],
                mode='lines',
                name=indicator
            ))
        indicators_str = ', '.join(selected_indicators)
        fig.update_layout(
            title=f"{indicators_str} for {selected_ticker} ({selected_timespan})"
        )
    else:
        fig.update_layout(title="No Technical Indicator Selected")
    
    return fig



# %%

@app.callback(
    Output('financial-plot', 'figure'),
    [Input('select-ticker', 'value'),
     Input('select-timeframe', 'value'),
     Input('select-metric', 'value'),
     Input('select-graph-type', 'value')]
)
def update_financial_graph(selected_ticker, selected_timeframe, selected_metric, selected_graph_type):
    # Fetch and process financial data
    # Assuming final_data is your preprocessed financial DataFrame

    # Filter the dataframe based on the selected ticker
    filtered_df = final_data[(final_data.index.get_level_values(0) == selected_ticker)]

    # Filter the dataframe based on the selected timeframe
    if selected_timeframe == 'quarterly':
        # Filtering only quarterly data
        filtered_df = filtered_df[filtered_df['timeframe'] == 'quarterly']
        filtered_df.sort_values(by='filing_date', ascending=True, inplace=True)
        # Use the 'period' column as the x-axis
        x_axis = filtered_df['period'].astype(str)
    else:
        # For FY or TTM, use fiscal_year as the x-axis
        filtered_df = filtered_df[filtered_df['timeframe'] == selected_timeframe]
        x_axis = filtered_df['fiscal_year']

    # Figure is based on the selected graph type
    if selected_graph_type == 'Line':
        fig = go.Figure(data=[
            go.Scatter(x=x_axis, y=filtered_df[selected_metric], mode='lines+markers')
        ])
    elif selected_graph_type == 'Bar':
        fig = go.Figure(data=[
            go.Bar(x=x_axis, y=filtered_df[selected_metric])
        ])
    elif selected_graph_type == 'Scatter':
        fig = go.Figure(data=[
            go.Scatter(x=x_axis, y=filtered_df[selected_metric], mode='markers')
        ])

    # Updating title and layout
    fig.update_layout(title=f"{selected_metric} for {selected_ticker} ({selected_timeframe})")
    return fig


# %%
# Caching function
@cache.memoize(timeout=300)  # Cache data for 5 minutes
def get_cached_data(ticker, timespan = 'daily'):
    # print(f"Fetching data for {ticker} with {timespan}")
    pipe = Pipeline([ticker])
    data = pipe.pipeline(combine=True, timespan=timespan)
    return data


# %%
if __name__ == '__main__':
    app.run_server(debug=True)


# %%



