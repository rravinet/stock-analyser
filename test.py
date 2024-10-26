# %%
#Imports 
import sys
import os

sys.path.append(os.getenv("CODE_PATH"))
sys.path.append(os.getenv("FIN_DATABASE_PATH"))

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from Pre_Processing.pre_processing import PreProcessing
from data_fetcher import DataFetcher
from pipeline import Pipeline
from fs_pre_processing import PreProcessingFinancials
from metrics import CalculateMetrics
import openai
import streamlit as st
from datetime import datetime
# %%
openai.api_key = os.getenv("OPENAI_API_KEY")

# %%
GENERAL_COLUMNS = ['company_name', 'start_date', 'end_date', 'filing_date',
       'fiscal_period', 'fiscal_year', 'acceptance_datetime',
       'timeframe', 'tickers', 'sic']
SECTIONS = ['balance_sheet', 'income_statement', 'cash_flow_statement', 'comprehensive_income']


# %%
#Selecting some tickers to analyse
# tickers = ['AAPL', 'MSFT']
tickers = ['MSFT', 'AAPL']

# %%
#Fetching data from our SQL database

fetch_data = DataFetcher(tickers)
company_data = fetch_data.get_company_data()

#Pre Processing the data
prepocess = PreProcessingFinancials(company_data, SECTIONS, tickers)
data_dict, df = prepocess.preprocess_financials(GENERAL_COLUMNS)

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

# Caching to optimise data fetching
@st.cache_data
def get_cached_data(tickers, timespan='daily'):
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = tuple(sorted(tickers))
    # pipeline process or data fetching here
    pipe = Pipeline(list(tickers))
    data = pipe.pipeline(combine=True, timespan=timespan)
    return data

def fetch_and_process_data(tickers):
    #Fetching data from SQL database
    fetch_data = DataFetcher(tickers)
    company_data = fetch_data.get_company_data()
    
    #Preprocess the data
    preprocess = PreProcessingFinancials(company_data, SECTIONS, tickers)
    data_dict, df = preprocess.preprocess_financials(GENERAL_COLUMNS)
    
    # Calculating financial ratios
    metrics = CalculateMetrics(df)
    final_data = metrics.calculate_metrics()

    return final_data


st.title("Financial Dashboard")

# Landing page content
def show_landing_page():
    st.title("Market Overview")
    st.write(f"**Date**: {datetime.today().strftime('%B %d, %Y')}")

    # Placeholder graphs for S&P 500 and Nasdaq
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("S&P 500 Daily Performance")
        sp500_fig = go.Figure()
        sp500_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4000, 4100, 4150], mode='lines+markers', name='S&P 500'))
        st.plotly_chart(sp500_fig, use_container_width=True)

    with col2:
        st.subheader("Nasdaq Daily Performance")
        nasdaq_fig = go.Figure()
        nasdaq_fig.add_trace(go.Scatter(x=[1, 2, 3], y=[15000, 15200, 15150], mode='lines+markers', name='Nasdaq'))
        st.plotly_chart(nasdaq_fig, use_container_width=True)

    st.markdown("---")

    # News and Commentary placeholders
    st.subheader("Financial News")
    st.write("Fetching today's financial news...")  # Placeholder for future news API integration

    st.subheader("Market Commentary")
    st.write("Fetching today's market commentary...")  # Placeholder for commentary logic

## Financial Ratios Tab
def show_financial_ratios_tab():
    st.subheader("Financial Ratios")

    # Multiselect for Tickers with a unique key
    selected_tickers = st.multiselect(
        "Select Ticker(s)", 
        options=tickers, 
        default=[tickers[0]], 
        key="financial_ratios_tickers"
    )

    # Selectbox for Timeframe with a unique key
    selected_timeframe = st.selectbox(
        "Select Timeframe", 
        timeframes, 
        index=0, 
        key="financial_ratios_timeframe"
    )

    # Selectbox for Metric with a unique key
    selected_metric = st.selectbox(
        "Select Metric", 
        metrics, 
        index=0, 
        key="financial_ratios_metric"
    )

    # Selectbox for Graph Type with a unique key
    selected_graph_type = st.selectbox(
        "Select Graph Type", 
        graph_types, 
        index=0, 
        key="financial_ratios_graph_type"
    )

    # Plot Financial Ratios Graph
    if selected_tickers:
        final_data = fetch_and_process_data(selected_tickers)

        fig = go.Figure()
        for ticker in selected_tickers:
            filtered_df = final_data.xs(ticker, level='ticker')
            df_ticker = filtered_df[filtered_df['timeframe'] == selected_timeframe]
            x_axis = df_ticker['fiscal_year']

            # Plot the selected graph type
            if selected_graph_type == 'Line':
                fig.add_trace(go.Scatter(x=x_axis, y=df_ticker[selected_metric], mode='lines+markers', name=ticker))
            elif selected_graph_type == 'Bar':
                fig.add_trace(go.Bar(x=x_axis, y=df_ticker[selected_metric], name=ticker))
            elif selected_graph_type == 'Scatter':
                fig.add_trace(go.Scatter(x=x_axis, y=df_ticker[selected_metric], mode='markers', name=ticker))

        fig.update_layout(
            title=f"{selected_metric} for {', '.join(selected_tickers)} ({selected_timeframe})",
            xaxis_title='Period',
            yaxis_title=selected_metric,
            legend_title="Ticker"
        )
        st.plotly_chart(fig)

# Stock Data Tab
def show_stock_data_tab():
    st.subheader("Stock Data")

    # Multiselect for Tickers with a unique key
    selected_tickers = st.multiselect(
        "Select Ticker(s)", 
        options=tickers, 
        default=[tickers[0]], 
        key="stock_data_tickers"
    )

    # Selectbox for Timespan with a unique key
    selected_timespan = st.selectbox(
        "Select Timespan", 
        ['minute', 'hour', 'daily'], 
        index=2, 
        key="stock_data_timespan"
    )

    # Selectbox for Graph Type with a unique key
    selected_graph_type = st.selectbox(
        "Select Graph Type", 
        graph_types, 
        index=0, 
        key="stock_data_graph_type"
    )
    
    start_date = st.date_input(
        "Select Start Date", 
        value=datetime(2005, 1, 1),
        min_value=datetime(2000, 1, 1), 
        max_value=datetime.today(),
        key="stock_data_start_date"
    )

    end_date = st.date_input(
        "Select End Date", 
        value=datetime.today(), 
        min_value=datetime(2000, 1, 1),
        max_value=datetime.today(),
        key="stock_data_end_date"
    )

        

    # Filter the stock data based on the selected date range
    if selected_tickers:
        stock_data = get_cached_data(selected_tickers, selected_timespan)

        # Apply date range filtering
        filtered_data = stock_data[
            (stock_data['date'] >= pd.to_datetime(start_date)) & 
            (stock_data['date'] <= pd.to_datetime(end_date))
    ]

        # Multiselect for Indicators with a unique key
        indicator_columns = set(stock_data.columns) - {'id', 'date', 'timestamp', 'ticker'}
        selected_indicators = st.multiselect(
            "Select Technical Indicator(s)", 
            list(indicator_columns), 
            key="stock_data_indicators"
        )

        fig = go.Figure()
        for ticker in selected_tickers:
            ticker_data = filtered_data[filtered_data['ticker'] == ticker]
            for indicator in selected_indicators:
                if selected_graph_type == 'Line':
                    fig.add_trace(go.Scatter(
                        x=ticker_data['date'], y=ticker_data[indicator], 
                        mode='lines', name=f"{ticker} - {indicator}"
                    ))
                elif selected_graph_type == 'Bar':
                    fig.add_trace(go.Bar(
                        x=ticker_data['date'], y=ticker_data[indicator], 
                        name=f"{ticker} - {indicator}"
                    ))
                elif selected_graph_type == 'Scatter':
                    fig.add_trace(go.Scatter(
                        x=ticker_data['date'], y=ticker_data[indicator], 
                        mode='markers', name=f"{ticker} - {indicator}"
                    ))

        # Display the graph
        st.plotly_chart(fig)




# Main function to run the app
def main():
    # Define the tabs
    tab1, tab2, tab3 = st.tabs(["Market Overview", "Financial Ratios", "Stock Data"])

    # Display content based on the selected tab
    with tab1:
        show_landing_page()
    with tab2:
        show_financial_ratios_tab()
    with tab3:
        show_stock_data_tab()

# Run the app
if __name__ == "__main__":
    main()
