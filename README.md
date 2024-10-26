### Financial Dashboard Project

This is a simple and interactive dashboard that allows you to visualize key financial metrics of DJIA companies.

---

### Overview

The purpose of this project is to dynamically fetch financial data from a previously created SQL database ([Fin_Database](https://github.com/rravinet/Fin_Database)), preprocess it to extract the necessary metrics, and display it in a user-friendly dashboard. Users can select one or more companies (tickers), view important financial ratios and metrics, and visualize data using different types of charts.

### Project Structure

Here’s a breakdown of the key components:

- **Data Gathering**: Data is pulled from a financial database via PostGresSQL using the `DataFetcher` class, which gathers all the necessary financial information for the selected companies.
- **Preprocessing**: Once the data is fetched, the `PreProcessingFinancials` class handles all the required transformations and manipulations of the raw financial data. This includes parsing balance sheets, income statements, and other sections to prepare the data for calculations.
- **Metrics Calculation**: After preprocessing, the `CalculateMetrics` class takes over. It calculates a series of important financial ratios and metrics such as Current Ratio, Debt-to-Equity Ratio, Accounts Receivable, etc. These metrics are directly used in the dashboard.
- **Dashboard Display**: Finally, the pre-processed and calculated data is displayed in a simple, one-page **Streamlit** dashboard. The user can select the tickers they want to analyze, and the relevant financial metrics are dynamically displayed. **Plotly** is also used to visualize trends and comparisons between companies.

### Under Construction

This dashboard is still under construction and the next steps currently being worked on are:

- Enhancing dashboard layout
- Adding more visualizations for trends over time (e.g., accounts receivable growth).
- Adding more filtering options (e.g., fiscal periods, specific sections like income statements or cash flow statements).
- Improving caching for performance, especially with larger datasets.

---
