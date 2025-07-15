%%writefile app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Title and description
st.title("📊 Fundamental Stock Analysis Dashboard")
st.markdown("""
This tool analyzes stocks based on fundamental factors and provides a scoring system to evaluate investment potential.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    
    # Stock selection
    default_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "WMT", "NFLX"]
    tickers = st.multiselect("Select Stocks (max 10)", default_tickers, default=default_tickers[:3])
    
    # Analysis period
    end_date = datetime.today()
    start_date = end_date - timedelta(days=10*365)  # 10 years data
    st.caption(f"Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Industry averages (user can adjust)
    st.subheader("Industry Benchmark Averages")
    industry_sales_cagr = st.number_input("Industry Avg Sales CAGR (%)", value=10.0)
    industry_pat_cagr   = st.number_input("Industry Avg PAT CAGR (%)", value=9.0)
    industry_de         = st.number_input("Industry Avg Debt-to-Equity", value=0.6)
    industry_pe         = st.number_input("Industry Avg P/E Ratio", value=18.0)
    industry_roe        = st.number_input("Industry Avg ROE (%)", value=15.0)

# Helper functions
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist = stock.history(period="10y")
        
        # Get financials
        financials    = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow      = stock.cashflow
        
        # Get info dict
        info = stock.info
        
        return {
            'stock': stock,
            'ticker': ticker,
            'hist': hist,
            'financials': financials,
            'balance_sheet': balance_sheet,
            'cashflow': cashflow,
            'info': info
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_cagr(start_value, end_value, years):
    if start_value == 0 or years == 0:
        return 0.0
    return ((end_value / start_value) ** (1/years) - 1) * 100

def calculate_fundamentals(data):
    import numpy as np

    if not data:
        return None

    try:
        stock         = data['stock']
        ticker        = data['ticker']
        financials    = data['financials']
        balance_sheet = data['balance_sheet']
        cashflow      = data['cashflow']
        info          = data['info']

        # Helper: pick the first row containing keyword (case-insensitive) and drop NaNs
        def safe_series(df, keyword):
            matches = [idx for idx in df.index if keyword.lower() in idx.lower()]
            if not matches:
                return pd.Series(dtype=float)
            s = df.loc[matches[0]].dropna()
            return s if isinstance(s, pd.Series) else pd.Series(dtype=float)

        # Helper: compute CAGR given a series (oldest→newest)
        def cagr(s):
            if len(s) < 2:
                return 0.0
            start, end = s.iloc[-1], s.iloc[0]
            if start <= 0 or end <= 0:
                return 0.0
            return ((end / start) ** (1 / (len(s) - 1)) - 1) * 100

        # 1) Sales & PAT CAGRs
        rev_s = safe_series(financials, 'revenue')
        ni_s  = safe_series(financials, 'net income')
        sales_cagr = cagr(rev_s)
        pat_cagr   = cagr(ni_s)

        # 2) Debt-to-Equity
        raw_de = info.get('debtToEquity')
        if raw_de not in (None, 0):
            de_ratio = float(raw_de)
        else:
            try:
                # sum any lines containing “debt”
                debt_vals = []
                for row in balance_sheet.index:
                    if 'debt' in row.lower():
                        val = balance_sheet.loc[row].dropna()
                        if len(val):
                            debt_vals.append(val.iloc[0])
                total_debt = sum(debt_vals)
                eq_s       = safe_series(balance_sheet, 'total stockholder equity')
                equity     = eq_s.iloc[0] if len(eq_s) else 0.0
                de_ratio   = (total_debt / equity) if equity else 0.0
            except:
                de_ratio = 0.0

        # 3) P/E Ratio
        pe_ratio = float(info.get('trailingPE') or 0.0)

        # 4) ROE %
        raw_roe = info.get('returnOnEquity')
        if raw_roe is not None:
            roe = float(raw_roe) * 100
        else:
            try:
                latest_ni = ni_s.iloc[0] if len(ni_s) else 0.0
                eq_s      = safe_series(balance_sheet, 'total stockholder equity')
                latest_eq = eq_s.iloc[0] if len(eq_s) else 0.0
                roe       = (latest_ni / latest_eq) * 100 if latest_eq else 0.0
            except:
                roe = 0.0

        # 5) Operating Cash Flow & 5Y Avg
        opcf_s = safe_series(cashflow, 'operating activities')
        if len(opcf_s):
            cfo       = float(opcf_s.iloc[0])
            cfo_5yr   = float(opcf_s.mean())
        else:
            cfo, cfo_5yr = 0.0, 0.0

        # 6) Free Cash Flow & 5Y Avg
        fcf_s = safe_series(cashflow, 'free cash flow')
        if len(fcf_s):
            fcf      = float(fcf_s.iloc[0])
            fcf_5yr  = float(fcf_s.mean())
        else:
            fcf, fcf_5yr = 0.0, 0.0

        # 7) Quarterly YoY growth
        try:
            qfin   = stock.quarterly_financials
            rev_q  = safe_series(qfin, 'revenue')
            ni_q   = safe_series(qfin, 'net income')
            rev_growth = ((rev_q.iloc[0] - rev_q.iloc[4]) / abs(rev_q.iloc[4]) * 100) if len(rev_q) >= 5 and rev_q.iloc[4] else 0.0
            ni_growth  = ((ni_q.iloc[0]  - ni_q.iloc[4])  / abs(ni_q.iloc[4])  * 100) if len(ni_q)  >= 5 and ni_q.iloc[4]  else 0.0
        except:
            rev_growth, ni_growth = 0.0, 0.0

        # Assemble results
        result = {
            'Ticker': ticker,
            'Sales CAGR (5Y) %': sales_cagr,
            'PAT CAGR (5Y) %': pat_cagr,
            'Debt-to-Equity': de_ratio,
            'P/E Ratio': pe_ratio,
            'ROE %': roe,
            'CFO (Latest)': cfo,
            'FCF (Latest)': fcf,
            'Revenue Growth (YoY) %': rev_growth,
            'Net Income Growth (YoY) %': ni_growth,
            'CFO 5Y Avg': cfo_5yr,
            'FCF 5Y Avg': fcf_5yr
        }

        # Sanitize any remaining NaN or inf values
        for k, v in result.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                result[k] = 0.0

        return result

    except Exception as e:
        st.error(f"Error calculating fundamentals for {data.get('ticker','')}: {e}")
        return None



def calculate_scores(fundamentals, industry_avgs):
    scores = {
        'Financial Health': 0,
        'Valuation': 0,
        'Industry Position': 0,
        'Management': 0,  # Placeholder - would need additional data
        'Risk Factors': 0   # Placeholder - would need additional data
    }
    
    # Financial Health (max 10 points)
    # Revenue & Profit Growth (2 points)
    if fundamentals['Sales CAGR (5Y) %'] > industry_avgs['sales_cagr']:
        scores['Financial Health'] += 1
    if fundamentals['PAT CAGR (5Y) %'] > industry_avgs['pat_cagr']:
        scores['Financial Health'] += 1
    
    # EPS growth (using NI growth as proxy) (2 points)
    if fundamentals['Net Income Growth (YoY) %'] > 0:
        scores['Financial Health'] += 1
    if fundamentals['Revenue Growth (YoY) %'] > 0:
        scores['Financial Health'] += 1
    
    # Debt-to-Equity (2 points)
    if fundamentals['Debt-to-Equity'] < industry_avgs['de']:
        scores['Financial Health'] += 2
    elif fundamentals['Debt-to-Equity'] < 1.0:
        scores['Financial Health'] += 1
    
    # Free Cash Flow (2 points)
    if fundamentals['FCF (Latest)'] > 0:
        scores['Financial Health'] += 1
    if fundamentals['FCF 5Y Avg'] > 0:
        scores['Financial Health'] += 1
    
    # ROE (2 points)
    if fundamentals['ROE %'] > industry_avgs['roe']:
        scores['Financial Health'] += 2
    elif fundamentals['ROE %'] > 15:
        scores['Financial Health'] += 1
    
    # Valuation (max 10 points)
    # P/E Ratio (3 points)
    if fundamentals['P/E Ratio'] < industry_avgs['pe']:
        scores['Valuation'] += 3
    elif fundamentals['P/E Ratio'] < industry_avgs['pe'] * 1.2:
        scores['Valuation'] += 1
    
    # P/B Ratio (3 points) - placeholder
    # Assuming P/B is reasonable if P/E is reasonable
    if fundamentals['P/E Ratio'] < industry_avgs['pe']:
        scores['Valuation'] += 3
    elif fundamentals['P/E Ratio'] < industry_avgs['pe'] * 1.2:
        scores['Valuation'] += 1
    
    # PEG Ratio (4 points) - placeholder
    # Assuming PEG is reasonable if growth is good
    if fundamentals['PAT CAGR (5Y) %'] > industry_avgs['pat_cagr'] and fundamentals['P/E Ratio'] < industry_avgs['pe']:
        scores['Valuation'] += 4
    elif fundamentals['PAT CAGR (5Y) %'] > industry_avgs['pat_cagr']:
        scores['Valuation'] += 2
    
    # Industry Position (max 10 points) - simplified
    # Assuming all stocks are in the same industry for this demo
    scores['Industry Position'] = 6  # Baseline
    if fundamentals['Sales CAGR (5Y) %'] > industry_avgs['sales_cagr']:
        scores['Industry Position'] += 2
    if fundamentals['ROE %'] > industry_avgs['roe']:
        scores['Industry Position'] += 2
    
    # Management (placeholder) - assume average
    scores['Management'] = 6
    
    # Risk Factors (placeholder) - assume average
    scores['Risk Factors'] = 6
    
    # Calculate total score
    total_score = sum(scores.values())
    
    return {
        **scores,
        'Total Score': total_score,
        'Recommendation': 'Strong Buy' if total_score >= 36 else 
                          'Buy' if total_score >= 30 else 
                          'Hold' if total_score >= 21 else 
                          'Sell'
    }

# Main app
if not tickers:
    st.warning("Please select at least one stock to analyze.")
else:
    if len(tickers) > 10:
        st.error("Please select no more than 10 stocks.")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Fetch and process data
    all_data = []
    fundamentals_list = []
    scores_list = []
    
    industry_avgs = {
        'sales_cagr': industry_sales_cagr,
        'pat_cagr': industry_pat_cagr,
        'de': industry_de,
        'pe': industry_pe,
        'roe': industry_roe
    }
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"Fetching data for {ticker} ({i+1}/{len(tickers)})...")
        progress_bar.progress((i + 1) / len(tickers))
        
        data = get_stock_data(ticker)
        if data:
            all_data.append(data)
            
            # Calculate fundamentals
            fundamentals = calculate_fundamentals(data)
            if fundamentals:
                fundamentals_list.append(fundamentals)
                
                # Calculate scores
                scores = calculate_scores(fundamentals, industry_avgs)
                scores_list.append({
                    'Ticker': ticker,
                    **scores
                })
    
    status_text.text("Analysis complete!")
    progress_bar.empty()
    
    if not fundamentals_list:
        st.error("No data could be fetched for the selected tickers.")
        st.stop()
    
    # Display results
    st.header("Fundamental Analysis Results")
    
    # Create dataframe for fundamentals
    fundamentals_df = pd.DataFrame(fundamentals_list)
    fundamentals_df.set_index('Ticker', inplace=True)
    
    # Format columns
    formatted_df = fundamentals_df.copy()
    for col in formatted_df.columns:
        if '%' in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}%")
        elif 'Ratio' in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.1f}x")
        elif 'Debt-to-Equity' in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")
        elif 'Avg' in col or 'Latest' in col:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"${x/1e9:.1f}B" if abs(x) >= 1e9 else f"${x/1e6:.1f}M" if abs(x) >= 1e6 else f"${x:.1f}")
    
    # Add industry average row
    industry_row = {
        'Sales CAGR (5Y) %': f"{industry_avgs['sales_cagr']:.1f}%",
        'PAT CAGR (5Y) %': f"{industry_avgs['pat_cagr']:.1f}%",
        'Debt-to-Equity': f"{industry_avgs['de']:.2f}",
        'P/E Ratio': f"{industry_avgs['pe']:.1f}x",
        'ROE %': f"{industry_avgs['roe']:.1f}%",
        'CFO (Latest)': "N/A",
        'FCF (Latest)': "N/A",
        'Revenue Growth (YoY) %': "N/A",
        'Net Income Growth (YoY) %': "N/A",
        'CFO 5Y Avg': "N/A",
        'FCF 5Y Avg': "N/A"
    }
    formatted_df.loc['Industry Avg'] = industry_row
    
    # Display fundamentals table
    st.subheader("Fundamental Metrics Comparison")

    st.dataframe(
        formatted_df.style.applymap(
            lambda x: 'color: green' if ('%' in x and float(x.replace('%','')) > 0) else 
                      ('color: red' if '%' in x and float(x.replace('%','')) < 0 else ''),
            subset=[col for col in formatted_df.columns if '%' in col]
        )
    )

    
    # Display scores
    st.subheader("Stock Scoring System")
    scores_df = pd.DataFrame(scores_list)
    scores_df.set_index('Ticker', inplace=True)
    
    # Format scores
    # 1) Component score text colors (unchanged)
    def color_score(val):
        if val >= 7:
            return 'color: green; font-weight: bold;'
        elif val >= 4:
            return 'color: orange; font-weight: bold;'
        else:
            return 'color: red; font-weight: bold;'

    # 2) Total-Score bands, with light backgrounds + contrasting text
    def style_total(val):
        if val >= 36:
            return (
                'background-color: rgba(46, 125, 50, 0.2); '  # light green wash
                'color: #1b5e20; font-weight: bold;'
            )
        elif val >= 30:
            return (
                'background-color: rgba(165, 214, 167, 0.2); '  # pale mint
                'color: #2e7d32; font-weight: bold;'
            )
        elif val >= 21:
            return (
                'background-color: rgba(255, 249, 196, 0.2); '  # pale yellow
                'color: #f9a825; font-weight: bold;'
            )
        else:
            return (
                'background-color: rgba(239, 154, 154, 0.2); '  # pale red
                'color: #c62828; font-weight: bold;'
            )

    # 3) Recommendation text
    def reco_color(val):
        if val in ['Strong Buy','Buy']:
            return 'color: green; font-weight: bold;'
        elif val == 'Hold':
            return 'color: orange; font-weight: bold;'
        else:
            return 'color: red; font-weight: bold;'

    st.dataframe(
        scores_df.style
            .applymap(color_score,
                      subset=['Financial Health','Valuation','Industry Position','Management','Risk Factors'])
            .applymap(style_total, subset=['Total Score'])
            .applymap(reco_color,  subset=['Recommendation'])
    )

    
    # Display interpretation
    st.subheader("Interpretation Guide")
    st.markdown("""
    - **Financial Health (0-10):** Evaluates revenue/profit growth, debt levels, cash flow, and ROE
    - **Valuation (0-10):** Assesses whether the stock is undervalued based on P/E, P/B, and PEG ratios
    - **Industry Position (0-10):** Evaluates the company's competitive position in its industry
    - **Management (0-10):** Assesses leadership quality and corporate governance (placeholder)
    - **Risk Factors (0-10):** Evaluates volatility and economic sensitivity (placeholder)
    
    **Total Score Interpretation:**
    - 36-50: Strong investment opportunity
    - 21-35: Medium risk, consider with caution
    - 0-20: High risk, avoid investment
    """)
    
    # Visualizations
    st.subheader("Key Metrics Visualization")
    
    # Select metric to visualize
    metric_options = ['Sales CAGR (5Y) %', 'PAT CAGR (5Y) %', 'Debt-to-Equity', 
                     'P/E Ratio', 'ROE %', 'Revenue Growth (YoY) %', 'Net Income Growth (YoY) %']
    selected_metric = st.selectbox("Select metric to visualize", metric_options)
    
    # Prepare data for visualization
    viz_df = fundamentals_df.copy().drop('Industry Avg', errors='ignore')
    if '%' in selected_metric:
        viz_df[selected_metric] = viz_df[selected_metric].astype(str).str.replace('%','').astype(float)
    
    # Create bar chart
    fig = go.Figure()
    
    if selected_metric in ['Sales CAGR (5Y) %', 'PAT CAGR (5Y) %', 'ROE %']:
        fig.add_trace(go.Bar(
            x=viz_df.index,
            y=viz_df[selected_metric],
            name=selected_metric,
            marker_color='green'
        ))
        fig.add_hline(y=industry_avgs[selected_metric.split()[0].lower() + '_cagr'] if 'CAGR' in selected_metric else industry_avgs[selected_metric.split()[0].lower()],
                     line_dash="dash", line_color="red",
                     annotation_text="Industry Avg", 
                     annotation_position="bottom right")
    elif selected_metric in ['Debt-to-Equity', 'P/E Ratio']:
        fig.add_trace(go.Bar(
            x=viz_df.index,
            y=viz_df[selected_metric],
            name=selected_metric,
            marker_color='blue'
        ))
        fig.add_hline(y=industry_avgs['de'] if selected_metric == 'Debt-to-Equity' else industry_avgs['pe'],
                     line_dash="dash", line_color="red",
                     annotation_text="Industry Avg", 
                     annotation_position="bottom right")
    else:
        fig.add_trace(go.Bar(
            x=viz_df.index,
            y=viz_df[selected_metric],
            name=selected_metric,
            marker_color='purple'
        ))
    
    fig.update_layout(
        title=f"{selected_metric} Comparison",
        xaxis_title="Ticker",
        yaxis_title=selected_metric,
        hovermode="x"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display raw data for debugging
    with st.expander("Show raw data (for debugging)"):
        st.write("Fundamentals Data:")
        st.write(fundamentals_df)
        st.write("Scores Data:")
        st.write(scores_df)