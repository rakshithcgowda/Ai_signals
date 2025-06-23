import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="AI Options Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load sample data with better memory management
@st.cache_data(show_spinner="Loading data...")
def load_sample_data():
    try:
        # Generate synthetic options data more efficiently
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", end="2023-01-15")  # Reduced date range for demo
        symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE']
        strike_prices = {
            'NIFTY': np.arange(17000, 18000, 200),  # Reduced strike range
            'BANKNIFTY': np.arange(38000, 40000, 400),
            'RELIANCE': np.arange(2200, 2600, 100)
        }

        data = []
        for date in dates:
            for symbol in symbols:
                for strike in strike_prices[symbol][:5]:  # Only 5 strikes per symbol
                    for option_type in ['CE', 'PE']:
                        data.append({
                            'DATE': date,
                            'SYMBOL': symbol,
                            'STRIKE_PR': strike,
                            'OPTION_TYPE': option_type,
                            'OPEN_INT': np.random.randint(10000, 100000),
                            'CHG_IN_OI': np.random.randint(-10000, 10000),
                            'VOLUME': np.random.randint(1000, 10000),
                            'SETTLE_PR': np.random.uniform(10, 100),
                            'CLOSE': np.random.uniform(10, 100),
                            'IV': np.random.uniform(10, 30),
                            'OPEN': np.random.uniform(10, 100),
                            'HIGH': np.random.uniform(10, 100),
                            'LOW': np.random.uniform(10, 100)
                        })

        df = pd.DataFrame(data)
        # Convert to optimal dtypes to reduce memory
        df['DATE'] = pd.to_datetime(df['DATE'])
        for col in ['OPEN_INT', 'CHG_IN_OI', 'VOLUME']:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        for col in ['SETTLE_PR', 'CLOSE', 'IV', 'OPEN', 'HIGH', 'LOW']:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error("Failed to load data. Please try again.")
        return pd.DataFrame()

df = load_sample_data()

# Sidebar filters
st.sidebar.header("Filters")
try:
    selected_symbol = st.sidebar.selectbox("Symbol", df['SYMBOL'].unique())
    selected_date = st.sidebar.selectbox("Date", sorted(df['DATE'].unique(), reverse=True))
    selected_option_type = st.sidebar.radio("Option Type", ['CE', 'PE', 'All'], index=2)

    # Filter data based on selections
    filtered_df = df[(df['SYMBOL'] == selected_symbol) & (df['DATE'] == selected_date)]
    if selected_option_type != 'All':
        filtered_df = filtered_df[filtered_df['OPTION_TYPE'] == selected_option_type]
except Exception as e:
    logger.error(f"Filter error: {str(e)}")
    st.error("Error applying filters")
    filtered_df = pd.DataFrame()

# Main app
st.title("📊 AI-Powered Options Trading Platform")

# Tab layout
tab_names = ["Dashboard", "Option Screener", "Signal Generator", "Trend Prediction", "Strategy Assistant", "Backtesting"]
if len(df) > 0:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_names)
else:
    st.warning("No data available. Please check your data source.")
    st.stop()

with tab1:  # Dashboard
    try:
        st.header("Smart Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Open Interest Heatmap")
            if not filtered_df.empty:
                heatmap_data = filtered_df.pivot_table(
                    index='STRIKE_PR',
                    columns='OPTION_TYPE',
                    values='OPEN_INT',
                    aggfunc='sum'
                ).fillna(0)
                fig = px.imshow(heatmap_data, text_auto=True, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected filters")

        with col2:
            st.subheader("Volume vs Open Interest")
            if not filtered_df.empty:
                fig = px.scatter(
                    filtered_df,
                    x='VOLUME',
                    y='OPEN_INT',
                    color='OPTION_TYPE',
                    size=np.abs(filtered_df['CHG_IN_OI']),
                    hover_data=['STRIKE_PR'],
                    size_max=15
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for the selected filters")

        st.subheader("OI Change by Strike Price")
        if not filtered_df.empty:
            fig = px.bar(
                filtered_df,
                x='STRIKE_PR',
                y='CHG_IN_OI',
                color='OPTION_TYPE',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters")
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        st.error("Error displaying dashboard")

with tab2:  # Option Screener
    try:
        st.header("AI-Powered Option Screener")

        # Screener filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_oi = st.number_input("Minimum Open Interest", min_value=0, value=10000)
            expiry_range = st.selectbox("Expiry Range", ["This Week", "Next Week", "This Month"])
        with col2:
            min_volume = st.number_input("Minimum Volume", min_value=0, value=1000)
            iv_filter = st.selectbox("IV Filter", ["All", "High IV (>25%)", "Low IV (<15%)"])
        with col3:
            oi_change = st.number_input("Minimum OI Change", min_value=0, value=5000)
            price_filter = st.selectbox("Price Filter", ["All", "Under 50", "50-100", "Over 100"])

        # Apply filters
        screened_df = df.copy()
        screened_df = screened_df[
            (screened_df['OPEN_INT'] >= min_oi) &
            (screened_df['VOLUME'] >= min_volume) &
            (screened_df['CHG_IN_OI'].abs() >= oi_change)
        ]

        # IV filter
        if iv_filter == "High IV (>25%)":
            screened_df = screened_df[screened_df['IV'] > 25]
        elif iv_filter == "Low IV (<15%)":
            screened_df = screened_df[screened_df['IV'] < 15]

        # Price filter
        # Price filter
        if price_filter == "Under 50":
            screened_df = screened_df[screened_df['CLOSE'] < 50]
        elif price_filter == "50-100":
            screened_df = screened_df[(screened_df['CLOSE'] >= 50) & (screened_df['CLOSE'] <= 100)]
        elif price_filter == "Over 100":
            screened_df = screened_df[screened_df['CLOSE'] > 100]

        # Display results
        if not screened_df.empty:
            st.dataframe(
                screened_df.sort_values('CHG_IN_OI', ascending=False).head(50),
                column_config={
                    "DATE": st.column_config.DateColumn("Date"),
                    "SYMBOL": "Symbol",
                    "STRIKE_PR": st.column_config.NumberColumn("Strike", format="%.0f"),
                    "OPTION_TYPE": "Type",
                    "OPEN_INT": st.column_config.NumberColumn("OI", format="%,d"),
                    "CHG_IN_OI": st.column_config.NumberColumn("ΔOI", format="%,d"),
                    "VOLUME": st.column_config.NumberColumn("Volume", format="%,d"),
                    "CLOSE": st.column_config.NumberColumn("Price", format="%.2f"),
                    "IV": st.column_config.NumberColumn("IV%", format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True,
                height=400
            )
        else:
            st.warning("No options match your filters. Try adjusting your criteria.")
    except Exception as e:
        logger.error(f"Option screener error: {str(e)}")
        st.error("Error in option screener")

with tab3:  # Signal Generator
    try:
        st.header("Best Time to Buy - Signal Generation")

        # Signal rules
        st.subheader("Signal Rules Configuration")
        oi_change_threshold = st.number_input("OI Change Threshold", value=10000, key='signal_oi_thresh')
        price_trend_days = st.number_input("Price Trend Days", min_value=1, value=3, key='signal_trend_days')
        volume_multiplier = st.number_input("Volume Multiplier", min_value=1, value=2, key='signal_vol_mult')

        # Generate signals
        st.subheader("Bullish Signals (Potential Buy Opportunities)")
        bullish_rule1 = df[
            (df['CHG_IN_OI'] > oi_change_threshold) &
            (df['SETTLE_PR'] > df['SETTLE_PR'].shift(1)) &
            (df['SETTLE_PR'].shift(1) > df['SETTLE_PR'].shift(2))
        ].copy()

        bullish_rule2 = df[
            (df['CLOSE'] < df['CLOSE'].shift(1)) &
            (df['OPEN_INT'] > df['OPEN_INT'].shift(1)) &
            (df['VOLUME'] > df['VOLUME'].shift(1) * volume_multiplier)
        ].copy()

        bullish_signals = pd.concat([bullish_rule1, bullish_rule2]).drop_duplicates()

        st.dataframe(
            bullish_signals.sort_values('DATE', ascending=False).head(20),
            column_config={
                "DATE": "Date",
                "SYMBOL": "Symbol",
                "STRIKE_PR": "Strike",
                "OPTION_TYPE": "Type",
                "OPEN_INT": "OI",
                "CHG_IN_OI": "ΔOI",
                "VOLUME": "Volume",
                "CLOSE": "Price"
            },
            hide_index=True,
            use_container_width=True
        )

        # Bearish signals
        st.subheader("Bearish Signals (Potential Sell Opportunities)")
        bearish_rule1 = df[
            (df['CHG_IN_OI'] < -oi_change_threshold) &
            (df['SETTLE_PR'] < df['SETTLE_PR'].shift(1)) &
            (df['SETTLE_PR'].shift(1) < df['SETTLE_PR'].shift(2))
        ].copy()

        bearish_rule2 = df[
            (df['CLOSE'] > df['CLOSE'].shift(1)) &
            (df['OPEN_INT'] < df['OPEN_INT'].shift(1)) &
            (df['VOLUME'] > df['VOLUME'].shift(1) * volume_multiplier)
        ].copy()

        bearish_signals = pd.concat([bearish_rule1, bearish_rule2]).drop_duplicates()

        st.dataframe(
            bearish_signals.sort_values('DATE', ascending=False).head(20),
            column_config={
                "DATE": "Date",
                "SYMBOL": "Symbol",
                "STRIKE_PR": "Strike",
                "OPTION_TYPE": "Type",
                "OPEN_INT": "OI",
                "CHG_IN_OI": "ΔOI",
                "VOLUME": "Volume",
                "CLOSE": "Price"
            },
            hide_index=True,
            use_container_width=True
        )
    except Exception as e:
        logger.error(f"Signal generator error: {str(e)}")
        st.error("Error generating signals")

with tab4:  # Trend Prediction
    try:
        st.header("AI Trend Prediction")

        # Select symbol for prediction
        pred_symbol = st.selectbox("Select Symbol for Prediction", df['SYMBOL'].unique(), key='pred_symbol')
        pred_option_type = st.selectbox("Option Type", ['CE', 'PE'], key='pred_option_type')

        # Prepare data for model
        symbol_data = df[(df['SYMBOL'] == pred_symbol) & (df['OPTION_TYPE'] == pred_option_type)]
        symbol_data = symbol_data.sort_values(['STRIKE_PR', 'DATE'])

        # Feature engineering
        symbol_data['PRICE_CHANGE'] = symbol_data['CLOSE'].pct_change()
        symbol_data['OI_CHANGE_PCT'] = symbol_data['OPEN_INT'].pct_change()
        symbol_data['VOLUME_CHANGE_PCT'] = symbol_data['VOLUME'].pct_change()
        symbol_data['TARGET'] = (symbol_data['CLOSE'].shift(-1) > symbol_data['CLOSE']).astype(int)
        symbol_data = symbol_data.dropna()

        if symbol_data.empty:
            st.warning("Not enough data for prediction")
        else:
            # Select strike price
            selected_strike = st.selectbox("Select Strike Price", symbol_data['STRIKE_PR'].unique(), key='pred_strike')
            strike_data = symbol_data[symbol_data['STRIKE_PR'] == selected_strike]

            # Model selection
            model_type = st.radio("Select Model", ['LSTM', 'Random Forest'], key='model_type')

            if model_type == 'Random Forest':
                X = strike_data[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'OPEN_INT', 'CHG_IN_OI', 'VOLUME']]
                y = strike_data['TARGET']

                if len(X) > 10:  # Minimum samples required
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced estimators
                    model.fit(X_train, y_train)

                    # Evaluate
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    st.write(f"Model Accuracy: {accuracy:.2%}")

                    # Feature importance
                    st.subheader("Feature Importance")
                    importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig = px.bar(importance, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)

                    # Make prediction
                    if st.button("Predict Next Day Movement"):
                        last_data = X.iloc[-1:].values
                        prediction = model.predict(last_data)
                        proba = model.predict_proba(last_data)

                        if prediction[0] == 1:
                            st.success(f"Prediction: UP (Probability: {proba[0][1]:.2%})")
                        else:
                            st.error(f"Prediction: DOWN (Probability: {proba[0][0]:.2%})")
                else:
                    st.warning("Not enough data for Random Forest model")

            else:  # LSTM
                # Prepare data for LSTM
                sequence_length = min(5, len(strike_data) - 1)  # Adjusted sequence length
                features = ['CLOSE', 'OPEN_INT', 'VOLUME']

                if sequence_length > 1:
                    data = strike_data[features].values
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(data)

                    X, y = [], []
                    for i in range(len(scaled_data) - sequence_length):
                        X.append(scaled_data[i:i+sequence_length])
                        y.append(strike_data['TARGET'].iloc[i+sequence_length])

                    X, y = np.array(X), np.array(y)
                    
                    if len(X) > 1:  # Need at least 2 samples for train/test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Build simplified LSTM model
                        model = Sequential([
                            LSTM(32, input_shape=(sequence_length, len(features))),  # Reduced units
                            Dense(1, activation='sigmoid')
                        ])

                        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                        if st.button("Train LSTM Model"):
                            with st.spinner("Training LSTM model..."):
                                history = model.fit(
                                    X_train, y_train,
                                    epochs=10,  # Reduced epochs
                                    batch_size=16,  # Reduced batch size
                                    validation_data=(X_test, y_test),
                                    verbose=0
                                )

                                # Plot training history
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    y=history.history['accuracy'],
                                    name='Train Accuracy'
                                ))
                                fig.add_trace(go.Scatter(
                                    y=history.history['val_accuracy'],
                                    name='Validation Accuracy'
                                ))
                                fig.update_layout(title="Training History", xaxis_title="Epoch", yaxis_title="Accuracy")
                                st.plotly_chart(fig, use_container_width=True)

                                # Make prediction
                                last_sequence = scaled_data[-sequence_length:]
                                prediction = model.predict(last_sequence.reshape(1, sequence_length, len(features)))

                                if prediction[0][0] > 0.5:
                                    st.success(f"Prediction: UP (Probability: {prediction[0][0]:.2%})")
                                else:
                                    st.error(f"Prediction: DOWN (Probability: {1 - prediction[0][0]:.2%})")
                    else:
                        st.warning("Not enough data sequences for LSTM model")
                else:
                    st.warning("Not enough data for LSTM model")
    except Exception as e:
        logger.error(f"Trend prediction error: {str(e)}")
        st.error("Error in trend prediction")

with tab5:  # Strategy Assistant
    try:
        st.header("Expiry Strategy Assistant")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Near Expiry vs Far Expiry")
            current_date = pd.to_datetime('2023-01-10')  # Simulated current date
            expiries = sorted(df['DATE'].unique())
            expiries = [pd.to_datetime(d) for d in expiries if pd.to_datetime(d) > current_date]

            if len(expiries) > 0:
                near_expiry = min(expiries)
                far_expiry = expiries[1] if len(expiries) > 1 else near_expiry

                st.write(f"Current Date: {current_date.strftime('%Y-%m-%d')}")
                st.write(f"Near Expiry: {near_expiry.strftime('%Y-%m-%d')}")
                st.write(f"Far Expiry: {far_expiry.strftime('%Y-%m-%d')}")

                # Compare IV
                near_iv = df[df['DATE'] == near_expiry]['IV'].mean()
                far_iv = df[df['DATE'] == far_expiry]['IV'].mean()

                st.metric("Near Expiry Avg IV", f"{near_iv:.2f}%")
                st.metric("Far Expiry Avg IV", f"{far_iv:.2f}%")

                if near_iv > far_iv * 1.2:
                    st.warning("High IV in near expiry - Consider selling options")
                elif far_iv > near_iv * 1.2:
                    st.warning("High IV in far expiry - Consider calendar spreads")
                else:
                    st.info("IVs are relatively balanced")
            else:
                st.warning("No expiry data available")

        with col2:
            st.subheader("Strike Selection Recommendation")
            underlying_price = {
                'NIFTY': 18000,
                'BANKNIFTY': 40000,
                'RELIANCE': 2400
            }.get(selected_symbol, 100)

            strikes = sorted(df['STRIKE_PR'].unique())
            if len(strikes) > 0:
                atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
                st.write(f"Underlying Price: {underlying_price}")
                st.write(f"ATM Strike: {atm_strike}")

                # Analyze OI concentrations
                oi_data = df[
                    (df['SYMBOL'] == selected_symbol) &
                    (df['DATE'] == near_expiry) &
                    (df['OPTION_TYPE'] == 'CE')
                ].groupby('STRIKE_PR')['OPEN_INT'].sum().reset_index()

                if not oi_data.empty:
                    max_oi_strike = oi_data.loc[oi_data['OPEN_INT'].idxmax(), 'STRIKE_PR']
                    st.write(f"Max OI Strike (Call): {max_oi_strike}")

                    if max_oi_strike > atm_strike:
                        st.success("Bullish OI concentration - Consider call options")
                    else:
                        st.warning("Bearish OI concentration - Consider put options")
                else:
                    st.warning("No OI data available")
            else:
                st.warning("No strike prices available")
    except Exception as e:
        logger.error(f"Strategy assistant error: {str(e)}")
        st.error("Error in strategy assistant")

with tab6:  # Backtesting
    try:
        st.header("Backtesting Engine")

        # Backtest parameters
        col1, col2 = st.columns(2)

        with col1:
            backtest_symbol = st.selectbox("Symbol", df['SYMBOL'].unique(), key='backtest_symbol')
            backtest_option_type = st.selectbox("Option Type", ['CE', 'PE'], key='backtest_option_type')
            min_oi_backtest = st.number_input("Minimum OI", min_value=0, value=10000, key='min_oi')

        with col2:
            max_price_backtest = st.number_input("Maximum Price", min_value=0, value=100, key='max_price')
            hold_period = st.number_input("Hold Period (days)", min_value=1, value=1, key='hold_period')

        # Run backtest
        if st.button("Run Backtest"):
            backtest_data = df[
                (df['SYMBOL'] == backtest_symbol) &
                (df['OPTION_TYPE'] == backtest_option_type) &
                (df['OPEN_INT'] >= min_oi_backtest) &
                (df['CLOSE'] <= max_price_backtest)
            ].copy()

            if not backtest_data.empty:
                # Calculate returns
                backtest_data['NEXT_CLOSE'] = backtest_data.groupby('STRIKE_PR')['CLOSE'].shift(-hold_period)
                backtest_data = backtest_data.dropna()
                backtest_data['RETURN'] = (backtest_data['NEXT_CLOSE'] - backtest_data['CLOSE']) / backtest_data['CLOSE']

                # Summary stats
                total_trades = len(backtest_data)
                winning_trades = len(backtest_data[backtest_data['RETURN'] > 0])
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                avg_return = backtest_data['RETURN'].mean() if total_trades > 0 else 0

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Trades", total_trades)
                col2.metric("Win Rate", f"{win_rate:.2%}")
                col3.metric("Avg Return", f"{avg_return:.2%}")

                # Plot returns distribution
                fig = px.histogram(backtest_data, x='RETURN', nbins=20, title="Returns Distribution")
                st.plotly_chart(fig, use_container_width=True)

                # Show top trades
                st.subheader("Top Performing Trades")
                st.dataframe(
                    backtest_data.sort_values('RETURN', ascending=False).head(10),
                    column_config={
                        "DATE": "Entry Date",
                        "STRIKE_PR": "Strike",
                        "CLOSE": "Entry Price",
                        "NEXT_CLOSE": "Exit Price",
                        "RETURN": "Return",
                        "OPEN_INT": "OI"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.warning("No trades match your criteria")
    except Exception as e:
        logger.error(f"Backtesting error: {str(e)}")
        st.error("Error in backtesting")

# Add footer
st.markdown("---")
st.markdown("### AI Options Trading Platform v1.0")
st.markdown("Using advanced analytics and machine learning to identify options trading opportunities")