import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler  # Added this import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from prophet import Prophet
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="AI Options Trading Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load sample data (in a real app, this would come from a database or API)
@st.cache_data
def load_sample_data():
    # Generate synthetic options data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-06-30")
    symbols = ['NIFTY', 'BANKNIFTY', 'RELIANCE', 'HDFCBANK', 'TCS']
    strike_prices = {
        'NIFTY': np.arange(17000, 19000, 100),
        'BANKNIFTY': np.arange(38000, 42000, 200),
        'RELIANCE': np.arange(2200, 2600, 50),
        'HDFCBANK': np.arange(1400, 1700, 50),
        'TCS': np.arange(3000, 3500, 50)
    }

    data = []
    for date in dates:
        for symbol in symbols:
            for strike in strike_prices[symbol][:10]:  # Limit to 10 strikes per symbol
                for option_type in ['CE', 'PE']:
                    oi = np.random.randint(10000, 500000)
                    chg_in_oi = np.random.randint(-20000, 20000)
                    volume = np.random.randint(1000, 50000)
                    settle_pr = np.random.uniform(10, 200)
                    close = np.random.uniform(10, 200)
                    iv = np.random.uniform(10, 40)

                    data.append({
                        'DATE': date,
                        'SYMBOL': symbol,
                        'STRIKE_PR': strike,
                        'OPTION_TYPE': option_type,
                        'OPEN_INT': oi,
                        'CHG_IN_OI': chg_in_oi,
                        'VOLUME': volume,
                        'SETTLE_PR': settle_pr,
                        'CLOSE': close,
                        'IV': iv,
                        'OPEN': np.random.uniform(10, 200),
                        'HIGH': np.random.uniform(10, 200),
                        'LOW': np.random.uniform(10, 200)
                    })

    return pd.DataFrame(data)

df = load_sample_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_symbol = st.sidebar.selectbox("Symbol", df['SYMBOL'].unique())
selected_date = st.sidebar.selectbox("Date", sorted(df['DATE'].unique(), reverse=True))
selected_option_type = st.sidebar.radio("Option Type", ['CE', 'PE', 'All'], index=2)

# Filter data based on selections
filtered_df = df[(df['SYMBOL'] == selected_symbol) & (df['DATE'] == selected_date)]
if selected_option_type != 'All':
    filtered_df = filtered_df[filtered_df['OPTION_TYPE'] == selected_option_type]

# Main app
st.title("📊 AI-Powered Options Trading Platform")

# Tab layout
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard", "Option Screener", "Signal Generator",
    "Trend Prediction", "Strategy Assistant", "Backtesting"
])

with tab1:  # Dashboard
    st.header("Smart Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Open Interest Heatmap")
        heatmap_data = filtered_df.pivot_table(
            index='STRIKE_PR',
            columns='OPTION_TYPE',
            values='OPEN_INT',
            aggfunc='sum'
        ).fillna(0)
        fig = px.imshow(heatmap_data, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Volume vs Open Interest")
        # Use absolute values for size parameter
        fig = px.scatter(
            filtered_df,
            x='VOLUME',
            y='OPEN_INT',
            color='OPTION_TYPE',
            size=filtered_df['CHG_IN_OI'].abs(),  # Fixed: using absolute values
            hover_data=['STRIKE_PR'],
            size_max=20  # Added to control maximum bubble size
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("OI Change by Strike Price")
    fig = px.bar(
        filtered_df,
        x='STRIKE_PR',
        y='CHG_IN_OI',  # Can keep original values for bar chart
        color='OPTION_TYPE',
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:  # Option Screener
    st.header("AI-Powered Option Screener")

    # Screener filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_oi = st.number_input("Minimum Open Interest", min_value=0, value=10000)
        expiry_range = st.selectbox("Expiry Range", ["This Week", "Next Week", "This Month"])
    with col2:
        min_volume = st.number_input("Minimum Volume", min_value=0, value=1000)
        iv_filter = st.selectbox("IV Filter", ["All", "High IV (>30%)", "Low IV (<20%)"])
    with col3:
        oi_change = st.number_input("Minimum OI Change", min_value=0, value=5000)
        price_filter = st.selectbox("Price Filter", ["All", "Under 50", "50-100", "Over 100"])

    try:
        # Apply filters with error handling
        screened_df = df.copy()

        # Basic filters
        screened_df = screened_df[screened_df['OPEN_INT'] >= min_oi]
        screened_df = screened_df[screened_df['VOLUME'] >= min_volume]
        screened_df = screened_df[screened_df['CHG_IN_OI'].abs() >= oi_change]

        # IV filter
        if iv_filter == "High IV (>30%)":
            screened_df = screened_df[screened_df['IV'] > 30]
        elif iv_filter == "Low IV (<20%)":
            screened_df = screened_df[screened_df['IV'] < 20]

        # Price filter
        if price_filter == "Under 50":
            screened_df = screened_df[screened_df['CLOSE'] < 50]
        elif price_filter == "50-100":
            screened_df = screened_df[(screened_df['CLOSE'] >= 50) & (screened_df['CLOSE'] <= 100)]
        elif price_filter == "Over 100":
            screened_df = screened_df[screened_df['CLOSE'] > 100]

        # Expiry filter (placeholder - implement based on your data)
        if expiry_range == "This Week":
            pass  # Add your expiry date filtering logic
        elif expiry_range == "Next Week":
            pass
        elif expiry_range == "This Month":
            pass

        # NLP search
        nlp_search = st.text_input("NLP Search (e.g., 'Show me NIFTY calls with OI > 50K')")
        if nlp_search:
            try:
                if "call" in nlp_search.lower():
                    screened_df = screened_df[screened_df['OPTION_TYPE'] == 'CE']
                if "put" in nlp_search.lower():
                    screened_df = screened_df[screened_df['OPTION_TYPE'] == 'PE']
                if "oi >" in nlp_search.lower():
                    oi_text = nlp_search.lower().split("oi >")[1].split()[0]
                    oi_val = int(oi_text.replace("k", "000").replace(",", ""))
                    screened_df = screened_df[screened_df['OPEN_INT'] > oi_val]
            except Exception as e:
                st.warning(f"Couldn't process NLP query: {str(e)}")

        # Visualization - OI vs Strike Price
        if not screened_df.empty:
            fig = go.Figure()

            # Add calls and puts separately
            for opt_type, color in [('CE', 'green'), ('PE', 'red')]:
                type_df = screened_df[screened_df['OPTION_TYPE'] == opt_type]
                if not type_df.empty:
                    fig.add_trace(go.Bar(
                        x=type_df['STRIKE_PR'],
                        y=type_df['OPEN_INT'],
                        name=f'{opt_type} OI',
                        marker_color=color,
                        opacity=0.7
                    ))

            fig.update_layout(
                title='Open Interest by Strike Price',
                xaxis_title='Strike Price',
                yaxis_title='Open Interest',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        # Display results with improved formatting
        if not screened_df.empty:
            st.dataframe(
                screened_df.sort_values('CHG_IN_OI', ascending=False).head(100),
                column_config={
                    "DATE": st.column_config.DateColumn("Date"),
                    "SYMBOL": "Symbol",
                    "STRIKE_PR": st.column_config.NumberColumn("Strike", format="%.2f"),
                    "OPTION_TYPE": st.column_config.TextColumn("Type"),
                    "OPEN_INT": st.column_config.NumberColumn("OI", format="%,d"),
                    "CHG_IN_OI": st.column_config.NumberColumn("ΔOI", format="%,d"),
                    "VOLUME": st.column_config.NumberColumn("Volume", format="%,d"),
                    "CLOSE": st.column_config.NumberColumn("Price", format="%.2f"),
                    "IV": st.column_config.NumberColumn("IV%", format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
        else:
            st.warning("No options match your filters. Try adjusting your criteria.")

    except Exception as e:
        st.error(f"An error occurred while filtering options: {str(e)}")
        logger.error(f"Option screener error: {str(e)}", exc_info=True)

with tab3:  # Signal Generator
    st.header("Best Time to Buy - Signal Generation")

    # Signal rules
    st.subheader("Signal Rules Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        oi_change_threshold = st.number_input("OI Change Threshold", value=10000)
    with col2:
        price_trend_days = st.number_input("Price Trend Days", min_value=1, value=3)
    with col3:
        volume_multiplier = st.number_input("Volume Multiplier", min_value=1, value=2)

    # Generate signals
    st.subheader("Bullish Signals (Potential Buy Opportunities)")

    # Rule 1: OI increase with price increase
    bullish_rule1 = df[
        (df['CHG_IN_OI'] > oi_change_threshold) &
        (df['SETTLE_PR'] > df['SETTLE_PR'].shift(1)) &
        (df['SETTLE_PR'].shift(1) > df['SETTLE_PR'].shift(2))
    ]

    # Rule 2: Price down but OI up (accumulation)
    bullish_rule2 = df[
        (df['CLOSE'] < df['CLOSE'].shift(1)) &
        (df['OPEN_INT'] > df['OPEN_INT'].shift(1)) &
        (df['VOLUME'] > df['VOLUME'].shift(1) * volume_multiplier)
    ]

    # Combine signals
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

    # Rule 1: OI decrease with price decrease
    bearish_rule1 = df[
        (df['CHG_IN_OI'] < -oi_change_threshold) &
        (df['SETTLE_PR'] < df['SETTLE_PR'].shift(1)) &
        (df['SETTLE_PR'].shift(1) < df['SETTLE_PR'].shift(2))
    ]

    # Rule 2: Price up but OI down (distribution)
    bearish_rule2 = df[
        (df['CLOSE'] > df['CLOSE'].shift(1)) &
        (df['OPEN_INT'] < df['OPEN_INT'].shift(1)) &
        (df['VOLUME'] > df['VOLUME'].shift(1) * volume_multiplier)
    ]

    # Combine signals
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

with tab4:  # Trend Prediction
    st.header("AI Trend Prediction")

    # Select symbol for prediction
    pred_symbol = st.selectbox("Select Symbol for Prediction", df['SYMBOL'].unique())
    pred_option_type = st.selectbox("Option Type", ['CE', 'PE'])

    # Prepare data for model
    symbol_data = df[(df['SYMBOL'] == pred_symbol) & (df['OPTION_TYPE'] == pred_option_type)]
    symbol_data = symbol_data.sort_values(['STRIKE_PR', 'DATE'])

    # Feature engineering
    symbol_data['PRICE_CHANGE'] = symbol_data['CLOSE'].pct_change()
    symbol_data['OI_CHANGE_PCT'] = symbol_data['OPEN_INT'].pct_change()
    symbol_data['VOLUME_CHANGE_PCT'] = symbol_data['VOLUME'].pct_change()
    symbol_data['TARGET'] = (symbol_data['CLOSE'].shift(-1) > symbol_data['CLOSE']).astype(int)
    symbol_data = symbol_data.dropna()

    # Select strike price
    selected_strike = st.selectbox("Select Strike Price", symbol_data['STRIKE_PR'].unique())
    strike_data = symbol_data[symbol_data['STRIKE_PR'] == selected_strike]

    # Model selection
    model_type = st.radio("Select Model", ['LSTM', 'Random Forest'])

    if model_type == 'Random Forest':
        # Random Forest Model
        X = strike_data[['OPEN', 'HIGH', 'LOW', 'CLOSE', 'OPEN_INT', 'CHG_IN_OI', 'VOLUME']]
        y = strike_data['TARGET']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
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

    else:  # LSTM
        # Prepare data for LSTM
        sequence_length = 10
        features = ['CLOSE', 'OPEN_INT', 'VOLUME']

        data = strike_data[features].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(strike_data['TARGET'].iloc[i+sequence_length])

        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, len(features))),
            LSTM(50),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train model (simplified for demo)
        if st.button("Train LSTM Model"):
            with st.spinner("Training LSTM model..."):
                history = model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=32,
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

                # Evaluate
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                st.write(f"Model Accuracy: {accuracy:.2%}")

                # Make prediction
                last_sequence = scaled_data[-sequence_length:]
                prediction = model.predict(last_sequence.reshape(1, sequence_length, len(features)))

                if prediction[0][0] > 0.5:
                    st.success(f"Prediction: UP (Probability: {prediction[0][0]:.2%})")
                else:
                    st.error(f"Prediction: DOWN (Probability: {1 - prediction[0][0]:.2%})")

with tab5:  # Strategy Assistant
    st.header("Expiry Strategy Assistant")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Near Expiry vs Far Expiry")
        current_date = pd.to_datetime('2023-06-15')  # Simulated current date
        expiries = df['DATE'].unique()
        expiries = sorted([pd.to_datetime(d) for d in expiries if pd.to_datetime(d) > current_date])

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

    with col2:
        st.subheader("Strike Selection Recommendation")

        # Get ATM strike
        underlying_price = {
            'NIFTY': 18000,
            'BANKNIFTY': 40000,
            'RELIANCE': 2400,
            'HDFCBANK': 1500,
            'TCS': 3200
        }.get(selected_symbol, 100)

        strikes = sorted(df['STRIKE_PR'].unique())
        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))

        st.write(f"Underlying Price: {underlying_price}")
        st.write(f"ATM Strike: {atm_strike}")

        # Analyze OI concentrations
        oi_data = df[
            (df['SYMBOL'] == selected_symbol) &
            (df['DATE'] == near_expiry) &
            (df['OPTION_TYPE'] == 'CE')
        ].groupby('STRIKE_PR')['OPEN_INT'].sum().reset_index()

        max_oi_strike = oi_data.loc[oi_data['OPEN_INT'].idxmax(), 'STRIKE_PR']

        st.write(f"Max OI Strike (Call): {max_oi_strike}")

        if max_oi_strike > atm_strike:
            st.success("Bullish OI concentration - Consider call options")
        else:
            st.warning("Bearish OI concentration - Consider put options")

with tab6:  # Backtesting
    st.header("Backtesting Engine")

    # Backtest parameters
    col1, col2 = st.columns(2)

    with col1:
        backtest_symbol = st.selectbox("Symbol", df['SYMBOL'].unique(), key='backtest_symbol')
        backtest_option_type = st.selectbox("Option Type", ['CE', 'PE'], key='backtest_option_type')
        min_oi_backtest = st.number_input("Minimum OI", min_value=0, value=10000, key='min_oi')

    with col2:
        max_price_backtest = st.number_input("Maximum Price", min_value=0, value=100, key='max_price')
        days_to_expiry = st.number_input("Days to Expiry Max", min_value=1, value=7, key='days_to_expiry')
        hold_period = st.number_input("Hold Period (days)", min_value=1, value=1, key='hold_period')

    # Run backtest
    if st.button("Run Backtest"):
        # Filter data based on parameters
        backtest_data = df[
            (df['SYMBOL'] == backtest_symbol) &
            (df['OPTION_TYPE'] == backtest_option_type) &
            (df['OPEN_INT'] >= min_oi_backtest) &
            (df['CLOSE'] <= max_price_backtest)
        ].copy()

        # Calculate days to expiry (simplified)
        backtest_data['DAYS_TO_EXPIRY'] = (backtest_data['DATE'].shift(-hold_period) - backtest_data['DATE']).dt.days
        backtest_data = backtest_data[backtest_data['DAYS_TO_EXPIRY'] <= days_to_expiry]

        if backtest_data.empty:
            st.warning("No trades match your criteria")
        else:
            # Calculate returns
            backtest_data['NEXT_CLOSE'] = backtest_data.groupby('STRIKE_PR')['CLOSE'].shift(-hold_period)
            backtest_data = backtest_data.dropna()
            backtest_data['RETURN'] = (backtest_data['NEXT_CLOSE'] - backtest_data['CLOSE']) / backtest_data['CLOSE']

            # Summary stats
            total_trades = len(backtest_data)
            winning_trades = len(backtest_data[backtest_data['RETURN'] > 0])
            win_rate = winning_trades / total_trades
            avg_return = backtest_data['RETURN'].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", total_trades)
            col2.metric("Win Rate", f"{win_rate:.2%}")
            col3.metric("Avg Return", f"{avg_return:.2%}")

            # Plot returns distribution
            fig = px.histogram(backtest_data, x='RETURN', nbins=20, title="Returns Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Show top 10 trades
            st.subheader("Top Performing Trades")
            st.dataframe(
                backtest_data.sort_values('RETURN', ascending=False).head(10),
                column_config={
                    "DATE": "Entry Date",
                    "STRIKE_PR": "Strike",
                    "CLOSE": "Entry Price",
                    "NEXT_CLOSE": "Exit Price",
                    "RETURN": "Return",
                    "OPEN_INT": "OI",
                    "DAYS_TO_EXPIRY": "Days to Expiry"
                },
                hide_index=True,
                use_container_width=True
            )

# Add footer
st.markdown("---")
st.markdown("### AI Options Trading Platform v1.0")
st.markdown("Using advanced analytics and machine learning to identify options trading opportunities")