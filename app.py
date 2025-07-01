import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
import pickle
import os
import warnings
from collections import deque
import random
import time
import shap
# Fix for numpy NaN import issues
import numpy as np
np.NaN = np.nan  # Monkey patch for older libraries like pandas_ta

import pandas_ta as ta  # Now safe to import


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.optimizers import Adam
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Alpha Dashboard")

# Title
st.title("🚀 Enhanced Explainable Alpha Dashboard")

# Initialize APIs (hardcoded inline)
FRED_API_KEY = "07083ad37b113222971a5b5271092294"  # FRED API key
ALPHA_VANTAGE_KEY = "D94XRATFMNGOTYRO"           # Alpha Vantage API key
NEWSAPI_KEY = "f75338e6e04d484bbb532826f3a02154" # NewsAPI key

# Sidebar for user inputs
with st.sidebar:
    st.header("🔧 Configuration")
    selected_date = st.date_input("Select date", datetime.today())
    num_stocks = st.slider("Number of top stocks to show", 1, 50, 10)
    risk_tolerance = st.select_slider("Risk tolerance", options=["Low", "Medium", "High"], value="Medium")

    # Stock selection
    default_stocks = ['SPY', 'QQQ', 'DIA', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
                     'TSLA', 'NVDA', 'JPM', 'V', 'GS', 'BAC', 'WMT', 'TGT', 'COST',
                     'XOM', 'CVX', 'COP', 'JNJ', 'PFE', 'UNH']
    selected_stocks = st.multiselect("Select stocks to analyze", default_stocks, default=default_stocks[:10])

    st.markdown("---")
    st.subheader("📉 Alpha Decay Parameters")
    tau = st.slider("Decay rate (τ)", 1.0, 10.0, 3.0, 0.1)

    st.markdown("---")
    st.subheader("🤖 RL Portfolio Allocation")
    rl_training_epochs = st.slider("RL Training Epochs", 1, 100, 20)

    st.markdown("---")
    st.subheader("🔔 Event Triggers")
    earnings_trigger = st.checkbox("Earnings Report Model", True)
    macro_trigger = st.checkbox("Macro News Model", True)
    technical_trigger = st.checkbox("Technical Pattern Model", True)
    news_sentiment_trigger = st.checkbox("News Sentiment Analysis", True)

    st.markdown("---")
    st.subheader("📊 Data Sources")
    use_live_data = st.checkbox("Use live market data (requires API keys)", False)
    historical_years = st.slider("Years of historical data", 1, 5, 2)

# Pattern Recognizer class
class PatternRecognizer:
    def __init__(self, price_data, technical_data):
        self.price_data = price_data
        self.technical_data = technical_data
        self.models = {}
        self.scalers = {}
        self.pattern_threshold = 0.7

    def train_models(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        for stock in self.price_data.columns:
            if stock not in self.technical_data:
                continue
            df = self.technical_data[stock].copy()
            df['returns'] = self.price_data[stock].pct_change()
            df['label'] = 0
            threshold = df['returns'].abs().quantile(0.9)
            df.loc[df['returns'] > threshold, 'label'] = 1
            df.loc[df['returns'] < -threshold, 'label'] = -1

            features = df[[
                'sma_20', 'sma_50', 'sma_200', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
                'rsi_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
            ]].dropna()

            for lag in [1, 2, 3, 5, 10]:
                features[f'ret_lag_{lag}'] = df['returns'].shift(lag)

            features = features.dropna()
            labels = df.loc[features.index, 'label']

            if len(features) < 100 or len(labels.unique()) < 2:
                st.warning(f"Insufficient data for {stock} pattern recognition")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)

            self.models[stock] = model
            self.scalers[stock] = scaler

    def detect_patterns(self, stock):
        if stock not in self.models:
            return None
        df = self.technical_data[stock].iloc[-30:].copy()
        df['returns'] = self.price_data[stock].pct_change().iloc[-30:]

        features = pd.DataFrame({
            'sma_20': df['sma_20'].iloc[-1],
            'sma_50': df['sma_50'].iloc[-1],
            'sma_200': df['sma_200'].iloc[-1],
            'BBL_20_2.0': df['BBL_20_2.0'].iloc[-1],
            'BBM_20_2.0': df['BBM_20_2.0'].iloc[-1],
            'BBU_20_2.0': df['BBU_20_2.0'].iloc[-1],
            'rsi_14': df['rsi_14'].iloc[-1],
            'MACD_12_26_9': df['MACD_12_26_9'].iloc[-1],
            'MACDh_12_26_9': df['MACDh_12_26_9'].iloc[-1],
            'MACDs_12_26_9': df['MACDs_12_26_9'].iloc[-1]
        }, index=[0])

        for lag in [1, 2, 3, 5, 10]:
            features[f'ret_lag_{lag}'] = df['returns'].iloc[-lag-1]

        scaler = self.scalers[stock]
        features_scaled = scaler.transform(features)
        prediction = self.models[stock].predict(features_scaled)[0]

        if prediction > self.pattern_threshold:
            return {
                'pattern': 'bullish',
                'confidence': prediction,
                'features': features.iloc[0].to_dict(),
                'reason': self.explain_prediction(stock, features_scaled[0])
            }
        elif prediction < -self.pattern_threshold:
            return {
                'pattern': 'bearish',
                'confidence': abs(prediction),
                'features': features.iloc[0].to_dict(),
                'reason': self.explain_prediction(stock, features_scaled[0])
            }
        else:
            return {
                'pattern': 'neutral',
                'confidence': 0,
                'features': None,
                'reason': "No strong pattern detected"
            }

    def explain_prediction(self, stock, features_scaled):
        model = self.models[stock]
        importances = model.feature_importances_
        feature_names = [
            'SMA 20', 'SMA 50', 'SMA 200', 'BB Lower', 'BB Middle', 'BB Upper',
            'RSI 14', 'MACD', 'MACD Hist', 'MACD Signal',
            'Ret Lag 1', 'Ret Lag 2', 'Ret Lag 3', 'Ret Lag 5', 'Ret Lag 10'
        ]
        top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:3]
        explanation = []
        for feature, importance in top_features:
            idx = feature_names.index(feature)
            value = features_scaled[idx] * self.scalers[stock].scale_[idx] + self.scalers[stock].mean_[idx]
            if feature.startswith('SMA'):
                if '20' in feature:
                    explanation.append(f"Short-term trend ({value:.2f})")
                elif '50' in feature:
                    explanation.append(f"Medium-term trend ({value:.2f})")
                else:
                    explanation.append(f"Long-term trend ({value:.2f})")
            elif feature.startswith('BB'):
                explanation.append(f"Bollinger Band position ({value:.2f})")
            elif feature == 'RSI 14':
                explanation.append(f"RSI at {value:.2f}" +
                                (" (Overbought)" if value > 70 else " (Oversold)" if value < 30 else ""))
            elif feature.startswith('MACD'):
                explanation.append(f"MACD momentum ({value:.2f})")
            elif feature.startswith('Ret Lag'):
                explanation.append(f"Recent return pattern ({value:.2f})")
        return " + ".join(explanation)

# Enhanced RL Agent
class EnhancedRLAgent:
    def __init__(self, state_size, action_size, window_size=10):
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        price_input = Input((self.window_size, self.action_size))
        lstm_out = LSTM(32)(price_input)            # ↓ half the units
        tech_input = Input((self.state_size - self.window_size*self.action_size,))
        dense_tech = Dense(16, activation='relu')(tech_input)  # ↓ half the size
        merged = concatenate([lstm_out, dense_tech])
        x = Dense(32, activation='relu')(merged)
        x = Dense(32, activation='relu')(x)
        output = Dense(self.action_size, activation='softmax')(x)
        model = Model([price_input, tech_input], output)
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.rand(self.action_size)

        price_seq = state[:self.window_size * self.action_size]      # unchanged
        tech_features = state[self.window_size * self.action_size:]  # unchanged

        act_values = self.model.predict([price_seq.reshape(1, self.window_size, self.action_size),
                                         tech_features.reshape(1, -1)],
                                        verbose=0)[0]
        # sanitize:
        act_values = np.nan_to_num(act_values, nan=1.0/self.action_size, posinf=1.0/self.action_size, neginf=1.0/self.action_size)
        # renormalize:
        act_values = np.clip(act_values, 0, None)
        if act_values.sum() == 0:
            act_values = np.ones_like(act_values) / self.action_size
        else:
            act_values = act_values / act_values.sum()
        return act_values


    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            price_seq = state[:self.window_size * self.action_size].reshape(1, self.window_size, self.action_size)
            tech_features = state[self.window_size * self.action_size:].reshape(1, -1)
            next_price_seq = next_state[:self.window_size * self.action_size].reshape(1, self.window_size, self.action_size)
            next_tech_features = next_state[self.window_size * self.action_size:].reshape(1, -1)
            target = reward
            if not done:
                try:
                    target = reward + self.gamma * np.amax(
                        self.model.predict([next_price_seq, next_tech_features], verbose=0)[0])
                except Exception as e:
                    print(f"Error in predict: {str(e)}")
                    continue
            target_f = self.model.predict([price_seq, tech_features], verbose=0)
            target_f[0] = action * target
            self.model.fit([price_seq, tech_features], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Enhanced Portfolio Environment
class EnhancedPortfolioEnv:
    def __init__(self, stocks, prices, technicals, macro, sentiment, window_size=10):
        self.stocks = stocks
        self.prices = prices
        self.technicals = technicals
        self.macro = macro
        self.sentiment = sentiment
        self.window_size = window_size
        self.current_step = window_size
        self.max_steps = len(prices) - 2
        self.n_stocks = len(stocks)

    def reset(self):
        self.current_step = self.window_size
        return self._get_state()

    def _get_state(self):
        price_seq = []
        for i in range(self.window_size):
            idx = self.current_step - self.window_size + i
            returns = (self.prices.iloc[idx] / self.prices.iloc[idx-1] - 1).values if idx > 0 else np.zeros(self.n_stocks)
            price_seq.append(returns)
        price_seq = np.array(price_seq).flatten()
        tech_features = []
        for stock in self.stocks:
            tech = self.technicals[stock].iloc[self.current_step]
            tech_features.extend([
                tech['sma_20'], tech['sma_50'], tech['sma_200'],
                tech['BBL_20_2.0'], tech['BBM_20_2.0'], tech['BBU_20_2.0'],
                tech['rsi_14'], tech['MACD_12_26_9'], tech['MACDh_12_26_9'], tech['MACDs_12_26_9']
            ])
        macro_features = self.macro.iloc[self.current_step].fillna(0).values
        sentiment_features = []
        for stock in self.stocks:
            sent = self.sentiment[stock].iloc[self.current_step]
            sentiment_features.extend([sent['polarity'], sent['subjectivity'], sent['vader_compound']])
        return np.concatenate([price_seq, tech_features, macro_features, sentiment_features])

    def step(self, action):
        action = np.clip(action, 0, 1)
        action = action / (np.sum(action) + 1e-10)
        current_prices = self.prices.iloc[self.current_step]
        next_prices = self.prices.iloc[self.current_step + 1]
        returns = (next_prices / current_prices - 1).values
        portfolio_return = np.sum(returns * action)
        self.current_step += 1
        done = self.current_step >= self.max_steps
        risk = np.sqrt(np.sum((returns - portfolio_return)**2 * action))
        sharpe_ratio = portfolio_return / (risk + 1e-10)
        reward = sharpe_ratio
        return self._get_state(), reward, done, {'return': portfolio_return, 'risk': risk}

# Recommendation Engine
class RecommendationEngine:
    def __init__(self, price_data, technical_data, sentiment_data, pattern_recognizer):
        self.price_data = price_data
        self.technical_data = technical_data
        self.sentiment_data = sentiment_data
        self.pattern_recognizer = pattern_recognizer

    def generate_recommendations(self, stock):
        recommendations = []
        confidence = 0

        pattern = self.pattern_recognizer.detect_patterns(stock)
        if pattern['pattern'] != 'neutral':
            recommendations.append({
                'type': 'technical',
                'signal': pattern['pattern'],
                'confidence': pattern['confidence'],
                'reason': pattern['reason']
            })
            confidence += pattern['confidence'] * 0.6

        sentiment = self.sentiment_data[stock].iloc[-1]
        if sentiment['vader_compound'] > 0.5:
            recommendations.append({
                'type': 'sentiment',
                'signal': 'bullish',
                'confidence': sentiment['vader_compound'],
                'reason': "Strong positive sentiment in news"
            })
            confidence += sentiment['vader_compound'] * 0.3
        elif sentiment['vader_compound'] < -0.5:
            recommendations.append({
                'type': 'sentiment',
                'signal': 'bearish',
                'confidence': abs(sentiment['vader_compound']),
                'reason': "Strong negative sentiment in news"
            })
            confidence += abs(sentiment['vader_compound']) * 0.3

        rsi = self.technical_data[stock]['rsi_14'].iloc[-1]
        if rsi > 70:
            recommendations.append({
                'type': 'technical',
                'signal': 'overbought',
                'confidence': (rsi - 70) / 30,
                'reason': f"RSI {rsi:.1f} indicates overbought condition"
            })
            confidence += ((rsi - 70) / 30) * 0.4
        elif rsi < 30:
            recommendations.append({
                'type': 'technical',
                'signal': 'oversold',
                'confidence': (30 - rsi) / 30,
                'reason': f"RSI {rsi:.1f} indicates oversold condition"
            })
            confidence += ((30 - rsi) / 30) * 0.4

        price = self.price_data[stock].iloc[-1]
        sma_20 = self.technical_data[stock]['sma_20'].iloc[-1]
        sma_50 = self.technical_data[stock]['sma_50'].iloc[-1]

        if price > sma_20 > sma_50:
            recommendations.append({
                'type': 'technical',
                'signal': 'bullish',
                'confidence': 0.7,
                'reason': "Price above short and medium-term moving averages"
            })
            confidence += 0.7 * 0.3
        elif price < sma_20 < sma_50:
            recommendations.append({
                'type': 'technical',
                'signal': 'bearish',
                'confidence': 0.7,
                'reason': "Price below short and medium-term moving averages"
            })
            confidence += 0.7 * 0.3

        if not recommendations:
            return {
                'recommendation': 'hold',
                'confidence': 0,
                'details': "No strong signals detected"
            }

        confidence = min(1.0, confidence / len(recommendations))
        bull_count = sum(1 for r in recommendations if r['signal'] in ['bullish', 'oversold'])
        bear_count = sum(1 for r in recommendations if r['signal'] in ['bearish', 'overbought'])

        if bull_count > bear_count:
            return {
                'recommendation': 'buy',
                'confidence': confidence,
                'details': recommendations
            }
        elif bear_count > bull_count:
            return {
                'recommendation': 'sell',
                'confidence': confidence,
                'details': recommendations
            }
        else:
            return {
                'recommendation': 'hold',
                'confidence': confidence,
                'details': recommendations
            }

# Data loading functions
@st.cache_data
def load_market_data(stocks, use_live_data=False, years=2):
    import yfinance as yf
    from fredapi import Fred

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*years)

    price_data = get_historical_prices(stocks, start_date, end_date, use_live_data)
    st.write("Price Data Shape:", price_data.shape)  # Debug
    st.write("Price Data Columns:", price_data.columns)  # Debug
    tech_data = calculate_technical_indicators(price_data)
    fred = Fred(api_key=FRED_API_KEY)
    macro_data = get_macroeconomic_data(fred, start_date, end_date)

    if use_live_data:
        news_sentiment = get_news_sentiment(stocks)
    else:
        news_sentiment = generate_mock_sentiment(stocks, start_date, end_date)

    return {
        'prices': price_data,
        'technicals': tech_data,
        'macro': macro_data,
        'sentiment': news_sentiment,
        'stocks': stocks
    }

def get_historical_prices(stocks, start_date, end_date, use_live=False):
    import yfinance as yf
    from alpha_vantage.timeseries import TimeSeries

    # Validate stock symbols
    valid_stocks = [stock for stock in stocks if isinstance(stock, str) and stock]
    if not valid_stocks:
        st.error("No valid stock symbols provided. Using mock data.")
        return generate_mock_prices(stocks, (end_date - start_date).days)

    alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_KEY, output_format='pandas')

    if use_live and ALPHA_VANTAGE_KEY != "demo":
        try:
            prices = {}
            for stock in valid_stocks:
                try:
                    data, _ = alpha_vantage.get_daily_adjusted(
                        symbol=stock,
                        outputsize='full' if (end_date - start_date).days > 365 else 'compact'
                    )
                    data = data.loc[start_date:end_date]
                    prices[stock] = data['5. adjusted close']
                    time.sleep(12)  # Respect Alpha Vantage rate limit (5 calls/min)
                except Exception as e:
                    st.warning(f"Alpha Vantage error for {stock}: {str(e)}, falling back to Yahoo Finance")
                    continue
            if prices:
                price_df = pd.DataFrame(prices).sort_index().ffill().bfill()
                st.write("Alpha Vantage Price Data Columns:", price_df.columns)  # Debug
                return price_df
        except Exception as e:
            st.warning(f"Alpha Vantage API error: {str(e)}, falling back to Yahoo Finance")

    # Yahoo Finance fallback
    try:
        data = yf.download(valid_stocks, start=start_date, end=end_date, group_by='ticker')
        if len(valid_stocks) == 1:
            prices = data['Adj Close'].to_frame(name=valid_stocks[0])
        else:
            # Handle multi-level column index
            prices = pd.DataFrame()
            for stock in valid_stocks:
                if stock in data:
                    if isinstance(data[stock], pd.DataFrame) and 'Adj Close' in data[stock]:
                        prices[stock] = data[stock]['Adj Close']
                    else:
                        st.warning(f"No 'Adj Close' data for {stock} from Yahoo Finance")
                        prices[stock] = pd.Series(index=data.index)  # Empty series for failed stock
        if prices.empty or prices.isna().all().all():
            st.error("No valid price data retrieved from Yahoo Finance. Using mock data.")
            return generate_mock_prices(valid_stocks, (end_date - start_date).days)
        price_df = prices.ffill().bfill()
        st.write("Yahoo Finance Price Data Columns:", price_df.columns)  # Debug
        return price_df
    except Exception as e:
        st.error(f"Error downloading price data from Yahoo Finance: {str(e)}. Using mock data.")
        return generate_mock_prices(valid_stocks, (end_date - start_date).days)
def calculate_technical_indicators(price_data):
    import pandas_ta as ta
    tech_data = {}
    if price_data.empty or not price_data.columns.any():
        st.error("No valid price data available for technical indicators")
        return tech_data
    for stock in price_data.columns:
        df = pd.DataFrame(index=price_data.index)
        df['price'] = price_data[stock]
        if df['price'].isna().all():
            st.warning(f"No valid price data for {stock}")
            continue
        df['sma_20'] = ta.sma(df['price'], length=20)
        df['sma_50'] = ta.sma(df['price'], length=50)
        df['sma_200'] = ta.sma(df['price'], length=200)
        bb = ta.bbands(df['price'], length=20)
        df = pd.concat([df, bb], axis=1)
        df['rsi_14'] = ta.rsi(df['price'], length=14)
        macd = ta.macd(df['price'])
        df = pd.concat([df, macd], axis=1)
        tech_data[stock] = df
    return tech_data

def get_macroeconomic_data(fred, start_date, end_date):
    macro_indicators = {
        'DGS10': '10-Year Treasury Yield',
        'SP500': 'S&P 500',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'CPI Inflation',
        'FEDFUNDS': 'Fed Funds Rate'
    }
    macro_data = pd.DataFrame(index=pd.date_range(start_date, end_date))
    for code, name in macro_indicators.items():
        try:
            series = fred.get_series(code, start_date, end_date)
            macro_data[name] = series
        except:
            st.warning(f"Could not fetch {name} data")
    return macro_data.ffill().bfill()

@st.cache_data(ttl=3600)
def get_news_sentiment(stocks):
    import requests
    from bs4 import BeautifulSoup
    from textblob import TextBlob
    from nltk.sentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    sentiments = {}
    for stock in stocks:
        try:
            if NEWSAPI_KEY == "demo":
                sentiment = np.random.normal(0.5, 0.2)
                sentiments[stock] = {'polarity': sentiment, 'subjectivity': np.random.normal(0.5, 0.1)}
                continue

            url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={NEWSAPI_KEY}"
            response = requests.get(url)
            articles = response.json().get('articles', [])

            polarities = []
            subjectivities = []
            for article in articles[:10]:
                text = article['title'] + ' ' + article.get('description', '')
                analysis = TextBlob(text)
                polarities.append(analysis.sentiment.polarity)
                subjectivities.append(analysis.sentiment.subjectivity)

            if polarities:
                sentiments[stock] = {
                    'polarity': np.mean(polarities),
                    'subjectivity': np.mean(subjectivities),
                    'vader': sia.polarity_scores(' '.join([a['title'] for a in articles[:5]]))
                }
            else:
                sentiments[stock] = {'polarity': 0, 'subjectivity': 0, 'vader': {'compound': 0}}
        except Exception as e:
            st.warning(f"Could not fetch news for {stock}: {str(e)}")
            sentiments[stock] = {'polarity': 0, 'subjectivity': 0, 'vader': {'compound': 0}}
    return sentiments

def generate_mock_sentiment(stocks, start_date, end_date):
    dates = pd.date_range(start_date, end_date)
    sentiment_data = {}
    for stock in stocks:
        df = pd.DataFrame(index=dates)
        df['polarity'] = np.clip(np.random.normal(0.5, 0.2, len(dates)), 0, 1)
        df['subjectivity'] = np.clip(np.random.normal(0.5, 0.1, len(dates)), 0, 1)
        df['vader_compound'] = np.clip(np.random.normal(0.5, 0.3, len(dates)), -1, 1)
        sentiment_data[stock] = df
    return sentiment_data

def generate_mock_prices(stocks, days):
    dates = pd.date_range(end=datetime.today(), periods=days)
    prices = {}
    for stock in stocks:
        prices[stock] = np.random.normal(100, 20, days)
    return pd.DataFrame(prices, index=dates)

# SHAP explainer setup
@st.cache_resource
def init_shap_explainer(df, technical_data, sentiment_data):
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X = pd.DataFrame()
    for idx, row in df.iterrows():
        features = {}
        tech = technical_data[row['Stock']].iloc[-1]
        sent = sentiment_data[row['Stock']].iloc[-1]
        features.update({
            'sma_20': tech['sma_20'],
            'sma_50': tech['sma_50'],
            'sma_200': tech['sma_200'],
            'rsi': tech['rsi_14'],
            'macd': tech['MACD_12_26_9'],
            'bb_width': (tech['BBU_20_2.0'] - tech['BBL_20_2.0']) / tech['BBM_20_2.0'],
            'sentiment': sent['vader_compound'],
            'subjectivity': sent['subjectivity'],
            '1d_return': row['1D Return'],
            '7d_return': row['7D Return'],
            'volatility': price_data[row['Stock']].pct_change().std()
        })
        X = pd.concat([X, pd.DataFrame(features, index=[idx])], axis=0)

    y = df['Confidence']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    return explainer, model

# ... (previous imports and class definitions remain unchanged until the data loading section)

# Load data (assumed to be defined in your original code)
market_data = load_market_data(selected_stocks, use_live_data, historical_years)
price_data = market_data['prices']
technical_data = market_data['technicals']
macro_data = market_data['macro']
sentiment_data = market_data['sentiment']
stocks = market_data['stocks']

# Initialize df to avoid NameError
df = pd.DataFrame()

# Define create_recommendation_df function before calling it
@st.cache_data
def create_recommendation_df(stocks, _recommendation_engine):
    data = []
    for stock in stocks:
        try:
            rec = _recommendation_engine.generate_recommendations(stock)
            data.append({
                'Stock': stock,
                'Recommendation': rec['recommendation'],
                'Confidence': rec['confidence'],
                'Details': rec['details'],  # Ensure this is a list of dicts
                'Price': price_data[stock].iloc[-1] if not pd.isna(price_data[stock].iloc[-1]) else 0,
                '1D Return': price_data[stock].pct_change().iloc[-1] if not pd.isna(price_data[stock].pct_change().iloc[-1]) else 0,
                '7D Return': price_data[stock].pct_change(7).iloc[-1] if not pd.isna(price_data[stock].pct_change(7).iloc[-1]) else 0,
                'RSI': technical_data[stock]['rsi_14'].iloc[-1] if not pd.isna(technical_data[stock]['rsi_14'].iloc[-1]) else 0,
                'Sentiment': sentiment_data[stock]['vader_compound'].iloc[-1]
            })
        except Exception as e:
            st.warning(f"Error generating recommendation for {stock}: {str(e)}")
            continue
    if not data:
        st.error("No valid recommendations generated. Check data inputs.")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Ensure Details column is not stringified
    df['Details'] = df['Details'].apply(lambda x: x if isinstance(x, list) else [])
    return df

# Initialize models (run automatically, no button dependency)
try:
    pattern_recognizer = PatternRecognizer(price_data, technical_data)
    pattern_recognizer.train_models()
    recommendation_engine = RecommendationEngine(price_data, technical_data, sentiment_data, pattern_recognizer)

    # Check if data is valid before calling create_recommendation_df
    if not stocks or price_data.empty or not technical_data or not sentiment_data:
        st.error("Invalid or empty data. Cannot generate recommendations.")
        df = pd.DataFrame()  # Empty DataFrame
    else:
        df = create_recommendation_df(stocks, recommendation_engine)
        if df.empty:
            st.error("Failed to create recommendation dataframe. Check data sources.")
        else:
            shap_explainer, shap_model = init_shap_explainer(df, technical_data, sentiment_data)
            st.session_state['models_initialized'] = True
            st.success("Models initialized successfully!")
except Exception as e:
    st.error(f"Error initializing models: {str(e)}")
    df = pd.DataFrame()  # Ensure df is defined even on failure
else:
    if 'models_initialized' not in st.session_state:
        st.info("Click 'Initialize Models' to load data and train models")
        df = pd.DataFrame()  # Initialize empty DataFrame if models not initialized
        st.stop()

# Apply decay only if df is not empty
if not df.empty:
    df['DecayedScore'] = df.apply(lambda x: x['Confidence'] * np.exp(-1 / tau), axis=1)
    df = df.sort_values('DecayedScore', ascending=False).reset_index(drop=True)
else:
    st.warning("No recommendations available to apply decay.")

# Main dashboard layout
# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Recommendations", "📈 Alpha Analysis", "🤖 RL Portfolio", "🔍 Stock Analyzer"])

with tab1:
    st.header(f"📈 Top {num_stocks} Stocks for {selected_date.strftime('%Y-%m-%d')}")
    if not df.empty:
        top_stocks = df.head(num_stocks)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Confidence", f"{top_stocks['Confidence'].mean():.2%}")
        col2.metric("Bullish Stocks", f"{sum(top_stocks['Recommendation'] == 'buy')}/{len(top_stocks)}")
        col3.metric("Bearish Stocks", f"{sum(top_stocks['Recommendation'] == 'sell')}/{len(top_stocks)}")
        col4.metric("Avg RSI", f"{top_stocks['RSI'].mean():.1f}")

        for i, row in top_stocks.iterrows():
            with st.expander(f"{i+1}. {row['Stock']} - {row['Recommendation'].upper()} (Confidence: {row['Confidence']:.1%})", expanded=(i==0)):
                col1, col2, col3 = st.columns([1, 2, 2])

                with col1:
                    st.subheader("📋 Recommendation Details")
                    st.metric("Price", f"${row['Price']:.2f}")
                    st.metric("1D Return", f"{row['1D Return']:.2%}",
                             f"{row['1D Return'] - df['1D Return'].mean():.2%} vs avg")
                    st.metric("7D Return", f"{row['7D Return']:.2%}",
                             f"{row['7D Return'] - df['7D Return'].mean():.2%} vs avg")
                    rsi_color = "red" if row['RSI'] > 70 else "green" if row['RSI'] < 30 else None
                    st.metric("RSI", f"{row['RSI']:.1f}", delta_color="off")
                    sentiment_color = "green" if row['Sentiment'] > 0 else "red" if row['Sentiment'] < 0 else None
                    st.metric("Sentiment", f"{row['Sentiment']:.2f}", delta_color="off")

                with col2:
                    st.subheader("📉 Price & Indicators")
                    try:
                        fig = px.line(price_data[row['Stock']].iloc[-90:],
                                     title=f"{row['Stock']} 90-Day Price")
                        tech = technical_data[row['Stock']].iloc[-90:]
                        fig.add_scatter(x=tech.index, y=tech['sma_20'], name='SMA 20', line=dict(color='orange'))
                        fig.add_scatter(x=tech.index, y=tech['sma_50'], name='SMA 50', line=dict(color='purple'))
                        fig.add_scatter(x=tech.index, y=tech['BBL_20_2.0'], name='BB Lower', line=dict(color='gray', dash='dot'))
                        fig.add_scatter(x=tech.index, y=tech['BBU_20_2.0'], name='BB Upper', line=dict(color='gray', dash='dot'))
                        st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{row['Stock']}_{i}")

                        fig2 = px.line(tech['rsi_14'], title="RSI 14")
                        fig2.add_hline(y=70, line_dash="dash", line_color="red")
                        fig2.add_hline(y=30, line_dash="dash", line_color="green")
                        st.plotly_chart(fig2, use_container_width=True, key=f"rsi_chart_{row['Stock']}_{i}")
                    except Exception as e:
                        st.error(f"Error rendering charts for {row['Stock']}: {str(e)}")

                with col3:
                    st.subheader("📊 Model Explanation")
                    with st.expander("Show SHAP Explanation"):
                        try:
                            tech = technical_data[row['Stock']].iloc[-1]
                            sent = sentiment_data[row['Stock']].iloc[-1]
                            features = pd.DataFrame({
                                'sma_20': tech['sma_20'],
                                'sma_50': tech['sma_50'],
                                'sma_200': tech['sma_200'],
                                'rsi': tech['rsi_14'],
                                'macd': tech['MACD_12_26_9'],
                                'bb_width': (tech['BBU_20_2.0'] - tech['BBL_20_2.0']) / tech['BBM_20_2.0'],
                                'sentiment': sent['vader_compound'],
                                'subjectivity': sent['subjectivity'],
                                '1d_return': row['1D Return'],
                                '7d_return': row['7D Return'],
                                'volatility': price_data[row['Stock']].pct_change().std()
                            }, index=[0])
                            shap_values = shap_explainer.shap_values(features)
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values, features,
                                             feature_names=['SMA 20', 'SMA 50', 'SMA 200', 'RSI', 'MACD',
                                                           'BB Width', 'Sentiment', 'Subjectivity',
                                                           '1D Return', '7D Return', 'Volatility'],
                                             plot_type='bar', show=False)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error rendering SHAP plot for {row['Stock']}: {str(e)}")

                    st.subheader("🔍 Recommendation Reasons")
                    details = row['Details']
                    if isinstance(details, str):
                        try:
                            details = ast.literal_eval(details)
                        except (ValueError, SyntaxError) as e:
                            st.error(f"Error parsing details for {row['Stock']}: {str(e)}")
                            details = []

                    for detail in details:
                        if not isinstance(detail, dict):
                            st.warning(f"Invalid detail format for {row['Stock']}: {detail}")
                            continue
                        if detail['signal'] == 'bullish':
                            st.success(f"✅ {detail['type'].upper()}: {detail['reason']} (Confidence: {detail['confidence']:.0%})")
                        elif detail['signal'] == 'bearish':
                            st.error(f"❌ {detail['type'].upper()}: {detail['reason']} (Confidence: {detail['confidence']:.0%})")
                        elif detail['signal'] == 'overbought':
                            st.warning(f"⚠️ {detail['type'].upper()}: {detail['reason']} (Confidence: {detail['confidence']:.0%})")
                        elif detail['signal'] == 'oversold':
                            st.info(f"ℹ️ {detail['type'].upper()}: {detail['reason']} (Confidence: {detail['confidence']:.0%})")

with tab2:
    st.header("📉 Alpha Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📶 Signal Decay Curve")
        times = np.linspace(0, 10, 100)
        decay = np.exp(-times / tau)
        fig = px.line(x=times, y=decay,
                     labels={'x': 'Days', 'y': 'Decay Factor'},
                     title=f"Alpha Decay Curve (τ={tau})")
        st.plotly_chart(fig, use_container_width=True, key="decay_curve")
        st.markdown(f"""
        The decay curve shows how quickly signals lose their predictive power.
        - **τ (tau)** controls the decay rate
        - Current τ = {tau}
        - Signals decay to **37%** of original value at t = τ
        - Signals decay to **5%** of original value at t = 3τ
        """)

    with col2:
        st.subheader("📅 Urgency Indicators")
        if not df.empty:
            urgency_df = df.copy()
            urgency_df['PeakHorizon'] = urgency_df['DecayedScore'].apply(
                lambda x: "1D" if x > 0.7 else "3D" if x > 0.5 else "7D")
            urgency_df['Urgency'] = urgency_df.apply(
                lambda x: f"🟢 Immediate (confidence >70%)" if x['DecayedScore'] > 0.7
                else f"🟡 Moderate (confidence 50-70%)" if x['DecayedScore'] > 0.5
                else "🔴 Low (confidence <50%)", axis=1)
            st.dataframe(urgency_df[['Stock', 'Recommendation', 'DecayedScore', 'Urgency']]
                        .sort_values('DecayedScore', ascending=False)
                        .style.format({'DecayedScore': '{:.1%}'}))
        else:
            st.warning("No recommendations available for urgency indicators.")

with tab3:
    st.header("🤖 Reinforcement Learning Portfolio Allocation")
    if st.button("Train RL Agent"):
        if df.empty:
            st.warning("No recommendations available for RL portfolio allocation.")
        else:
            # pick top stocks and prepare data
            top_stock_symbols = df.sort_values('DecayedScore', ascending=False).head(10)['Stock'].tolist()
            rl_price_data = price_data[top_stock_symbols].dropna(axis=1)

            if len(rl_price_data.columns) < 2:
                st.error("Cannot proceed with portfolio allocation: fewer than 2 stocks with valid data")
            else:
                with st.spinner("🤖 Training RL Agent… this may take a while"):
                    # prepare slices
                    rl_technical_data = {s: technical_data[s] for s in rl_price_data.columns}
                    rl_sentiment_data = {s: sentiment_data[s] for s in rl_price_data.columns}

                    # build environment & agent
                    env = EnhancedPortfolioEnv(
                        stocks=rl_price_data.columns,
                        prices=rl_price_data,
                        technicals=rl_technical_data,
                        macro=macro_data,
                        sentiment=rl_sentiment_data,
                        window_size=10
                    )
                    state = env.reset()
                    state_size = state.shape[0]
                    action_size = env.n_stocks
                    agent = EnhancedRLAgent(state_size, action_size, window_size=10)

                    # UI elements for training feedback
                    progress_bar = st.progress(0)
                    status_text  = st.empty()
                    reward_chart = st.line_chart([], height=200)

                    # training loop
                    rewards = []
                    for e in range(rl_training_epochs):
                        state = env.reset()
                        total_reward = 0

                        # STEP through the episode
                        done = False
                        while not done:
                            action = agent.act(state)
                            next_state, reward, done, _ = env.step(action)
                            agent.remember(state, action, reward, next_state, done)
                            state = next_state
                            total_reward += reward

                        # now run replay only once
                        if len(agent.memory) > 32:
                            agent.replay(32)

                        # update UI
                        progress_bar.progress((e+1)/rl_training_epochs)
                        status_text.text(f"Epoch {e+1}/{rl_training_epochs} – Reward: {total_reward:.2f}")
                        reward_chart.add_rows([total_reward])


                    # post-training: compute & show allocation
                    current_state = env._get_state()
                    recommended_weights = agent.act(current_state)
                    recommended_weights /= (recommended_weights.sum() + 1e-10)

                    st.subheader("🎯 Recommended Portfolio Allocation")
                    allocation_df = pd.DataFrame({
                        'Stock': rl_price_data.columns,
                        'Weight': recommended_weights,
                        'Price': rl_price_data.iloc[-1].values,
                        '1D Return': rl_price_data.pct_change().iloc[-1].values
                    }).sort_values('Weight', ascending=False)

                    fig = px.pie(
                        allocation_df,
                        values='Weight',
                        names='Stock',
                        title="Optimal Portfolio Weights"
                    )
                    st.plotly_chart(fig, use_container_width=True, key="portfolio_allocation_pie")
                    st.dataframe(allocation_df.style.format({
                        'Weight': '{:.1%}',
                        'Price': '${:.2f}',
                        '1D Return': '{:.2%}'
                    }))

                    # position sizing
                    st.subheader("📊 Position Sizing Calculator")
                    col1, col2 = st.columns(2)
                    with col1:
                        portfolio_value = st.number_input(
                            "Portfolio Value ($)",
                            min_value=1000,
                            value=10000
                        )
                    with col2:
                        max_risk = st.slider(
                            "Max Risk per Trade (%)",
                            0.1, 5.0, 1.0
                        )

                    allocation_df['Dollar Amount'] = allocation_df['Weight'] * portfolio_value
                    allocation_df['Shares'] = allocation_df['Dollar Amount'] / allocation_df['Price']
                    allocation_df['Risk Amount'] = allocation_df['Dollar Amount'] * (max_risk / 100)

                    st.write("**Detailed Position Sizing:**")
                    st.dataframe(allocation_df.style.format({
                        'Weight': '{:.1%}',
                        'Price': '${:.2f}',
                        '1D Return': '{:.2%}',
                        'Dollar Amount': '${:,.0f}',
                        'Shares': '{:.0f}',
                        'Risk Amount': '${:,.0f}'
                    }))



with tab4:
    st.header("🔍 Stock Analyzer")
    if not df.empty:
        selected_stock = st.selectbox("Select Stock", df['Stock'].unique())
        if selected_stock:
            stock_data = df[df['Stock'] == selected_stock].iloc[0]
            tech_data = technical_data[selected_stock].iloc[-90:]
            price_data_90d = price_data[selected_stock].iloc[-90:]
            sentiment_data_90d = sentiment_data[selected_stock].iloc[-90:]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📈 {selected_stock} Technical Analysis")
                try:
                    fig = px.line(price_data_90d, title=f"{selected_stock} 90-Day Price")
                    fig.add_scatter(x=tech_data.index, y=tech_data['sma_20'], name='SMA 20', line=dict(color='orange'))
                    fig.add_scatter(x=tech_data.index, y=tech_data['sma_50'], name='SMA 50', line=dict(color='purple'))
                    fig.add_scatter(x=tech_data.index, y=tech_data['BBL_20_2.0'], name='BB Lower', line=dict(color='gray', dash='dot'))
                    fig.add_scatter(x=tech_data.index, y=tech_data['BBU_20_2.0'], name='BB Upper', line=dict(color='gray', dash='dot'))
                    st.plotly_chart(fig, use_container_width=True, key=f"price_chart_{selected_stock}")

                    fig2 = px.line(tech_data[['rsi_14']], title="RSI 14")
                    fig2.add_hline(y=70, line_dash="dash", line_color="red")
                    fig2.add_hline(y=30, line_dash="dash", line_color="green")
                    st.plotly_chart(fig2, use_container_width=True, key=f"rsi_chart_{selected_stock}")

                    fig3 = px.line(tech_data[['MACD_12_26_9', 'MACDs_12_26_9']], title="MACD")
                    fig3.add_bar(x=tech_data.index, y=tech_data['MACDh_12_26_9'], name='MACD Histogram')
                    st.plotly_chart(fig3, use_container_width=True, key=f"macd_chart_{selected_stock}")
                except Exception as e:
                    st.error(f"Error rendering technical charts for {selected_stock}: {str(e)}")

            with col2:
                st.subheader(f"📰 {selected_stock} Sentiment Analysis")
                try:
                    fig = px.line(sentiment_data_90d[['vader_compound']],
                                 title="VADER Sentiment Score (Compound)")
                    fig.add_hline(y=0.5, line_dash="dash", line_color="green")
                    fig.add_hline(y=-0.5, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True, key=f"sentiment_chart_{selected_stock}")
                except Exception as e:
                    st.error(f"Error rendering sentiment chart for {selected_stock}: {str(e)}")

                current_sent = sentiment_data[selected_stock].iloc[-1]
                st.metric("Current Sentiment", f"{current_sent['vader_compound']:.2f}",
                          delta_color="off",
                          help="VADER compound sentiment (-1 to 1)")
                st.metric("Subjectivity", f"{current_sent['subjectivity']:.2f}",
                          delta_color="off",
                          help="0 = very objective, 1 = very subjective")

                st.subheader("🔍 Historical Pattern Detection")
                pattern = pattern_recognizer.detect_patterns(selected_stock)
                if pattern['pattern'] == 'bullish':
                    st.success(f"🐂 Bullish Pattern Detected (Confidence: {pattern['confidence']:.0%})")
                elif pattern['pattern'] == 'bearish':
                    st.error(f"🐻 Bearish Pattern Detected (Confidence: {pattern['confidence']:.0%})")
                else:
                    st.info("🔍 No strong pattern detected")
                st.write("**Pattern Reasoning:**")
                st.write(pattern['reason'])

                st.subheader("🎯 Trading Recommendation")
                st.write(f"**Recommendation:** {stock_data['Recommendation'].upper()}")
                st.write(f"**Confidence:** {stock_data['Confidence']:.0%}")

                if stock_data['Recommendation'] == 'buy':
                    st.success("✅ Strong Buy Signal")
                    st.write("**Entry Strategy:**")
                    st.write("- Consider buying in 2 tranches (50% now, 50% on pullback)")
                    st.write("- Set stop loss at 3-5% below entry")
                    st.write("- Target 1: 3-5% profit, Target 2: 7-10% profit")
                elif stock_data['Recommendation'] == 'sell':
                    st.error("❌ Strong Sell Signal")
                    st.write("**Exit Strategy:**")
                    st.write("- Consider selling in 2 tranches (50% now, 50% on bounce)")
                    st.write("- For short positions, set tight stop above resistance")
                else:
                    st.info("🔄 Hold Position")
                    st.write("**Strategy:**")
                    st.write("- Monitor for breakout/breakdown signals")
                    st.write("- Consider reducing position size if holding long-term")
    else:
        st.warning("No stocks available for analysis. Please check data sources or initialize models.")
# Event-triggered models section
st.markdown("---")
st.header("🎯 Event-Triggered Alerts")

if earnings_trigger or macro_trigger or technical_trigger or news_sentiment_trigger:
    cols = st.columns(2)
    col_idx = 0

    if earnings_trigger:
        with cols[col_idx]:
            st.subheader("📊 Earnings Report Signals")
            if not df.empty:
                earnings_df = df.copy()
                earnings_df['EarningsSignal'] = earnings_df['Details'].apply(
                    lambda x: any(d['type'] == 'earnings' for d in (ast.literal_eval(x) if isinstance(x, str) else x)))
                earnings_df = earnings_df[earnings_df['EarningsSignal']]

                if not earnings_df.empty:
                    st.write("Stocks with earnings signals:")
                    st.dataframe(earnings_df[['Stock', 'Recommendation', 'Confidence']]
                                .sort_values('Confidence', ascending=False)
                                .style.format({'Confidence': '{:.1%}'}))

                    st.plotly_chart(fig, use_container_width=True, key="earnings_confidence_dist")
                else:
                    st.info("No strong earnings signals detected")
            else:
                st.info("No recommendations available for earnings signals")
            col_idx += 1

    if macro_trigger:
        with cols[col_idx % 2]:
            st.subheader("🌍 Macroeconomic Signals")
            latest_macro = macro_data.iloc[-1]
            vix_signal = "High" if latest_macro['10-Year Treasury Yield'] > 3.5 else "Medium" if latest_macro['10-Year Treasury Yield'] > 2.5 else "Low"
            st.metric("10-Year Treasury Yield", f"{latest_macro['10-Year Treasury Yield']:.2f}%",
                      f"Signal: {vix_signal}")
            unemp_signal = "High" if latest_macro['Unemployment Rate'] > 5.0 else "Medium" if latest_macro['Unemployment Rate'] > 4.0 else "Low"
            st.metric("Unemployment Rate", f"{latest_macro['Unemployment Rate']:.1f}%",
                      f"Signal: {unemp_signal}")
            infl_signal = "High" if latest_macro['CPI Inflation'] > 3.0 else "Medium" if latest_macro['CPI Inflation'] > 2.0 else "Low"
            st.metric("CPI Inflation", f"{latest_macro['CPI Inflation']:.1f}%",
                      f"Signal: {infl_signal}")
            st.markdown("""
            **Macro Signal Guide:**
            - **High**: Likely to significantly impact market
            - **Medium**: Moderate market impact expected
            - **Low**: Minimal expected market impact
            """)
            col_idx += 1

    if technical_trigger:
        with cols[col_idx % 2]:
            st.subheader("📊 Technical Pattern Signals")
            if not df.empty:
                tech_df = df.copy()
                tech_df['TechSignal'] = tech_df['Details'].apply(
                    lambda x: any(d['type'] == 'technical' for d in (ast.literal_eval(x) if isinstance(x, str) else x)))
                tech_df = tech_df[tech_df['TechSignal']]

                if not tech_df.empty:
                    st.write("Stocks with technical patterns:")
                    st.dataframe(tech_df[['Stock', 'Recommendation', 'Confidence']]
                                .sort_values('Confidence', ascending=False)
                                .style.format({'Confidence': '{:.1%}'}))

                    pattern_types = []
                    for _, row in tech_df.iterrows():
                        details = row['Details']
                        if isinstance(details, str):
                            details = ast.literal_eval(details)
                        for detail in details:
                            if detail['type'] == 'technical':
                                pattern_types.append(detail['signal'])
                    fig = px.histogram(x=pattern_types, title="Technical Pattern Types")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No strong technical patterns detected")
            else:
                st.info("No recommendations available for technical signals")
            col_idx += 1

    if news_sentiment_trigger:
        with cols[col_idx % 2]:
            st.subheader("📰 News Sentiment Signals")
            if not df.empty:
                sent_df = df.copy()
                sent_df['SentimentSignal'] = sent_df['Details'].apply(
                    lambda x: any(d['type'] == 'sentiment' for d in (ast.literal_eval(x) if isinstance(x, str) else x)))
                sent_df = sent_df[sent_df['SentimentSignal']]

                if not sent_df.empty:
                    st.write("Stocks with sentiment signals:")
                    st.dataframe(sent_df[['Stock', 'Recommendation', 'Confidence', 'Sentiment']]
                                .sort_values('Sentiment', ascending=False)
                                .style.format({'Confidence': '{:.1%}', 'Sentiment': '{:.2f}'}))

                    fig = px.scatter(sent_df, x='Sentiment', y='Confidence', color='Recommendation',
                                   size='Confidence', title="Sentiment vs Confidence")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No strong sentiment signals detected")
            else:
                st.info("No recommendations available for sentiment signals")
            col_idx += 1
else:
    st.info("No event triggers selected. Enable triggers in the sidebar.")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer:** This dashboard is for educational purposes only. The information provided should not be construed as investment advice.
Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
""")