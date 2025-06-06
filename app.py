# 1. Imports
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import plotly.graph_objs as go
from streamlit_autorefresh import st_autorefresh

# 2. Streamlit page config (MUST BE FIRST)
st.set_page_config(page_title="NAIVIT: Stock Insights", layout="wide")

# 3. Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# 4. Function definitions

# Fetch stock data
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.warning(f"Error fetching stock data: {e}")
        return pd.DataFrame()

# Analyze news sentiment
def analyze_sentiment(news):
    scores = []
    for headline in news:
        score = analyzer.polarity_scores(headline)
        scores.append(score['compound'])
    return scores

# Predict next day's price
def predict_price(stock_data):
    data = stock_data[['Close']].copy()
    data['Date'] = data.index.map(pd.Timestamp.timestamp)
    X, y = data[['Date']], data['Close']
    model = LinearRegression().fit(X, y)
    future_date = pd.Timestamp.today() + pd.Timedelta(days=1)
    prediction = model.predict(np.array([[future_date.timestamp()]]))
    prediction = float(prediction)  # force scalar
    return prediction

# Fetch news
def fetch_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey=e82901a5482446f4ace3d4e66d8154fe&pageSize=3&sortBy=publishedAt&language=en"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json().get("articles", [])
    except requests.RequestException as e:
        st.warning(f"Error fetching news: {e}")
        return []

# Sidebar: Inputs & news
def sidebar(news_data):
    icons = ["📰", "💹", "⚡", "📈", "🔍"]
    st.sidebar.title("📊 NAIVIT")
    st.sidebar.markdown("#### Live Market News")
    for article in news_data:
        icon = random.choice(icons)
        st.sidebar.markdown(
            f"{icon} **{article['title']}**\n\n"
            f"<small style='color:gray;'>{article['source']['name']} — {article['publishedAt'][:10]}</small>",
            unsafe_allow_html=True
        )

# 5. main() function
def main():
    # Auto-refresh every 5 minutes
    st_autorefresh(interval=5 * 60 * 1000, key="data_refresh")

    st.markdown("""
        <style>
        body {background-color: #f5f5f5;}
        .main {background-color: white; padding: 2rem; border-radius: 10px;}
        .headline {font-weight: bold; font-size: 1.1rem;}
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        ticker = st.text_input("Enter Stock Ticker", "RELIANCE.NS")
        start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.today())

    if ticker and start_date < end_date:
        news_data = fetch_news(ticker.split(".")[0])
        sidebar(news_data)

        with st.container():
            st.title(f"📈 {ticker} Dashboard")
            data = fetch_stock_data(ticker, start_date, end_date)

            if data.empty:
                st.error("No data found for this ticker.")
                return

            # Calculate moving averages
            data['50_MA'] = data['Close'].rolling(window=50).mean()
            data['200_MA'] = data['Close'].rolling(window=200).mean()

            # Headline data
            st.subheader("Recent Data Snapshot")
            st.dataframe(data.tail(5).style.format({"Close": "{:.2f}"}))

            # Sentiment analysis
            news_headlines = [article["title"] for article in news_data]
            sentiments = analyze_sentiment(news_headlines)

            with st.expander("📰 News Sentiment Analysis"):
                for headline, score in zip(news_headlines, sentiments):
                    color = "green" if score > 0 else "red" if score < 0 else "gray"
                    st.markdown(
                        f"<span class='headline' style='color:{color}'>{headline} ({score:.2f})</span>",
                        unsafe_allow_html=True
                    )

            # Price prediction
            prediction = predict_price(data)
            st.markdown(f"### 🔮 Predicted price for next day: ₹{prediction:.2f}")

            # Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['50_MA'], mode='lines', name='50-day MA', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=data['200_MA'], mode='lines', name='200-day MA', line=dict(color='green')))

            # Regression trendline
            x_num = data.index.map(datetime.timestamp)
            reg = LinearRegression().fit(np.array(x_num).reshape(-1, 1), data['Close'])
            trendline = reg.predict(np.array(x_num).reshape(-1, 1))
            fig.add_trace(go.Scatter(x=data.index, y=trendline, mode='lines', name='Trendline', line=dict(color='purple', dash='dash')))

            fig.update_layout(
                title=f"{ticker} Price Trends",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                plot_bgcolor="white",
                paper_bgcolor="#f5f5f5",
                legend=dict(orientation="h")
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter a valid ticker and date range.")

# 6. Main entry
if __name__ == "__main__":
    main()
