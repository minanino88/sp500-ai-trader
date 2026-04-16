import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import pytz
import os
import asyncio
from telegram import Bot

# 1. 환경 설정 및 시간대 고정 (한국 시간)
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)

# 2. 데이터 수집 함수 (거시지표 + 기술적지표)
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX']
    df = yf.download(tickers, period='5y')['Close']
    df.columns = ['DXY_Yield', 'S&P500', 'VIX'] # 컬럼명 정렬
    
    # 지표 계산 (20/60/120 이평선, RSI, MACD)
    df['MA20'] = df['S&P500'].rolling(20).mean()
    df['MA60'] = df['S&P500'].rolling(60).mean()
    df['MA120'] = df['S&P500'].rolling(120).mean()
    
    delta = df['S&P500'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    
    df['MACD'] = df['S&P500'].ewm(span=12).mean() - df['S&P500'].ewm(span=26).mean()
    return df.dropna()

# 3. AI 모델 학습 및 예측
def predict_market(df):
    df['Target'] = (df['S&P500'].shift(-1) > df['S&P500']).astype(int)
    features = ['S&P500', 'VIX', 'DXY_Yield', 'MA20', 'MA60', 'MA120', 'RSI', 'MACD']
    X = df[features].iloc[:-1]
    y = df['Target'].iloc[:-1]
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    latest = df[features].tail(1)
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]
    return pred, prob

# 4. 텔레그램 메시지 전송 (GitHub Actions에서만 실행됨)
async def send_telegram(msg):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    if token and chat_id:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=msg)

# 5. 메인 실행 (Streamlit UI)
st.set_page_config(page_title="S&P 500 AI Trader", layout="wide")
st.title("📈 S&P 500 데일리 AI 분석 리포트")

df = get_data()
pred, prob = predict_market(df)

# 대시보드 화면 구성
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("🤖 오늘 밤의 AI 예측")
    result = "🚀 LONG (매수)" if pred == 1 else "📉 SHORT (매도/관망)"
    st.metric("최종 신호", result)
    st.write(f"예측 신뢰도: {max(prob)*100:.1f}%")
    st.write(f"분석 기준(한국): {now_kst.strftime('%Y-%m-%d %H:%M')}")

with col2:
    fig = go.Figure(data=[go.Candlestick(x=df.index[-60:],
                    open=df['S&P500'][-60:], high=df['S&P500'][-60:],
                    low=df['S&P500'][-60:], close=df['S&P500'][-60:])])
    fig.update_layout(title="최근 S&P 500 흐름", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# 자동화 실행 시 텔레그램 발송 로직
if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    message = f"🔔 [S&P 500 AI 리포트]\n\n예측: {result}\n신뢰도: {max(prob)*100:.1f}%\n일시: {now_kst.strftime('%Y-%m-%d %H:%M')}"
    asyncio.run(send_telegram(message))
