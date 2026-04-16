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

# 1. 시간 설정 (한국 시간 기준)
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

# 8시 실행인지 10시 실행인지 구분
report_type = "1차 신호 (준비)" if current_hour < 21 else "최종 신호 (확정)"

# 2. 데이터 수집 및 지표 계산
def get_data():
    # S&P 500, VIX, 10년물 금리 가져오기
    tickers = ['^GSPC', '^VIX', '^TNX']
    df = yf.download(tickers, period='5y')['Close']
    df.columns = ['DXY_Yield', 'S&P500', 'VIX']
    
    # 기술적 지표 (20/60/120 이평선, RSI, MACD)
    df['MA20'] = df['S&P500'].rolling(20).mean()
    df['MA60'] = df['S&P500'].rolling(60).mean()
    df['MA120'] = df['S&P500'].rolling(120).mean()
    
    delta = df['S&P500'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    
    df['MACD'] = df['S&P500'].ewm(span=12).mean() - df['S&P500'].ewm(span=26).mean()
    return df.dropna()

# 3. AI 모델 예측
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

# 4. 텔레그램 메시지 전송
async def send_telegram(msg):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    if token and chat_id:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=msg)

# 5. 실행 로직
df = get_data()
pred, prob = predict_market(df)
result_text = "🚀 LONG (매수)" if pred == 1 else "📉 SHORT (매도/관망)"

# --- Streamlit 화면 (브라우저 확인용) ---
st.set_page_config(page_title=f"S&P 500 AI {report_type}", layout="wide")
st.title(f"📈 S&P 500 AI {report_type}")
st.subheader(f"결과: {result_text} (신뢰도 {max(prob)*100:.1f}%)")
st.write(f"분석 시간: {now_kst.strftime('%Y-%m-%d %H:%M')}")

fig = go.Figure(data=[go.Candlestick(x=df.index[-60:], open=df['S&P500'][-60:], high=df['S&P500'][-60:], low=df['S&P500'][-60:], close=df['S&P500'][-60:])])
st.plotly_chart(fig, use_container_width=True)

# --- 자동화 알림 (GitHub Actions 실행용) ---
if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    message = (
        f"🔔 [{report_type} 리포트]\n\n"
        f"예측: {result_text}\n"
        f"신뢰도: {max(prob)*100:.1f}%\n"
        f"일시: {now_kst.strftime('%H:%M')} (KST)\n\n"
        f"💡 8시와 10시 신호가 일치할 때 진입하세요!"
    )
    asyncio.run(send_telegram(message))
