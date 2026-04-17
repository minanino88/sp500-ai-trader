import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import os
import asyncio
from telegram import Bot

# 시간 및 환경 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

# 1. 데이터 수집 및 예측 (기존 로직 유지)
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX']
    df = yf.download(tickers, period='5y')['Close']
    df.columns = ['DXY_Yield', 'S&P500', 'VIX']
    df['MA20'] = df['S&P500'].rolling(20).mean()
    df['MA60'] = df['S&P500'].rolling(60).mean()
    df['RSI'] = 100 - (100 / (1 + df['S&P500'].diff().clip(lower=0).rolling(14).mean() / -df['S&P500'].diff().clip(upper=0).rolling(14).mean()))
    df['MACD'] = df['S&P500'].ewm(span=12).mean() - df['S&P500'].ewm(span=26).mean()
    return df.dropna()

def predict_market(df):
    df['Target'] = (df['S&P500'].shift(-1) > df['S&P500']).astype(int)
    features = ['S&P500', 'VIX', 'DXY_Yield', 'MA20', 'MA60', 'RSI', 'MACD']
    X = df[features].iloc[:-1]
    y = df['Target'].iloc[:-1]
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    latest = df[features].tail(1)
    pred = model.predict(latest)[0]
    prob = model.predict_proba(latest)[0]
    return pred, prob

# 2. 히스토리 기록 함수 (새로 추가)
def update_history(date, pred, actual=None):
    file_name = 'history.csv'
    new_data = pd.DataFrame([[date, pred, actual]], columns=['Date', 'Pred', 'Actual'])
    
    if os.path.exists(file_name):
        history = pd.read_csv(file_name)
        # 이미 해당 날짜의 예측이 있다면 업데이트, 없으면 추가
        if date in history['Date'].values:
            if actual is not None:
                history.loc[history['Date'] == date, 'Actual'] = actual
        else:
            history = pd.concat([history, new_data])
    else:
        history = new_data
    history.to_csv(file_name, index=False)
    return history

# --- 실행 로직 ---
df = get_data()
pred, prob = predict_market(df)
result_text = "🚀 LONG" if pred == 1 else "📉 SHORT"

# 히스토리 업데이트 (예측 시점에 날짜와 예측값 저장)
today_str = now_kst.strftime('%Y-%m-%d')
history = update_history(today_str, pred)

# 아침 7시 실행 시 결과 정산 (어제 종가 확인)
if current_hour == 7:
    yesterday_str = (now_kst - timedelta(days=1)).strftime('%Y-%m-%d')
    # 어제 실제 지수 등락 확인 로직 (간소화)
    actual_change = 1 if df['S&P500'].iloc[-1] > df['S&P500'].iloc[-2] else 0
    history = update_history(yesterday_str, None, actual_change)

# --- 스트림릿 화면 (성적표 추가) ---
st.title("📊 S&P 500 AI 성적표 & 리포트")

if os.path.exists('history.csv'):
    h_df = pd.read_csv('history.csv').dropna()
    if not h_df.empty:
        h_df['Hit'] = (h_df['Pred'] == h_df['Actual']).astype(int)
        win_rate = h_df['Hit'].mean() * 100
        st.metric("AI 누적 정합성(승률)", f"{win_rate:.1f}%")
        st.write("최근 10일 이력")
        st.table(h_df.tail(10))

# 텔레그램 발송 (아침 7시 성적표 전용)
async def send_telegram(msg):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    if token and chat_id:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=msg)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    if current_hour == 7:
        h_df = pd.read_csv('history.csv').dropna()
        win_rate = (h_df['Pred'] == h_df['Actual']).mean() * 100
        msg = f"☀️ [모닝 성적표]\n\n어제 예측 결과: {'적중' if h_df['Hit'].iloc[-1] == 1 else '실패'}\n누적 승률: {win_rate:.1f}%"
    else:
        msg = f"🔔 [데일리 리포트]\n\n예측: {result_text}\n신뢰도: {max(prob)*100:.1f}%"
    asyncio.run(send_telegram(msg))
