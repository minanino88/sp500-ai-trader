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

# 시간 설정 (한국 기준)
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

# 1. 데이터 수집 및 예측 로직
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX']
    df = yf.download(tickers, period='5y')['Close']
    df.columns = ['DXY_Yield', 'S&P500', 'VIX']
    df['MA20'] = df['S&P500'].rolling(20).mean()
    df['MA60'] = df['S&P500'].rolling(60).mean()
    df['RSI'] = 100 - (100 / (1 + df['S&P500'].diff().clip(lower=0).rolling(14).mean() / 
                                   -df['S&P500'].diff().clip(upper=0).rolling(14).mean()))
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

# 2. 이력(history.csv) 관리 함수
def update_history(date, pred, actual=None):
    file_name = 'history.csv'
    new_data = pd.DataFrame([[date, pred, actual]], columns=['날짜', '예측', '실제'])
    
    if os.path.exists(file_name):
        history = pd.read_csv(file_name)
        if date in history['날짜'].values:
            if actual is not None:
                history.loc[history['날짜'] == date, '실제'] = actual
        else:
            history = pd.concat([history, new_data])
    else:
        history = new_data
    history.to_csv(file_name, index=False)
    return history

# --- 실행 ---
df = get_data()
pred, prob = predict_market(df)
report_type = "1차 신호 (준비)" if current_hour < 21 else "최종 신호 (확정)"
result_text = "🚀 LONG (매수)" if pred == 1 else "📉 SHORT (매도)"

# 이력 기록 (오늘의 예측값 저장)
today_str = now_kst.strftime('%Y-%m-%d')
update_history(today_str, pred)

# 아침 7시 정산 로직 (어제 종가와 비교)
if current_hour == 7:
    yesterday_str = (now_kst - timedelta(days=1)).strftime('%Y-%m-%d')
    actual_change = 1 if df['S&P500'].iloc[-1] > df['S&P500'].iloc[-2] else 0
    update_history(yesterday_str, None, actual_change)

# --- 스트림릿 화면 구성 ---
st.set_page_config(page_title="S&P 500 AI Trader", layout="wide")
st.title(f"📊 S&P 500 AI {report_type}")

# 상단 승률 메트릭 및 이력 테이블
if os.path.exists('history.csv'):
    h_df = pd.read_csv('history.csv').dropna()
    if not h_df.empty:
        h_df['적중'] = (h_df['예측'] == h_df['실제']).astype(int)
        win_rate = h_df['적중'].mean() * 100
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("AI 누적 정합성(승률)", f"{win_rate:.1f}%")
        col_m2.metric("오늘의 신뢰도", f"{max(prob)*100:.1f}%")
        
        st.subheader("📅 최근 10일 예측 이력")
        # 보기 편하게 변환 (1 -> LONG, 0 -> SHORT)
        display_df = h_df.tail(10).copy()
        display_df['예측'] = display_df['예측'].map({1: 'LONG', 0: 'SHORT'})
        display_df['실제'] = display_df['실제'].map({1: 'LONG', 0: 'SHORT'})
        st.table(display_df)

# 메인 예측 결과 및 차트
st.divider()
st.subheader(f"결과: {result_text}")
st.write(f"분석 시간: {now_kst.strftime('%Y-%m-%d %H:%M')}")

fig = go.Figure(data=[go.Candlestick(x=df.index[-60:], open=df['S&P500'][-60:], 
                                    high=df['S&P500'][-60:], low=df['S&P500'][-60:], close=df['S&P500'][-60:])])
st.plotly_chart(fig, use_container_width=True)

# 텔레그램 발송 로직 (GitHub Actions 전용)
async def send_telegram(msg):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    if token and chat_id:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=msg)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    message = f"🔔 [{report_type}]\n예측: {result_text}\n신뢰도: {max(prob)*100:.1f}%"
    asyncio.run(send_telegram(message))
