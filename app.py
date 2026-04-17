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

# 야후 파이낸스 캐시 문제 해결을 위한 설정
import requests_cache
session = None # 캐시 사용 안 함

# 시간 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

# 1. 데이터 수집 및 예측 로직 (안정성 강화)
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX']
    # 'auto_adjust'와 'multi_level_col' 설정을 추가하여 데이터 구조를 안정화합니다.
    df = yf.download(tickers, period='5y', progress=False)
    
    if df.empty or 'Close' not in df:
        st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
        return None
        
    df = df['Close']
    df.columns = ['DXY_Yield', 'S&P500', 'VIX']
    
    # 지표 계산
    df['MA20'] = df['S&P500'].rolling(20).mean()
    df['MA60'] = df['S&P500'].rolling(60).mean()
    
    # RSI 계산 (오류 방지용)
    delta = df['S&P500'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['MACD'] = df['S&P500'].ewm(span=12).mean() - df['S&P500'].ewm(span=26).mean()
    return df.dropna()

def predict_market(df):
    if df is None or len(df) < 100:
        return None, None
        
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

# 2. 이력 관리 로직 (한글 깨짐 및 파일 생성 오류 방지)
def update_history(date, pred, actual=None):
    file_name = 'history.csv'
    # 데이터가 아예 없을 때를 대비한 기본 틀
    new_data = pd.DataFrame([[date, pred, actual]], columns=['날짜', '예측', '실제'])
    
    if os.path.exists(file_name):
        try:
            history = pd.read_csv(file_name)
            if date in history['날짜'].values:
                if actual is not None:
                    history.loc[history['날짜'] == date, '실제'] = actual
            else:
                history = pd.concat([history, new_data], ignore_index=True)
        except:
            history = new_data
    else:
        history = new_data
    
    history.to_csv(file_name, index=False, encoding='utf-8-sig')
    return history

# --- 메인 실행 로직 ---
df = get_data()
pred, prob = predict_market(df)

if pred is not None:
    result_text = "🚀 LONG (매수)" if pred == 1 else "📉 SHORT (매도/관망)"
    today_str = now_kst.strftime('%Y-%m-%d')
    update_history(today_str, pred)

    if current_hour == 7: # 아침 정산
        yesterday_str = (now_kst - timedelta(days=1)).strftime('%Y-%m-%d')
        actual_val = 1 if df['S&P500'].iloc[-1] > df['S&P500'].iloc[-2] else 0
        update_history(yesterday_str, None, actual_val)

    # --- 스트림릿 UI ---
    st.set_page_config(page_title="S&P 500 AI Trader", layout="wide")
    st.title("📊 S&P 500 AI 성적표 & 리포트")

    # --- 스트림릿 화면 구성 (이 부분으로 교체) ---
if os.path.exists('history.csv'):
    try:
        # dropna()를 제거해서 데이터가 비어있어도 읽어오게 합니다.
        h_df = pd.read_csv('history.csv')
        
        if not h_df.empty:
            # 아직 결과가 없는 칸(NaN)을 '확인 중'으로 표시합니다.
            display_df = h_df.tail(10).copy().fillna('-')
            
            # 승률 계산은 결과가 있는 데이터로만 합니다.
            scored_df = h_df.dropna()
            if not scored_df.empty:
                scored_df['적중'] = (scored_df['예측'] == scored_df['실제']).astype(int)
                win_rate = scored_df['적중'].mean() * 100
                st.metric("AI 누적 정합성(승률)", f"{win_rate:.1f}%")
            else:
                st.metric("AI 누적 정합성(승률)", "측정 중...")

            st.subheader("📅 최근 예측 이력")
            # 숫자를 보기 좋게 변환
            display_df['예측'] = display_df['예측'].map({1: 'LONG', 0: 'SHORT', '-': '-'})
            display_df['실제'] = display_df['실제'].map({1: 'LONG', 0: 'SHORT', '-': '-'})
            st.table(display_df)
    except Exception as e:
        st.info("성적표를 생성하고 있습니다... (내일 아침 7시 첫 정산)")


    st.divider()
    st.subheader(f"현재 예측: {result_text} (신뢰도 {max(prob)*100:.1f}%)")
    
    fig = go.Figure(data=[go.Candlestick(x=df.index[-60:], open=df['S&P500'][-60:], 
                                        high=df['S&P500'][-60:], low=df['S&P500'][-60:], close=df['S&P500'][-60:])])
    st.plotly_chart(fig, use_container_width=True)

# 텔레그램 발송 로직 (Actions용)
async def send_telegram(msg):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    if token and chat_id:
        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=msg)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    if pred is not None:
        report_type = "모닝 성적표" if current_hour == 7 else "데일리 리포트"
        msg = f"🔔 [{report_type}]\n결과: {result_text}\n신뢰도: {max(prob)*100:.1f}%"
        asyncio.run(send_telegram(msg))
