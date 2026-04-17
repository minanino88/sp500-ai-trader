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

# 1. 고도화된 데이터 수집 (DXY, XLK, Gold, Oil, QQQ 추가)
def get_data():
    # S&P500, VIX, 10년물 금리, 달러인덱스, 기술주섹터, 금, 유가, 나스닥
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    try:
        df = yf.download(tickers, period='5y', progress=False)['Close']
        df.columns = ['Oil', 'Gold', 'Dollar', 'SP500', 'QQQ', 'Tech', 'VIX', 'Yield']
        
        # 기술적 지표 계산
        df['MA20'] = df['SP500'].rolling(20).mean()
        df['MA60'] = df['SP500'].rolling(60).mean()
        df['MA200'] = df['SP500'].rolling(200).mean()
        
        # RSI
        delta = df['SP500'].diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
        
        # MACD
        df['MACD'] = df['SP500'].ewm(span=12).mean() - df['SP500'].ewm(span=26).mean()
        
        # 상대 강도 및 계절성 변수
        df['Tech_Relative'] = df['Tech'] / df['SP500']
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        
        return df.dropna()
    except Exception as e:
        st.error(f"데이터 수집 중 오류 발생: {e}")
        return None

# 2. 최종 진화형 예측 로직
def predict_market(df):
    if df is None or len(df) < 200: return None, None
        
    df['Target'] = (df['SP500'].shift(-1) > df['SP500']).astype(int)
    features = ['SP500', 'VIX', 'Yield', 'Dollar', 'Tech', 'Gold', 'Oil', 'QQQ', 
                'MA20', 'MA60', 'MA200', 'RSI', 'MACD', 'Tech_Relative', 'DayOfWeek', 'Month']
    
    X = df[features].iloc[:-1]
    y = df['Target'].iloc[:-1]
    
    # 모델 고도화 (트리 개수 및 깊이 조정)
    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)
    model.fit(X, y)
    
    latest = df[features].tail(1)
    prob = model.predict_proba(latest)[0] # [Short 확률, Long 확률]
    
    # 신뢰도 기반 판정 (60% 미만은 보합)
    THRESHOLD = 0.60
    if prob[1] >= THRESHOLD: pred = 1
    elif prob[0] >= THRESHOLD: pred = 0
    else: pred = 2
        
    return pred, prob

# 3. 이력 관리 함수
def update_history(date, pred, actual=None):
    file_name = 'history.csv'
    new_data = pd.DataFrame([[date, pred, actual]], columns=['날짜', '예측', '실제'])
    if os.path.exists(file_name):
        try:
            history = pd.read_csv(file_name)
            if date in history['날짜'].values:
                if actual is not None: history.loc[history['날짜'] == date, '실제'] = actual
            else: history = pd.concat([history, new_data], ignore_index=True)
        except: history = new_data
    else: history = new_data
    history.to_csv(file_name, index=False, encoding='utf-8-sig')

# --- 메인 실행 ---
df = get_data()
pred, prob = predict_market(df)

if pred is not None:
    today_str = now_kst.strftime('%Y-%m-%d')
    update_history(today_str, pred)
    
    if current_hour == 7: # 아침 정산
        yesterday_str = (now_kst - timedelta(days=1)).strftime('%Y-%m-%d')
        actual_val = 1 if df['SP500'].iloc[-1] > df['SP500'].iloc[-2] else 0
        update_history(yesterday_str, None, actual_val)

    # --- 스트림릿 UI ---
    st.set_page_config(page_title="S&P 500 AI Trader V3", layout="wide")
    st.title("🚀 S&P 500 AI 최종 진화형 리포트")

    # 승률 및 최근 이력 표시
    if os.path.exists('history.csv'):
        h_df = pd.read_csv('history.csv').fillna('-')
        scored_df = h_df[h_df['실제'] != '-']
        if not scored_df.empty:
            win_rate = (scored_df['예측'].astype(float) == scored_df['실제'].astype(float)).mean() * 100
            st.metric("AI 누적 승률 (성적표)", f"{win_rate:.1f}%")
        st.subheader("📅 최근 예측 이력")
        st.table(h_df.tail(10))

    # 현재 신호 및 종목 가이드
    conf = max(prob) * 100
    if pred == 1:
        res_title = "🚀 LONG (매수)"
        guide = "✅ 추천 종목: SPY (1배), UPRO (3배)"
    elif pred == 0:
        res_title = "📉 SHORT (인버스)"
        guide = "✅ 추천 종목: SH (1배 인버스), SPXU (3배 인버스)"
    else:
        res_title = "⚪ 보합 (관망)"
        guide = "✅ 추천 종목: 현금 보유 (진입 자제)"

    st.divider()
    if conf >= 70:
        st.error(f"🔥 [강력 추천 - 비중 확대] {res_title}")
    else:
        st.success(f"🔔 [일반 신호] {res_title}")
    st.info(f"신뢰도: {conf:.1f}% | {guide}")

# 텔레그램 발송 (메시지에 종목 가이드 포함)
async def send_telegram(pred, prob):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    if not (token and chat_id): return

    conf = max(prob) * 100
    is_strong = "🔥 [강력 추천 - 비중 확대]\n" if conf >= 70 else "🔔 [일반 신호]\n"
    
    if pred == 1:
        msg = f"{is_strong}예측: 🚀 LONG (상승)\n신뢰도: {conf:.1f}%\n📍 추천: SPY(1배), UPRO(3배)"
    elif pred == 0:
        msg = f"{is_strong}예측: 📉 SHORT (하락)\n신뢰도: {conf:.1f}%\n📍 추천: SH(1배 인버스), SPXU(3배 인버스)"
    else:
        msg = f"🔔 [보합 관망]\n신뢰도: {conf:.1f}%\n📍 추천: 현금 보유"

    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text=msg)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    if pred is not None:
        asyncio.run(send_telegram(pred, prob))
