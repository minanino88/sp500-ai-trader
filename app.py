import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st
import os
import asyncio
from telegram import Bot
from datetime import datetime, timedelta
import pytz

# 시간 및 환경 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    try:
        df = yf.download(tickers, period='5y', progress=False)['Close']
        df.columns = ['Oil', 'Gold', 'Dollar', 'SP500', 'QQQ', 'Tech', 'VIX', 'Yield']
        df['MA20'], df['MA60'], df['MA200'] = df['SP500'].rolling(20).mean(), df['SP500'].rolling(60).mean(), df['SP500'].rolling(200).mean()
        delta = df['SP500'].diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up, ema_down = up.ewm(com=13, adjust=False).mean(), down.ewm(com=13, adjust=False).mean()
        df['RSI'] = 100 - (100 / (1 + (ema_up / ema_down)))
        df['MACD'] = df['SP500'].ewm(span=12).mean() - df['SP500'].ewm(span=26).mean()
        df['Tech_Relative'], df['DayOfWeek'], df['Month'] = df['Tech'] / df['SP500'], df.index.dayofweek, df.index.month
        return df.dropna()
    except: return None

def predict_market(df):
    if df is None or len(df) < 200: return None, None
    df['Target'] = (df['SP500'].shift(-1) > df['SP500']).astype(int)
    features = ['SP500', 'VIX', 'Yield', 'Dollar', 'Tech', 'Gold', 'Oil', 'QQQ', 'MA20', 'MA60', 'MA200', 'RSI', 'MACD', 'Tech_Relative', 'DayOfWeek', 'Month']
    X, y = df[features].iloc[:-1], df['Target'].iloc[:-1]
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42).fit(X, y)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42).fit(X, y)
    latest = df[features].tail(1)
    avg_prob = (rf.predict_proba(latest)[0] + xgb.predict_proba(latest)[0]) / 2
    if avg_prob[1] >= 0.60: pred = 1
    elif avg_prob[0] >= 0.60: pred = 0
    else: pred = 2
    return pred, avg_prob

# 신호 저장 및 비교 로직
def handle_signal_consistency(current_pred):
    signal_file = 'last_8pm_signal.txt'
    consistency_msg = ""
    
    if current_hour == 20: # 오후 8시: 신호 저장
        with open(signal_file, 'w') as f:
            f.write(str(current_pred))
        consistency_msg = "🕒 8시 신호가 저장되었습니다."
    
    elif current_hour == 0: # 밤 12시: 신호 비교
        if os.path.exists(signal_file):
            with open(signal_file, 'r') as f:
                last_pred = f.read().strip()
            if last_pred == str(current_pred):
                consistency_msg = "✅ [신호 일치] 8시 신호와 방향이 같습니다! (신뢰도 업)"
            else:
                consistency_msg = "⚠️ [신호 불일치] 8시 신호와 방향이 달라졌습니다. 주의하세요!"
        else:
            consistency_msg = "❓ 8시 신호 기록을 찾을 수 없습니다."
            
    return consistency_msg

# --- 실행부 ---
df = get_data()
pred, prob = predict_market(df)
consistency_text = handle_signal_consistency(pred)

async def send_telegram(pred, prob, consistency_msg):
    token, chat_id = os.getenv('TELEGRAM_TOKEN'), os.getenv('CHAT_ID')
    if not (token and chat_id): return
    conf = max(prob) * 100
    is_strong = "🔥 [강력 추천 - 비중 확대]\n" if conf >= 70 and "일치" in consistency_msg else "🔔 [알림]\n"
    
    status = "🚀 LONG" if pred == 1 else "📉 SHORT" if pred == 0 else "⚪ 보합"
    guide = "SPY/UPRO" if pred == 1 else "SH/SPXU" if pred == 0 else "현금 보유"
    
    msg = f"{is_strong}방향: {status}\n확신도: {conf:.1f}%\n{consistency_msg}\n👉 종목: {guide}"
    await Bot(token=token).send_message(chat_id=chat_id, text=msg)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    if pred is not None:
        asyncio.run(send_telegram(pred, prob, consistency_text))
