import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import os
import asyncio
import requests
import json
from telegram import Bot

# 1. 환경 및 시간 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

# --- 한국투자증권 API (보안을 위해 환경변수 사용) ---
class KIS_Trader:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv('KIS_APPKEY')
        self.app_secret = os.getenv('KIS_SECRET')
        self.cano = os.getenv('KIS_CANO')
        self.acnt_prdt_cd = os.getenv('KIS_ACNT_PRDT_CD')
        self.token = self.get_token()

    def get_token(self):
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            data = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
            res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(data))
            return res.json().get('access_token')
        except: return None

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token: return {"error": "Token발급 실패"}
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
        headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id}
        data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NASD", "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
        return requests.post(url, headers=headers, data=json.dumps(data)).json()

# --- 모델 예측 로직 ---
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    try:
        df = yf.download(tickers, period='5y', progress=False)['Close']
        df.columns = ['Oil', 'Gold', 'Dollar', 'SP500', 'QQQ', 'Tech', 'VIX', 'Yield']
        df['MA20'], df['MA200'] = df['SP500'].rolling(20).mean(), df['SP500'].rolling(200).mean()
        delta = df['SP500'].diff()
        df['RSI'] = 100 - (100 / (1 + (delta.clip(lower=0).ewm(13).mean() / (-1*delta.clip(upper=0)).ewm(13).mean())))
        df['Tech_Relative'], df['DayOfWeek'], df['Month'] = df['Tech'] / df['SP500'], df.index.dayofweek, df.index.month
        return df.dropna()
    except: return None

def predict_market(df):
    df['Target'] = (df['SP500'].shift(-1) > df['SP500']).astype(int)
    features = ['SP500', 'VIX', 'Yield', 'Dollar', 'Tech', 'Gold', 'Oil', 'QQQ', 'MA20', 'MA200', 'RSI', 'Tech_Relative', 'DayOfWeek', 'Month']
    X, y = df[features].iloc[:-1], df['Target'].iloc[:-1]
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42).fit(X, y)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42).fit(X, y)
    prob = (rf.predict_proba(df[features].tail(1))[0] + xgb.predict_proba(df[features].tail(1))[0]) / 2
    pred = 1 if prob[1] >= 0.60 else 0 if prob[0] >= 0.60 else 2
    return pred, prob

# --- 자동 매매 로직 (에러 방지 강화) ---
async def execution_logic(pred, prob):
    trader = KIS_Trader()
    conf = max(prob) * 100
    signal_file = 'last_8pm_signal.txt'
    
    # 8시 신호 안전하게 읽기
    is_consistent = False
    if os.path.exists(signal_file):
        with open(signal_file, 'r') as f:
            is_consistent = (f.read().strip() == str(pred))
    
    # 밤 12시 매수 조건
    if current_hour == 0 and conf >= 70 and is_consistent:
        ticker = "SPY" if pred == 1 else "SH"
        res = trader.send_order(ticker, 1, "BUY")
        return f"🤖 [자동매매] {ticker} 매수 완료: {res.get('rt_msg', '응답없음')}"
    
    return "💡 [대기] 조건 미충족 혹은 신호 분석 중"

# --- 메인 실행부 ---
df = get_data()
pred, prob = predict_market(df)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    # 8시 신호 저장
    if current_hour == 20:
        with open('last_8pm_signal.txt', 'w') as f: f.write(str(pred))
    
    # 매매 실행 및 텔레그램 알림
    exec_msg = asyncio.run(execution_logic(pred, prob))
    
    # 텔레그램 메시지 구성
    async def send_msg():
        token, chat_id = os.getenv('TELEGRAM_TOKEN'), os.getenv('CHAT_ID')
        if not (token and chat_id): return
        msg = f"🔔 [AI 리포트]\n방향: {'LONG' if pred==1 else 'SHORT' if pred==0 else '보합'}\n확신도: {max(prob)*100:.1f}%\n결과: {exec_msg}"
        await Bot(token=token).send_message(chat_id=chat_id, text=msg)
    
    asyncio.run(send_msg())
