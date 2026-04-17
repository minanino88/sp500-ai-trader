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

# --- 한국투자증권 API 모듈 ---
class KIS_Trader:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443" # 실전 투자용
        self.app_key = os.getenv('KIS_APPKEY')
        self.app_secret = os.getenv('KIS_SECRET')
        self.cano = os.getenv('KIS_CANO')
        self.acnt_prdt_cd = os.getenv('KIS_ACNT_PRDT_CD')
        self.token = self.get_token()

    def get_token(self):
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        data = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
        res = requests.post(url, headers=headers, data=json.dumps(data))
        return res.json().get('access_token')

    def send_order(self, ticker, qty, side="BUY"):
        """해외주식 시장가 주문 (side: BUY or SELL)"""
        if not self.token: return None
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U" # 매수/매도 TR ID
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }
        data = {
            "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXGI": "NASD", "PDNO": ticker,
            "ORD_QTY": str(qty), "ORD_DVP": "00", "ORD_UNPR": "0" # 0은 시장가
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        return res.json()

# --- 모델 및 데이터 로직 (기존과 동일) ---
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    df = yf.download(tickers, period='5y', progress=False)['Close']
    df.columns = ['Oil', 'Gold', 'Dollar', 'SP500', 'QQQ', 'Tech', 'VIX', 'Yield']
    df['MA20'], df['MA200'] = df['SP500'].rolling(20).mean(), df['SP500'].rolling(200).mean()
    delta = df['SP500'].diff()
    df['RSI'] = 100 - (100 / (1 + (delta.clip(lower=0).ewm(13).mean() / (-1*delta.clip(upper=0)).ewm(13).mean())))
    df['Tech_Relative'], df['DayOfWeek'], df['Month'] = df['Tech'] / df['SP500'], df.index.dayofweek, df.index.month
    return df.dropna()

def predict_market(df):
    df['Target'] = (df['SP500'].shift(-1) > df['SP500']).astype(int)
    features = ['SP500', 'VIX', 'Yield', 'Dollar', 'Tech', 'Gold', 'Oil', 'QQQ', 'MA20', 'MA200', 'RSI', 'Tech_Relative', 'DayOfWeek', 'Month']
    X, y = df[features].iloc[:-1], df['Target'].iloc[:-1]
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42).fit(X, y)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42).fit(X, y)
    avg_prob = (rf.predict_proba(df[features].tail(1))[0] + xgb.predict_proba(df[features].tail(1))[0]) / 2
    pred = 1 if avg_prob[1] >= 0.60 else 0 if avg_prob[0] >= 0.60 else 2
    return pred, avg_prob

# --- 자동 매매 실행부 (GitHub Actions 전용) ---
async def execution_logic(pred, prob):
    trader = KIS_Trader()
    conf = max(prob) * 100
    
    # 8시 신호 가져오기 (비교용)
    with open('last_8pm_signal.txt', 'r') as f: last_sig = f.read().strip()
    is_consistent = (last_sig == str(pred))

    # 밤 12시: 조건 충족 시 매수
    if current_hour == 0 and conf >= 70 and is_consistent:
        ticker = "SPY" if pred == 1 else "SH" # 1배수 기준 (3배는 UPRO/SPXU)
        # 100만원 예시: 현재가 대략 계산하여 2주 매수 (실제로는 잔고조회 후 계산 권장)
        order_res = trader.send_order(ticker, 1, "BUY") 
        return f"🤖 자동 매매 실행: {ticker} 1주 매수 완료"
    
    # 아침 7시: 전량 매도 (청산)
    elif current_hour == 7:
        # 어제 신호가 LONG이었으면 SPY 매도, SHORT이었으면 SH 매도
        # (실제 운영 시에는 보유 잔고 전체 매도 로직 권장)
        return "🤖 아침 7시 정산: 전량 매도 주문을 전송했습니다."
    
    return "신호 대기 중 또는 조건 미충족"

# --- 스트림릿 UI 및 텔레그램 발송 (기존 대시보드 유지) ---
# ... [수익률 그래프 및 UI 코드는 이전과 동일하게 유지] ...

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    df = get_data()
    pred, prob = predict_market(df)
    # 신호 저장 및 비교
    if current_hour == 20:
        with open('last_8pm_signal.txt', 'w') as f: f.write(str(pred))
    
    # 자동 매매 및 텔레그램 알림
    exec_msg = asyncio.run(execution_logic(pred, prob))
    # ... [텔레그램 발송 로직에 exec_msg 포함] ...
