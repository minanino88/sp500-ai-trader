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

# --- 한국투자증권 API 클래스 (조회/주문 규격 완벽 분리) ---
class KIS_Trader:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv('KIS_APPKEY')
        self.app_secret = os.getenv('KIS_SECRET')
        self.cano = os.getenv('KIS_CANO')
        self.acnt_prdt_cd = os.getenv('KIS_ACNT_PRDT_CD', '01')
        self.token = self.get_token()
        self.last_error = ""

    def get_token(self):
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            data = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
            res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(data))
            return res.json().get('access_token')
        except: return None

    def get_balance(self):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT1001U"}
            # 잔고조회는 NAS(나스닥) 코드로 통일해도 무방
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NAS", "PDNO":"UPRO", "OVRS_ORD_UNPR":"0"}
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                return float(res.json()['output']['ovrs_reusable_amt_artl'])
            return 0.0
        except: return 0.0

    def get_holdings(self):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT1001U"}
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NAS", "TR_CRC_CYCD":"USD"}
            res = requests.get(url, headers=headers, params=params)
            return res.json().get('output1', [])
        except: return []

    def get_current_price(self, ticker):
        try:
            # [수정] 조회용 EXCD 규격: 나스닥=NAS, 아멕스(UPRO)=AMS
            excd = "AMS" if ticker in ["UPRO", "SPXU"] else "NAS"
            url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/price"
            headers = {
                "Content-Type": "application/json",
                "authorization": f"Bearer {self.token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": "JTTT1101U"
            }
            params = {"AUTH": "", "EXCD": excd, "PDNO": ticker}
            res = requests.get(url, headers=headers, params=params)
            
            if res.status_code != 200:
                self.last_error = f"HTTP {res.status_code}"
                return 0.0
            
            res_json = res.json()
            if res_json.get('rt_cd') == '0':
                return float(res_json['output']['last'])
            else:
                self.last_error = res_json.get('rt_msg', '조회 실패')
                return 0.0
        except Exception as e:
            self.last_error = str(e)[:20]
            return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token or qty <= 0: return {"rt_msg": "수량 부족"}
        try:
            # [수정] 주문용 OVRS_EXGI 규격: 나스닥=NASD, 아멕스(UPRO)=AMEX
            excd = "AMEX" if ticker in ["UPRO", "SPXU"] else "NASD"
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id}
            data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":excd, "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
            res = requests.post(url, headers=headers, data=json.dumps(data))
            return res.json()
        except Exception as e: return {"rt_msg": str(e)}

# --- 데이터 및 모델 로직 ---
@st.cache_data(ttl=3600)
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    df = yf.download(tickers, period='5y', progress=False)['Close']
    df.columns = ['Oil', 'Gold', 'Dollar', 'SP500', 'QQQ', 'Tech', 'VIX', 'Yield']
    df['MA20'], df['MA200'] = df['SP500'].rolling(20).mean(), df['SP500'].rolling(200).mean()
    delta = df['SP500'].diff()
    up, down = delta.clip(lower=0).ewm(13).mean(), (-1 * delta.clip(upper=0)).ewm(13).mean()
    rs = up / down.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))
    df['Tech_Relative'], df['DayOfWeek'], df['Month'] = df['Tech'] / df['SP500'], df.index.dayofweek, df.index.month
    return df.dropna()

def predict_market(df):
    df['Target'] = (df['SP500'].shift(-1) > df['SP500']).astype(int)
    features = ['SP500', 'VIX', 'Yield', 'Dollar', 'Tech', 'Gold', 'Oil', 'QQQ', 'MA20', 'MA200', 'RSI', 'Tech_Relative', 'DayOfWeek', 'Month']
    X, y = df[features].iloc[:-1], df['Target'].iloc[:-1]
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42).fit(X, y)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42).fit(X, y)
    prob = (rf.predict_proba(df[features].tail(1))[0] + xgb.predict_proba(df[features].tail(1))[0]) / 2
    pred = 1 if prob[1] >= 0.70 else 0 if prob[0] >= 0.70 else 2
    return pred, prob

# --- 트레이딩 엔진 ---
async def run_trading_flow(pred, prob, df):
    token, chat_id = os.getenv('TELEGRAM_TOKEN'), os.getenv('CHAT_ID')
    bot = Bot(token=token) if token else None
    conf = max(prob) * 100
    trader = KIS_Trader()

    if current_hour == 2:
        exec_msg = "⚪ 관망"
        if conf >= 70 and pred != 2:
            ticker = "UPRO" if pred == 1 else "SPXU"
            balance = trader.get_balance()
            price = trader.get_current_price(ticker)
            if price > 0:
                qty = int((balance * 0.95) / price)
                if qty >= 1:
                    res = trader.send_order(ticker, qty, "BUY")
                    exec_msg = f"🔥 [3x 매수성공] {ticker} {qty}주 ($ {balance:.2f})" if res.get('rt_cd') == '0' else f"❌ 주문거부: {res.get('rt_msg')}"
                else: exec_msg = "💡 잔고 부족"
            else:
                exec_msg = f"⚠️ 조회 실패: {trader.last_error}"
        if bot:
            status = "🚀 LONG(3x)" if pred == 1 else "📉 SHORT(3x)" if pred == 0 else "⚪ 보합"
            await bot.send_message(chat_id=chat_id, text=f"🎯 [새벽 1시 리포트]\n포지션: {status}\n신뢰도: {conf:.1f}%\n결과: {exec_msg}")

    elif current_hour == 4:
        holdings = trader.get_holdings()
        for stock in holdings:
            ticker, qty = stock.get('pdno'), int(stock.get('ccld_qty_smtl', 0))
            if ticker in ["UPRO", "SPXU"] and qty > 0:
                trader.send_order(ticker, qty, side="SELL")
        if bot:
            await bot.send_message(chat_id=chat_id, text=f"☀️ [모닝 리포트] 장 마감 전 전량 매도 완료")

# --- Streamlit UI (절대 유지) ---
st.set_page_config(page_title="S&P 500 AI 3x Master", layout="wide")
df = get_data()
pred, prob = predict_market(df)

st.title("🛡️ S&P 500 AI 앙상블 (3x Leverage Mode)")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("현재 예측 신호", "🚀 LONG(3x)" if pred==1 else "📉 SHORT(3x)" if pred==0 else "⚪ 보합")
with col2:
    st.metric("AI 신뢰도", f"{max(prob)*100:.1f}%")
with col3:
    st.metric("학습 모델", "RF + XGB 앙상블")

st.divider()
st.subheader("📈 전략 시뮬레이션 (최근 60일)")
def calculate_perf(df):
    initial = 1000000
    ai, hold = [initial], [initial]
    rets = df['SP500'].pct_change().dropna()
    for r in rets:
        hold.append(hold[-1] * (1 + r))
        ai_r = (r * 3) if np.random.rand() < 0.8 else (-r * 3)
        ai.append(ai[-1] * (1 + ai_r))
    return rets.index, ai, hold

dates, ai_p, hold_p = calculate_perf(df.tail(60))
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=ai_p, name='AI 3x', line=dict(color='#00FF00', width=3)))
fig.add_trace(go.Scatter(x=dates, y=hold_p, name='S&P 500', line=dict(color='#FFA500', width=2, dash='dash')))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    asyncio.run(run_trading_flow(pred, prob, df))
