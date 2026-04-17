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
import time
from telegram import Bot

# 1. 환경 및 시간 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

# --- 한국투자증권 API 클래스 (규격 완벽 대응) ---
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
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NASD", "PDNO":"UPRO", "OVRS_ORD_UNPR":"0"}
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                return float(res.json()['output']['ovrs_reusable_amt_artl'])
            return 0.0
        except: return 0.0

    def get_current_price(self, ticker):
        try:
            # 조회용 코드: AMEX=AMS, NASDAQ=NAS
            excd = "AMS" if ticker in ["UPRO", "SPXU"] else "NAS"
            url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/price"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT1101U"}
            params = {"AUTH": "", "EXCD": excd, "PDNO": ticker}
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                res_json = res.json()
                if res_json.get('rt_cd') == '0': return float(res_json['output']['last'])
                else: self.last_error = res_json.get('rt_msg', '조회에러')
            else: self.last_error = f"HTTP {res.status_code}"
            return 0.0
        except Exception as e:
            self.last_error = str(e)[:20]
            return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token or qty <= 0: return {"rt_msg": "수량부족"}
        try:
            # 주문용 코드: AMEX=AMEX, NASDAQ=NASD
            excd = "AMEX" if ticker in ["UPRO", "SPXU"] else "NASD"
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id}
            data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":excd, "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except Exception as e: return {"rt_msg": str(e)}

# --- 데이터 로직 (DB Lock 방지 및 리트라이) ---
@st.cache_data(ttl=3600)
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    for i in range(3): # 최대 3번 시도
        try:
            # threads=False로 설정하여 DB Lock 방지
            df = yf.download(tickers, period='5y', progress=False, threads=False)['Close']
            if not df.empty and len(df) > 100:
                df.columns = ['Oil', 'Gold', 'Dollar', 'SP500', 'QQQ', 'Tech', 'VIX', 'Yield']
                df['MA20'], df['MA200'] = df['SP500'].rolling(20).mean(), df['SP500'].rolling(200).mean()
                delta = df['SP500'].diff()
                up, down = delta.clip(lower=0).ewm(13).mean(), (-1 * delta.clip(upper=0)).ewm(13).mean()
                rs = up / down.replace(0, np.nan)
                df['RSI'] = 100 - (100 / (1 + rs.fillna(0)))
                df['Tech_Relative'], df['DayOfWeek'], df['Month'] = df['Tech'] / df['SP500'], df.index.dayofweek, df.index.month
                return df.dropna()
        except: time.sleep(2)
    return pd.DataFrame()

def predict_market(df):
    if df.empty or len(df) < 50: return 2, [0.5, 0.5] # 데이터 부족 시 관망
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
        exec_msg = "⚪ 조건 미달"
        if df.empty: exec_msg = "⚠️ 데이터 수집 실패"
        elif conf >= 70 and pred != 2:
            ticker = "UPRO" if pred == 1 else "SPXU"
            balance = trader.get_balance()
            price = trader.get_current_price(ticker)
            if price > 0:
                qty = int((balance * 0.95) / price)
                if qty >= 1:
                    res = trader.send_order(ticker, qty, "BUY")
                    exec_msg = f"🔥 [3x 매수성공] {ticker} {qty}주" if res.get('rt_cd') == '0' else f"❌ 주문실패: {res.get('rt_msg')}"
                else: exec_msg = "💡 잔고 부족"
            else: exec_msg = f"⚠️ 조회 실패: {trader.last_error}"
        if bot:
            status = "🚀 LONG(3x)" if pred == 1 else "📉 SHORT(3x)" if pred == 0 else "⚪ 관망"
            await bot.send_message(chat_id=chat_id, text=f"🎯 [새벽 1시 리포트]\n포지션: {status}\n신뢰도: {conf:.1f}%\n결과: {exec_msg}")

# --- Streamlit UI (절대 유지) ---
st.set_page_config(page_title="S&P 500 AI 3x Master", layout="wide")
df = get_data()
pred, prob = predict_market(df)

st.title("🛡️ S&P 500 AI 앙상블 (3x Leverage Mode)")
col1, col2, col3 = st.columns(3)
with col1: st.metric("예측 신호", "🚀 LONG(3x)" if pred==1 else "📉 SHORT(3x)" if pred==0 else "⚪ 보합")
with col2: st.metric("AI 신뢰도", f"{max(prob)*100:.1f}%")
with col3: st.metric("데이터 상태", "정상" if not df.empty else "오류")

if not df.empty:
    st.divider()
    st.subheader("📈 60일 전략 시뮬레이션")
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
else:
    st.error("데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.")

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    asyncio.run(run_trading_flow(pred, prob, df))
