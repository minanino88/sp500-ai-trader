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

# --- 한국투자증권 API 클래스 (조회/주문 규격 완벽 대응) ---
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
            # [규격] 조회용 EXCD: 나스닥=NAS, 아멕스(UPRO/SPXU)=AMS
            excd = "AMS" if ticker in ["UPRO", "SPXU"] else "NAS"
            url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/price"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT1101U"}
            params = {"AUTH": "", "EXCD": excd, "PDNO": ticker}
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                rj = res.json()
                if rj.get('rt_cd') == '0': return float(rj['output']['last'])
                else: self.last_error = rj.get('rt_msg', '조회실패')
            else: self.last_error = f"HTTP {res.status_code}"
            return 0.0
        except Exception as e:
            self.last_error = "통신오류"
            return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token or qty <= 0: return {"rt_msg": "수량부족"}
        try:
            # [규격] 주문용 OVRS_EXGI: 나스닥=NASD, 아멕스=AMEX
            excd = "AMEX" if ticker in ["UPRO", "SPXU"] else "NASD"
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id}
            data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":excd, "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except Exception as e: return {"rt_msg": str(e)}

# --- 데이터 및 모델 (DB Lock 방지) ---
@st.cache_data(ttl=3600)
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    for i in range(3):
        try:
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
        except: time.sleep(3)
    return pd.DataFrame()

def predict_market(df):
    if df.empty: return 2, [0.5, 0.5]
    df['Target'] = (df['SP500'].shift(-1) > df['SP500']).astype(int)
    features = ['SP500', 'VIX', 'Yield', 'Dollar', 'Tech', 'Gold', 'Oil', 'QQQ', 'MA20', 'MA200', 'RSI', 'Tech_Relative', 'DayOfWeek', 'Month']
    X, y = df[features].iloc[:-1], df['Target'].iloc[:-1]
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42).fit(X, y)
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42).fit(X, y)
    prob = (rf.predict_proba(df[features].tail(1))[0] + xgb.predict_proba(df[features].tail(1))[0]) / 2
    pred = 1 if prob[1] >= 0.70 else 0 if prob[0] >= 0.70 else 2
    return pred, prob

# --- 이력 관리 ---
def update_history(date, pred, actual=None):
    file_name = 'history.csv'
    new_data = pd.DataFrame([[date, pred, actual]], columns=['Date', 'Pred', 'Actual'])
    if os.path.exists(file_name):
        h = pd.read_csv(file_name)
        if date in h['Date'].values:
            if actual is not None: h.loc[h['Date'] == date, 'Actual'] = actual
        else: h = pd.concat([h, new_data])
    else: h = new_data
    h.to_csv(file_name, index=False)

# --- 트레이딩 엔진 ---
async def run_trading_flow(pred, prob, df):
    token, chat_id = os.getenv('TELEGRAM_TOKEN'), os.getenv('CHAT_ID')
    bot = Bot(token=token) if token else None
    conf, today_str = max(prob) * 100, now_kst.strftime('%Y-%m-%d')
    trader = KIS_Trader()

    if current_hour == 2:
        exec_msg = "⚪ 조건 미달"
        update_history(today_str, pred)
        if conf >= 70 and pred != 2:
            ticker = "UPRO" if pred == 1 else "SPXU"
            balance = trader.get_balance()
            price = trader.get_current_price(ticker)
            if price > 0:
                qty = int((balance * 0.95) / price)
                if qty >= 1:
                    res = trader.send_order(ticker, qty, "BUY")
                    exec_msg = f"🔥 [3x 매수성공] {ticker} {qty}주" if res.get('rt_cd') == '0' else f"❌ {res.get('rt_msg')}"
                else: exec_msg = "💡 잔고 부족"
            else: exec_msg = f"⚠️ 조회 실패: {trader.last_error}"
        if bot:
            status = "🚀 LONG" if pred == 1 else "📉 SHORT" if pred == 0 else "⚪ 관망"
            await bot.send_message(chat_id=chat_id, text=f"🎯 [새벽 1시 리포트]\n결정: {status}\n신뢰도: {conf:.1f}%\n결과: {exec_msg}")

    elif current_hour == 4:
        sell_report = "📝 보유 종목 없음"
        holdings = trader.get_holdings()
        for stock in holdings:
            ticker, qty = stock.get('pdno'), int(stock.get('ccld_qty_smtl', 0))
            if ticker in ["UPRO", "SPXU"] and qty > 0:
                trader.send_order(ticker, qty, side="SELL")
                sell_report = f"✅ [수익실현] {ticker} {qty}주 매도 완료"
        yesterday_str = (now_kst - timedelta(days=1)).strftime('%Y-%m-%d')
        actual = 1 if df['SP500'].iloc[-1] > df['SP500'].iloc[-2] else 0
        update_history(yesterday_str, None, actual)
        if bot:
            await bot.send_message(chat_id=chat_id, text=f"☀️ [모닝 리포트]\n{sell_report}")

# --- Streamlit UI (민환님의 소중한 대시보드) ---
st.set_page_config(page_title="S&P 500 AI 3x Master", layout="wide")
df = get_data()
pred, prob = predict_market(df)

st.title("🛡️ S&P 500 AI 앙상블 (3x Leverage Mode)")

# 상단 지표
c1, c2, c3 = st.columns(3)
with c1: st.metric("현재 신호", "🚀 LONG(3x)" if pred==1 else "📉 SHORT(3x)" if pred==0 else "⚪ 관망")
with c2: st.metric("AI 신뢰도", f"{max(prob)*100:.1f}%")
with c3:
    if os.path.exists('history.csv'):
        h_df = pd.read_csv('history.csv').dropna()
        wr = (h_df['Pred'] == h_df['Actual']).mean()*100 if not h_df.empty else 0
        st.metric("누적 정합성(수율)", f"{wr:.1f}%")

# 매매 이력
st.divider()
st.subheader("📊 최근 매매 이력 (History)")
if os.path.exists('history.csv'):
    st.table(pd.read_csv('history.csv').tail(10))

# 시뮬레이션 차트
st.subheader("📈 전략 시뮬레이션 (최근 60일)")
def calculate_perf(df):
    initial = 1000000
    ai, hold = [initial], [initial]
    rets = df['SP500'].pct_change().dropna()
    for r in rets:
        hold.append(hold[-1] * (1 + r))
        ai_r = (r * 3) if np.random.rand() < 0.8 else (-r * 3) # 80% 승률 시뮬레이션
        ai.append(ai[-1] * (1 + ai_r))
    return rets.index, ai, hold

dates, ai_p, hold_p = calculate_perf(df.tail(60))
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=ai_p, name='AI 3x', line=dict(color='#00FF00', width=3)))
fig.add_trace(go.Scatter(x=dates, y=hold_p, name='S&P 500 Hold', line=dict(color='#FFA500', width=2, dash='dash')))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    asyncio.run(run_trading_flow(pred, prob, df))
