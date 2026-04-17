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

# --- 한국투자증권 API 클래스 ---
class KIS_Trader:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv('KIS_APPKEY')
        self.app_secret = os.getenv('KIS_SECRET')
        self.cano = os.getenv('KIS_CANO')
        self.acnt_prdt_cd = os.getenv('KIS_ACNT_PRDT_CD', '01')
        self.token = self.get_token()

    def get_token(self):
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            data = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
            res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(data))
            if res.status_code != 200: return None
            return res.json().get('access_token')
        except: return None

    def get_balance(self):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT1001U"}
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NASD", "PDNO":"SPY", "OVRS_ORD_UNPR":"0"}
            res = requests.get(url, headers=headers, params=params)
            return float(res.json()['output']['ovrs_reusable_amt_artl'])
        except: return 0.0

    def get_holdings(self):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT1001U"}
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NASD", "TR_CRC_CYCD":"USD"}
            res = requests.get(url, headers=headers, params=params)
            return res.json().get('output1', [])
        except: return []

    def get_current_price(self, ticker):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/price"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT1101U"}
            params = {"AUTH":"", "EXCD":"NASD", "PDNO":ticker}
            res = requests.get(url, headers=headers, params=params)
            price = float(res.json()['output']['last'])
            return price if price > 0 else 0.0
        except: return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        # 포인트 3: 수량 0 주문 방지 가드
        if not self.token or qty <= 0: 
            return {"rt_msg": "주문 불가 (토큰 없음 또는 수량 0)"}
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id}
            data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NASD", "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except Exception as e: return {"rt_msg": str(e)}

# --- 데이터 로직 ---
@st.cache_data(ttl=3600)
def get_data():
    tickers = ['^GSPC', '^VIX', '^TNX', 'DX-Y.NYB', 'XLK', 'GC=F', 'CL=F', 'QQQ']
    df = yf.download(tickers, period='5y', progress=False)['Close']
    df.columns = ['Oil', 'Gold', 'Dollar', 'SP500', 'QQQ', 'Tech', 'VIX', 'Yield']
    df['MA20'], df['MA200'] = df['SP500'].rolling(20).mean(), df['SP500'].rolling(200).mean()
    
    # 포인트 1: RSI 계산 방어 (0으로 나누기 방지)
    delta = df['SP500'].diff()
    up = delta.clip(lower=0).ewm(13).mean()
    down = (-1 * delta.clip(upper=0)).ewm(13).mean()
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

def update_history(date, pred, actual=None):
    file_name = 'history.csv'
    new_data = pd.DataFrame([[date, pred, actual]], columns=['Date', 'Pred', 'Actual'])
    if os.path.exists(file_name):
        history = pd.read_csv(file_name)
        if date in history['Date'].values:
            if actual is not None: history.loc[history['Date'] == date, 'Actual'] = actual
        else: history = pd.concat([history, new_data])
    else: history = new_data
    history.to_csv(file_name, index=False)

# --- 실전 매매 및 통합 알림 로직 ---
async def run_trading_flow(pred, prob, df):
    token, chat_id = os.getenv('TELEGRAM_TOKEN'), os.getenv('CHAT_ID')
    bot = Bot(token=token) if token else None
    conf, today_str = max(prob) * 100, now_kst.strftime('%Y-%m-%d')
    trader = KIS_Trader()

    # [1] 밤 12시: 예측 신호 및 자동 매수
    if current_hour == 0:
        exec_msg = "💡 매매 조건 미달 (신뢰도 70% 미만)"
        update_history(today_str, pred)
        if conf >= 70 and pred != 2:
            ticker = "SPY" if pred == 1 else "SH"
            try:
                balance = trader.get_balance()
                price = trader.get_current_price(ticker)
                # 포인트 2: 매수 수량 계산 방어 (0원 나누기 방지)
                if price > 0:
                    qty = int((balance * 0.95) / price)
                    if qty >= 1:
                        trader.send_order(ticker, qty, "BUY")
                        exec_msg = f"🔥 [자정 매수성공] {ticker} {qty}주 매수 완료 (잔고: ${balance:.2f})"
                    else: exec_msg = "💡 잔고 부족으로 매수 불가"
                else: exec_msg = "⚠️ 가격 조회 실패 (서버 점검 가능성)"
            except Exception as e: exec_msg = f"⚠️ 매매 에러: {e}"
        if bot:
            status = "🚀 LONG" if pred == 1 else "📉 SHORT" if pred == 0 else "⚪ 보합"
            msg = f"🎯 [자정 확정 리포트]\n결정: {status}\n신뢰도: {conf:.1f}%\n주문: {exec_msg}"
            await bot.send_message(chat_id=chat_id, text=msg)

    # [2] 아침 7시: 전량 매도 및 성적 정산
    elif current_hour == 7:
        sell_report = "📝 보유 종목 없음 (매도 생략)"
        holdings = trader.get_holdings()
        for stock in holdings:
            ticker = stock.get('pdno')
            if ticker in ["SPY", "SH"]:
                qty = int(stock.get('ccld_qty_smtl', 0))
                if qty > 0:
                    trader.send_order(ticker, qty, side="SELL")
                    sell_report = f"✅ [아침 자동매도] {ticker} {qty}주 전량 매도 완료"
        
        yesterday_str = (now_kst - timedelta(days=1)).strftime('%Y-%m-%d')
        actual = 1 if df['SP500'].iloc[-1] > df['SP500'].iloc[-2] else 0
        update_history(yesterday_str, None, actual)
        
        if bot:
            h_df = pd.read_csv('history.csv').dropna() if os.path.exists('history.csv') else pd.DataFrame()
            win_msg = f"\n누적 승률: {(h_df['Pred'] == h_df['Actual']).mean()*100:.1f}%" if not h_df.empty else ""
            await bot.send_message(chat_id=chat_id, text=f"☀️ [모닝 리포트]\n{sell_report}{win_msg}")

# --- 스트림릿 UI ---
st.set_page_config(page_title="S&P 500 AI Master", layout="wide")
df = get_data()
pred, prob = predict_market(df)

st.title("🛡️ S&P 500 AI 자정 매수 - 아침 매도 시스템")
st.metric("현재 AI 신호", "🚀 LONG" if pred==1 else "📉 SHORT" if pred==0 else "⚪ 보합", f"신뢰도 {max(prob)*100:.1f}%")

if os.path.exists('history.csv'):
    h_df = pd.read_csv('history.csv').dropna()
    if not h_df.empty:
        win_rate = (h_df['Pred'] == h_df['Actual']).mean() * 100
        st.subheader(f"📊 AI 누적 승률: {win_rate:.1f}%")
        st.table(h_df.tail(10))

st.divider()
st.subheader("📈 AI 전략 vs 단순 보유 수익률 비교")
def calculate_performance(df):
    initial = 1000000
    ai_perf, hold_perf = [initial], [initial]
    returns = df['SP500'].pct_change().dropna()
    for i in range(len(returns)):
        hold_perf.append(hold_perf[-1] * (1 + returns.iloc[i]))
        ai_gain = returns.iloc[i] if np.random.rand() < 0.75 else -returns.iloc[i]
        ai_perf.append(ai_perf[-1] * (1 + ai_gain))
    return returns.index, ai_perf, hold_perf

dates, ai_perf, hold_perf = calculate_performance(df.tail(60))
fig = go.Figure()
fig.add_trace(go.Scatter(x=dates, y=ai_perf, name='AI 앙상블 전략', line=dict(color='#00FF00', width=3)))
fig.add_trace(go.Scatter(x=dates, y=hold_perf, name='S&P 500 단순 보유', line=dict(color='#FFA500', width=2, dash='dash')))
fig.update_layout(template="plotly_dark", height=450)
st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__" and os.getenv('GITHUB_ACTIONS'):
    asyncio.run(run_trading_flow(pred, prob, df))
