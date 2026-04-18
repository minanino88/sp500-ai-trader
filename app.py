import os
import json
import asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import pytz
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from telegram import Bot
import warnings

warnings.filterwarnings('ignore')

# 1. 핵심 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour
TICKER = 'SPY'
STATE_FILE = 'trend_state.json'
HISTORY_FILE = 'history_trend.csv'

# 2. 한투 API 클래스 (기존과 동일)
class KIS_Trader:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv('KIS_APPKEY')
        self.app_secret = os.getenv('KIS_SECRET')
        self.cano = os.getenv('KIS_CANO')
        self.acnt_prdt_cd = os.getenv('KIS_ACNT_PRDT_CD', '01')
        self.token = self._get_token()

    def _get_token(self):
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            data = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
            res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(data))
            return res.json().get('access_token')
        except: return None

    def get_balance(self):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT3012R"}
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NAS", "TR_CRC_CYCD":"USD", "CTX_AREA_FK200":"", "CTX_AREA_NK200":""}
            res = requests.get(url, headers=headers, params=params)
            return float(res.json()['output2']['frcr_dncl_amt_2'])
        except: return 0.0

    def get_holdings(self):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT3012R"}
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NAS", "TR_CRC_CYCD":"USD", "CTX_AREA_FK200":"", "CTX_AREA_NK200":""}
            res = requests.get(url, headers=headers, params=params)
            return res.json().get('output1', [])
        except: return []

    def get_current_price(self, ticker=TICKER):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/price"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"HHDFS00000300"}
            params = {"AUTH": "", "EXCD": "AMS", "PDNO": ticker}
            res = requests.get(url, headers=headers, params=params)
            return float(res.json()['output']['last'])
        except: return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id}
            data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"AMEX", "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except: return {"rt_cd": "1"}

# 3. 판단 엔진
def get_market_data():
    spy = yf.download(TICKER, period='6mo', progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
    spy_close = spy['Close'].copy()
    monthly = spy_close.resample('ME').last().pct_change().dropna()
    return spy_close, monthly

def get_signal(spy_close, monthly):
    recent_returns = monthly.tail(2).values
    consec_down = 0
    for ret in reversed(recent_returns):
        if ret < 0: consec_down += 1
        else: break
    current_price = float(spy_close.iloc[-1])
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)
    else: state = {"in_market": True, "last_exit_price": 0}
    
    if state['in_market']:
        if consec_down >= 2: return "EXIT", f"2개월 연속 하락 ({consec_down}회)", current_price, state
        return "KEEP", "상승 추세 유지", current_price, state
    else:
        rebound = (current_price - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        if rebound >= 0.02: return "RE-ENTER", f"2% 반등 확인 ({rebound*100:.1f}%)", current_price, state
        return "WAIT", f"반등 대기 중 ({rebound*100:.1f}%)", current_price, state

# 4. 트레이딩 실행
async def run_trading():
    spy_close, monthly = get_market_data()
    signal, reason, price, state = get_signal(spy_close, monthly)
    trader = KIS_Trader()
    bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
    chat_id = os.getenv('CHAT_ID')

    if current_hour == 20: # 저녁 8시 매수
        exec_msg = "관망"
        if signal in ["KEEP", "RE-ENTER"]:
            balance = trader.get_balance()
            cur_p = trader.get_current_price()
            qty = int((balance * 0.95) / cur_p)
            if qty >= 1:
                res = trader.send_order(TICKER, qty, "BUY")
                if res.get('rt_cd') == '0': exec_msg = f"✅ 매수 완료 ({qty}주)"
                with open(STATE_FILE, 'w') as f: json.dump({"in_market": True, "last_exit_price": 0}, f)
        elif signal == "EXIT":
            holdings = trader.get_holdings()
            for stock in holdings:
                if stock.get('pdno') == TICKER:
                    qty = int(stock.get('ccld_qty_smtl', 0))
                    trader.send_order(TICKER, qty, "SELL")
                    exec_msg = f"⚠️ 하락 추세 전량 매도"
            with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": price}, f)
        await bot.send_message(chat_id=chat_id, text=f"📊 [20:00 리포트]\n결정: {signal}\n실행: {exec_msg}")

    elif current_hour == 7: # 아침 7시 매도
        holdings = trader.get_holdings()
        for stock in holdings:
            if stock.get('pdno') == TICKER:
                qty = int(stock.get('ccld_qty_smtl', 0))
                trader.send_order(TICKER, qty, "SELL")
        await bot.send_message(chat_id=chat_id, text=f"☀️ [07:00 매도 정산]\n오버나잇 수익 실현")

# 5. 스트림릿 대시보드
def run_dashboard():
    st.set_page_config(page_title="SP500 추세추종", layout="wide")
    st.title("🛡️ SP500 추세추종 상황실")
    try:
        spy_close, monthly = get_market_data()
        signal, reason, price, state = get_signal(spy_close, monthly)
        c1, c2, c3 = st.columns(3)
        c1.metric("현재 포지션", "시장 진입(IN)" if state['in_market'] else "현금 보유(OUT)")
        c2.metric("최신 신호", signal)
        c3.metric("SPY 현재가", f"${price:,.2f}")
        st.info(f"💡 판단 근거: {reason}")
        
        st.subheader("📈 최근 6개월 흐름")
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
        fig.add_trace(go.Scatter(x=spy_close.index, y=spy_close.values, name="SPY", line=dict(color='#00FF00')), row=1, col=1)
        vix = yf.download('^VIX', period='6mo', progress=False, auto_adjust=True)['Close']
        fig.add_trace(go.Bar(x=vix.index, y=vix, name="VIX", marker_color='red'), row=2, col=1)
        fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"데이터를 불러오는 중입니다... (잠시만 기다려주세요)\n에러내용: {e}")

# ==========================================
# 6. 메인 실행 입구 (여기가 수정됨!)
# ==========================================
if __name__ == "__main__":
    # 깃허브 액션 환경이면 트레이딩 실행
    if os.getenv('GITHUB_ACTIONS') == 'true':
        asyncio.run(run_trading())
    # 그 외 모든 상황(스트림릿 클라우드 포함)에서는 대시보드 실행
    else:
        run_dashboard()
