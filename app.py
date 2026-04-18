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
from datetime import datetime
from telegram import Bot
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 공정 설정
# ==========================================
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

SIGNAL_TICKER = 'SPY' 
TRADE_TICKER = 'UPRO'

STATE_FILE = 'trend_state.json'
HISTORY_FILE = 'history_trend.csv'

# ==========================================
# 2. 한국투자증권 API 클래스
# ==========================================
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

    def get_holdings(self, ticker=TRADE_TICKER):
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT3012R"}
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NAS", "TR_CRC_CYCD":"USD", "CTX_AREA_FK200":"", "CTX_AREA_NK200":""}
            res = requests.get(url, headers=headers, params=params)
            output1 = res.json().get('output1', [])
            for item in output1:
                if item.get('pdno') == ticker:
                    return int(item.get('ccld_qty_smtl', 0))
            return 0
        except: return 0

    def get_current_price(self, ticker=TRADE_TICKER):
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

# ==========================================
# 3. 데이터 및 엔진
# ==========================================
def get_market_data():
    spy_raw = yf.download(SIGNAL_TICKER, period='8mo', progress=False, auto_adjust=True)
    vix_raw = yf.download('^VIX', period='8mo', progress=False, auto_adjust=True)
    if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
    if spy_raw.empty or vix_raw.empty: return pd.Series(), pd.Series(), pd.Series()
    spy_close = spy_raw['Close'].copy()
    vix_close = vix_raw['Close'].copy()
    monthly = spy_close.resample('ME').last().pct_change().dropna()
    return spy_close, monthly, vix_close

def get_signal(spy_close, monthly, vix_close):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)
    else: state = {"in_market": True, "last_exit_price": 0}

    if spy_close.empty or len(spy_close) < 5:
        return "WAIT", "Loading Data", 0.0, state

    current_price = float(spy_close.iloc[-1])
    vix_daily_ret = (vix_close.iloc[-1] / vix_close.iloc[-2]) - 1
    spy_daily_ret = (spy_close.iloc[-1] / spy_close.iloc[-2]) - 1
    spy_3day_cum_ret = (spy_close.iloc[-1] / spy_close.iloc[-4]) - 1

    if vix_daily_ret >= 0.3:
        return "EXIT", f"EMERGENCY: VIX Spike (+{vix_daily_ret*100:.1f}%)", current_price, state
    if spy_daily_ret <= -0.03:
        return "EXIT", f"EMERGENCY: SPY Shock ({spy_daily_ret*100:.1f}%)", current_price, state
    if spy_3day_cum_ret <= -0.05:
        return "EXIT", f"EMERGENCY: 3rd-Day Cum ({spy_3day_cum_ret*100:.1f}%)", current_price, state

    recent_returns = monthly.tail(2).values
    consec_down = 0
    for ret in reversed(recent_returns):
        if ret < 0: consec_down += 1
        else: break
            
    if state.get('in_market', True):
        if consec_down >= 2: return "EXIT", "Trend: 2 months down", current_price, state
        return "KEEP", "Uptrend Holding", current_price, state
    else:
        rebound = (current_price - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        if rebound >= 0.02: return "RE-ENTER", "2% Rebound OK", current_price, state
        return "WAIT", f"Waiting rebound ({rebound*100:.1f}%)", current_price, state

# ==========================================
# 4. 트레이딩 실행
# ==========================================
async def run_trading():
    trader = KIS_Trader()
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    bot = Bot(token=token) if token else None
    
    if current_hour == 20:
        spy_close, monthly, vix_close = get_market_data()
        if spy_close.empty: return
        signal, reason, price, state = get_signal(spy_close, monthly, vix_close)
        exec_msg = "Holding"
        current_holding_qty = trader.get_holdings(TRADE_TICKER)

        if signal in ["KEEP", "RE-ENTER"] and current_holding_qty == 0:
            balance = trader.get_balance()
            cur_p = trader.get_current_price(TRADE_TICKER)
            if cur_p > 0:
                qty = int((balance * 0.95) / cur_p)
                if qty >= 1:
                    res = trader.send_order(TRADE_TICKER, qty, "BUY")
                    if res.get('rt_cd') == '0':
                        exec_msg = f"BUY SUCCESS ({qty} shares)"
                        with open(STATE_FILE, 'w') as f:
                            json.dump({"in_market": True, "last_exit_price": 0}, f)
        
        elif signal == "EXIT" and current_holding_qty > 0:
            res = trader.send_order(TRADE_TICKER, current_holding_qty, "SELL")
            if res.get('rt_cd') == '0':
                exec_msg = "EXIT SUCCESS (SELL ALL)"
                with open(STATE_FILE, 'w') as f:
                    json.dump({"in_market": False, "last_exit_price": price}, f)
        
        if bot: await bot.send_message(chat_id=chat_id, text=f"[20:00 Report]\nSignal: {signal}\nAction: {exec_msg}\nReason: {reason}")

    elif current_hour == 1:
        spy_intraday = yf.download(SIGNAL_TICKER, period='1d', interval='5m', progress=False, auto_adjust=True)
        if isinstance(spy_intraday.columns, pd.MultiIndex): spy_intraday.columns = spy_intraday.columns.get_level_values(0)
        spy_daily_ret = (float(spy_intraday['Close'].iloc[-1]) / float(spy_intraday['Open'].iloc[0])) - 1 if not spy_intraday.empty else 0.0

        vix_intraday = yf.download('^VIX', period='1d', interval='5m', progress=False, auto_adjust=True)
        if isinstance(vix_intraday.columns, pd.MultiIndex): vix_intraday.columns = vix_intraday.columns.get_level_values(0)
        vix_daily_ret = (float(vix_intraday['Close'].iloc[-1]) / float(vix_intraday['Open'].iloc[0])) - 1 if not vix_intraday.empty else 0.0
        
        current_holding_qty = trader.get_holdings(TRADE_TICKER)
        if current_holding_qty > 0:
            emergency_hit = False
            reason = ""
            if spy_daily_ret <= -0.03:
                emergency_hit = True
                reason = f"Intraday SPY Shock ({spy_daily_ret*100:.1f}%)"
            elif vix_daily_ret >= 0.3:
                emergency_hit = True
                reason = f"Intraday VIX Spike (+{vix_daily_ret*100:.1f}%)"
            
            if emergency_hit:
                res = trader.send_order(TRADE_TICKER, current_holding_qty, "SELL")
                if res.get('rt_cd') == '0':
                    cur_p = trader.get_current_price(SIGNAL_TICKER)
                    with open(STATE_FILE, 'w') as f:
                        json.dump({"in_market": False, "last_exit_price": cur_p}, f)
                    if bot: await bot.send_message(chat_id=chat_id, text=f"[01:00 EMERGENCY] {reason}\nAction: UPRO All Sold")

# ==========================================
# 5. 스트림릿 대시보드 (v2.6)
# ==========================================
def run_dashboard():
    st.set_page_config(page_title="SP500 Trend Station v2.6", layout="wide")
    st.markdown("<style>.metric-card {background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 16px; text-align: center;}</style>", unsafe_allow_html=True)

    spy_close, monthly, vix_close = get_market_data()
    if spy_close.empty: return
    signal, reason, price, state = get_signal(spy_close, monthly, vix_close)
    
    # 사이드바 설정 (4번 요구사항)
    st.sidebar.subheader("Emergency Rules")
    st.sidebar.write("VIX spike: +30% in 1 day")
    st.sidebar.write("SPY drop: -3% in 1 day")
    st.sidebar.write("3-day cum: -5% total")
    st.sidebar.divider()
    st.sidebar.subheader("Trend Rules")
    st.sidebar.write("Exit: 2 months consecutive down")
    st.sidebar.write("Re-entry: +2% rebound from exit price")

    st.title(f"🛡️ {TRADE_TICKER} Watchtower Station")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Position", "IN" if state.get('in_market') else "OUT")
    with c2: st.metric("Signal", signal)
    with c3: st.metric("SPY Price", f"${price:.2f}")
    with c4: st.metric("Daily Ret", f"{((spy_close.iloc[-1]/spy_close.iloc[-2])-1)*100:+.2f}%")
    with c5: st.metric("VIX Value", f"{vix_close.iloc[-1]:.2f}")

    # 신호 분기 (2번 요구사항: 이모지 제거)
    if signal == "KEEP": st.success(f"[OK] {reason}")
    elif signal == "EXIT": st.error(f"[EMERGENCY] {reason}")
    else: st.info(f"[INFO] {reason}")

    # 차트 (1번 요구사항: tail(120) 제거 및 정렬)
    ohlc = yf.download(SIGNAL_TICKER, period='6mo', progress=False, auto_adjust=True)
    if isinstance(ohlc.columns, pd.MultiIndex): ohlc.columns = ohlc.columns.get_level_values(0)
    common_idx = ohlc.index.intersection(vix_close.index)
    ohlc = ohlc.loc[common_idx]
    vix_plot = vix_close.loc[common_idx]

    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=ohlc.index, open=ohlc['Open'], high=ohlc['High'], low=ohlc['Low'], close=ohlc['Close'], name='SPY'), row=1, col=1)
    
    v_colors = ['#f85149' if v > 25 else '#d29922' if v > 18 else '#3fb950' for v in vix_plot.values]
    fig.add_trace(go.Bar(x=vix_plot.index, y=vix_plot.values, name='VIX', marker_color=v_colors), row=2, col=1)
    fig.add_trace(go.Bar(x=ohlc.index, y=ohlc['Volume'], name='Volume', marker_color='#58a6ff', opacity=0.7), row=3, col=1)
    
    fig.update_layout(template='plotly_dark', height=600, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 전략 설명 섹션 (3번 요구사항)
    st.divider()
    st.subheader("How This Works")
    st.markdown("""
    **Trading Tickers**
    - Signal: SPY (S&P 500 ETF) : 판단 기준
    - Trade: UPRO (3x Leveraged) : 실제 매매

    **Daily Schedule**
    - KST 20:00: 추세 체크 + 긴급 조건 체크 + 매수/매도 실행
    - KST 01:00: 장중 긴급 탈출 전용 (SPY -3% 또는 VIX +30% 시 즉시 매도)

    **Exit Conditions (우선순위 순)**
    1. [긴급] VIX 하루 30% 이상 급등
    2. [긴급] SPY 하루 -3% 이상 급락
    3. [긴급] SPY 3일 누적 -5% 이하
    4. [추세] 월말 기준 2개월 연속 하락

    **Re-entry Condition**
    - 매도가 기준으로 +2% 반등 확인 시 재매수

    **Chart Guide**
    - 1번 차트 (캔들): SPY 가격 흐름 (초록=상승봉, 빨강=하락봉)
    - 2번 차트 (VIX): 공포지수 (초록=안정 <18, 노랑=주의 18-25, 빨강=공포 >25)
    - 3번 차트 (Volume): 거래량 (급등 시 변동성 신호)
    """)
    st.info("Backtest 2022-2026: 100만원 : 801만원 (+701%) | 최대낙폭 -14.8% | 월평균 +13.5%")

# ==========================================
# 6. 메인 입구
# ==========================================
if __name__ == "__main__":
    import sys
    if os.getenv('GITHUB_ACTIONS') == 'true':
        asyncio.run(run_trading())
    else:
        run_dashboard()
