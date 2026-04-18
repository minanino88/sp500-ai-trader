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

# ==========================================
# 1. 공정 설정
# ==========================================
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour
TICKER = 'SPY'
STATE_FILE = 'trend_state.json'
HISTORY_FILE = 'history_trend.csv'

# ==========================================
# 2. 한국투자증권 API
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

# ==========================================
# 3. 데이터 및 엔진
# ==========================================
def get_market_data():
    spy_raw = yf.download(TICKER, period='8mo', progress=False, auto_adjust=True)
    vix_raw = yf.download('^VIX', period='8mo', progress=False, auto_adjust=True)
    
    if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
    
    if spy_raw.empty or vix_raw.empty: return pd.Series(), pd.Series(), pd.Series()
        
    spy_close = spy_raw['Close'].copy()
    vix_close = vix_raw['Close'].copy()
    monthly = spy_close.resample('ME').last().pct_change().dropna()
    return spy_close, monthly, vix_close

def get_signal(spy_close, monthly):
    if spy_close.empty or len(monthly) < 2:
        return "WAIT", "Data loading...", 0.0, {"in_market": True, "last_exit_price": 0}
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
        if consec_down >= 2: return "EXIT", f"2 months down ({consec_down})", current_price, state
        return "KEEP", "Uptrend OK", current_price, state
    else:
        rebound = (current_price - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        if rebound >= 0.02: return "RE-ENTER", f"2% Rebound ({rebound*100:.1f}%)", current_price, state
        return "WAIT", f"Waiting rebound ({rebound*100:.1f}%)", current_price, state

# ==========================================
# 4. 트레이딩 실행 (수정: BUY 성공 시에만 상태 저장)
# ==========================================
async def run_trading():
    spy_close, monthly, vix_close = get_market_data()
    if spy_close.empty: return
    signal, reason, price, state = get_signal(spy_close, monthly)
    trader = KIS_Trader()
    
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    bot = Bot(token=token) if token else None

    if current_hour == 20:
        exec_msg = "No action"
        if signal in ["KEEP", "RE-ENTER"]:
            balance = trader.get_balance()
            cur_p = trader.get_current_price()
            if cur_p > 0:
                qty = int((balance * 0.95) / cur_p)
                if qty >= 1:
                    res = trader.send_order(TICKER, qty, "BUY")
                    # [수정] BUY 성공 시에만 state 저장하도록 들여쓰기 적용
                    if res.get('rt_cd') == '0':
                        exec_msg = f"BUY OK ({qty} shares)"
                        with open(STATE_FILE, 'w') as f:
                            json.dump({"in_market": True, "last_exit_price": 0}, f)
        elif signal == "EXIT":
            holdings = trader.get_holdings()
            sell_success = False
            for stock in holdings:
                if stock.get('pdno') == TICKER:
                    qty = int(stock.get('ccld_qty_smtl', 0))
                    res = trader.send_order(TICKER, qty, "SELL")
                    if res.get('rt_cd') == '0':
                        exec_msg = f"EXIT SELL OK"
                        sell_success = True
            # SELL 역시 성공 시에만 상태 업데이트 (일관성 유지)
            if sell_success:
                with open(STATE_FILE, 'w') as f:
                    json.dump({"in_market": False, "last_exit_price": price}, f)
                    
        if bot: await bot.send_message(chat_id=chat_id, text=f"[20:00 Report] Signal: {signal}, Action: {exec_msg}")

    elif current_hour == 7:
        holdings = trader.get_holdings()
        for stock in holdings:
            if stock.get('pdno') == TICKER:
                qty = int(stock.get('ccld_qty_smtl', 0))
                trader.send_order(TICKER, qty, "SELL")
        if bot: await bot.send_message(chat_id=chat_id, text="[07:00 Report] Overnight Sell Completed")

# ==========================================
# 5. 스트림릿 대시보드
# ==========================================
def run_dashboard():
    st.set_page_config(page_title="SP500 Trend Station", layout="wide")

    st.markdown("""
    <style>
      .main { background-color: #0d1117; }
      .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
      }
    </style>
    """, unsafe_allow_html=True)

    spy_close, monthly, vix_close = get_market_data()
    if spy_close.empty:
        st.error("Data loading failed. Check connection.")
        return
    
    signal, reason, price, state = get_signal(spy_close, monthly)
    
    ohlc = yf.download(TICKER, period='6mo', progress=False, auto_adjust=True)
    if isinstance(ohlc.columns, pd.MultiIndex): ohlc.columns = ohlc.columns.get_level_values(0)
    
    common_idx = ohlc.index.intersection(vix_close.index)
    ohlc = ohlc.loc[common_idx]
    vix_plot = vix_close.loc[common_idx]

    st.sidebar.title("Settings")
    st.sidebar.metric("Exit Criteria", "2 Months Down")
    st.sidebar.metric("Re-entry Criteria", "2% Rebound")
    st.sidebar.metric("Position Size", "95%")
    if st.sidebar.button("Rerun Signal Analysis"):
        st.rerun()

    st.title("🛡️ SP500 Trend Following Station")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Position", "IN" if state.get('in_market') else "OUT")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Signal", signal)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("SPY Price", f"${price:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        m_ret = monthly.iloc[-1] * 100 if not monthly.empty else 0
        st.metric("Monthly Ret", f"{m_ret:+.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        v_last = vix_close.iloc[-1] if not vix_close.empty else 0
        st.metric("VIX Value", f"{v_last:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.info(f"Analysis: {reason}")

    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], 
                        shared_xaxes=True, vertical_spacing=0.05)
    
    fig.add_trace(go.Candlestick(
        x=ohlc.index, open=ohlc['Open'], high=ohlc['High'], low=ohlc['Low'], close=ohlc['Close'],
        name='SPY OHLC', increasing_line_color='#3fb950', decreasing_line_color='#f85149'
    ), row=1, col=1)

    v_colors = ['#f85149' if v > 25 else '#d29922' if v > 18 else '#3fb950' for v in vix_plot.values]
    fig.add_trace(go.Bar(x=vix_plot.index, y=vix_plot.values, name='VIX', marker_color=v_colors), row=2, col=1)

    fig.add_trace(go.Bar(x=ohlc.index, y=ohlc['Volume'], name='Volume', marker_color='#58a6ff', opacity=0.7), row=3, col=1)

    fig.update_layout(template='plotly_dark', height=600, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Returns (Last 12 Months)")
    m_data = monthly.tail(12)
    m_colors = ['#3fb950' if v > 0 else '#f85149' for v in m_data.values]
    m_fig = go.Figure(go.Bar(
        x=m_data.index.strftime('%y/%m'), y=m_data.values * 100,
        marker_color=m_colors
    ))
    m_fig.update_layout(template='plotly_dark', height=200, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(m_fig, use_container_width=True)

    st.subheader("History Logs")
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        def style_row(row):
            if row['signal'] == 'EXIT': return ['background-color: #441111'] * len(row)
            if row['signal'] == 'RE-ENTER': return ['background-color: #114411'] * len(row)
            return [''] * len(row)
        st.dataframe(df_hist.style.apply(style_row, axis=1), use_container_width=True)
    else:
        st.write("No logs found.")

# ==========================================
# 6. 메인 입구 (수정: 요청하신 __main__ 블록 교체)
# ==========================================
if __name__ == "__main__":
    import sys
    if os.getenv('GITHUB_ACTIONS') == 'true':
        asyncio.run(run_trading())
    else:
        run_dashboard()
