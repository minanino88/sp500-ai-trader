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

# 2. 한투 API 클래스
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

# 3. 데이터 엔진
def get_market_data():
    spy_raw = yf.download(TICKER, period='8mo', progress=False, auto_adjust=True)
    vix_raw = yf.download('^VIX', period='8mo', progress=False, auto_adjust=True)
    
    if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
    
    if spy_raw.empty: return pd.Series(), pd.Series(), pd.Series()
        
    spy_close = spy_raw['Close'].copy()
    vix_close = vix_raw['Close'].copy()
    monthly = spy_close.resample('ME').last().pct_change().dropna()
    return spy_close, monthly, vix_close

def get_signal(spy_close, monthly):
    if spy_close.empty or len(monthly) < 2:
        return "WAIT", "데이터 수집 중...", 0.0, {"in_market": True, "last_exit_price": 0}
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

# 4. 트레이딩 로직
async def run_trading():
    spy_close, monthly, vix_close = get_market_data()
    if spy_close.empty: return
    signal, reason, price, state = get_signal(spy_close, monthly)
    trader = KIS_Trader()
    bot = Bot(token=os.getenv('TELEGRAM_TOKEN'))
    chat_id = os.getenv('CHAT_ID')

    if current_hour == 20:
        exec_msg = "관망"
        if signal in ["KEEP", "RE-ENTER"]:
            balance = trader.get_balance()
            cur_p = trader.get_current_price()
            if cur_p > 0:
                qty = int((balance * 0.95) / cur_p)
                if qty >= 1:
                    res = trader.send_order(TICKER, qty, "BUY")
                    if res.get('rt_cd') == '0': exec_msg = f"매수 완료 ({qty}주)"
                    with open(STATE_FILE, 'w') as f: json.dump({"in_market": True, "last_exit_price": 0}, f)
        elif signal == "EXIT":
            holdings = trader.get_holdings()
            for stock in holdings:
                if stock.get('pdno') == TICKER:
                    qty = int(stock.get('ccld_qty_smtl', 0))
                    trader.send_order(TICKER, qty, "SELL")
                    exec_msg = f"전량 매도"
            with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": price}, f)
        if bot: await bot.send_message(chat_id=chat_id, text=f"📊 [20:00 리포트]\n결정: {signal}\n실행: {exec_msg}")

    elif current_hour == 7:
        holdings = trader.get_holdings()
        for stock in holdings:
            if stock.get('pdno') == TICKER:
                qty = int(stock.get('ccld_qty_smtl', 0))
                trader.send_order(TICKER, qty, "SELL")
        if bot: await bot.send_message(chat_id=chat_id, text=f"☀️ [07:00 매도 정산] 완료")

# 5. [수정됨] 개선된 스트림릿 대시보드
def run_dashboard():
    st.set_page_config(page_title="SP500 추세추종 상황실", layout="wide")

    # (1) 커스텀 CSS
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
      .signal-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
      }
    </style>
    """, unsafe_allow_html=True)

    # 데이터 로드
    spy_close, monthly, vix_close = get_market_data()
    if spy_close.empty:
        st.error("데이터 로딩 실패")
        return
    
    signal, reason, price, state = get_signal(spy_close, monthly)
    ohlc = yf.download(TICKER, period='6mo', progress=False, auto_adjust=True)
    if isinstance(ohlc.columns, pd.MultiIndex): ohlc.columns = ohlc.columns.get_level_values(0)

    # (5) 사이드바 추가
    st.sidebar.title("설정")
    st.sidebar.metric("연속 하락 기준", "2개월")
    st.sidebar.metric("재진입 반등 기준", "2%")
    st.sidebar.metric("포지션 크기", "95%")
    if st.sidebar.button("수동 신호 새로고침"):
        st.rerun()

    st.title("🛡️ SP500 Trend Following Dashboard")

    # (2) 상단 지표 카드 5개
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("현재 포지션", "IN" if state.get('in_market') else "OUT")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("신호", signal)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("SPY 현재가", f"${price:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        m_ret = monthly.iloc[-1] * 100 if not monthly.empty else 0
        st.metric("이번달 수익률", f"{m_ret:+.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        vix_val = vix_close.iloc[-1] if not vix_close.empty else 0
        st.metric("VIX 현재값", f"{vix_val:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.info(f"💡 판단 근거: {reason}")

    # (3) 차트 3개 서브플롯
    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], 
                        shared_xaxes=True, vertical_spacing=0.05)
    
    # Row 1: OHLC Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlc.index, open=ohlc['Open'], high=ohlc['High'], low=ohlc['Low'], close=ohlc['Close'],
        name='SPY OHLC', increasing_line_color='#3fb950', decreasing_line_color='#f85149'
    ), row=1, col=1)

    # Row 2: VIX Bar with Dynamic Colors
    vix_data = vix_close.tail(120)
    vix_colors = ['#f85149' if v > 25 else '#d29922' if v > 18 else '#3fb950' for v in vix_data.values]
    fig.add_trace(go.Bar(x=vix_data.index, y=vix_data.values, name='VIX', marker_color=vix_colors), row=2, col=1)

    # Row 3: Volume Bar
    fig.add_trace(go.Bar(x=ohlc.index, y=ohlc['Volume'], name='Volume', marker_color='#58a6ff', opacity=0.7), row=3, col=1)

    fig.update_layout(template='plotly_dark', height=600, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # (4) 월별 수익률 바 차트
    st.subheader("Monthly Returns (Last 12 Months)")
    m_data = monthly.tail(12)
    m_colors = ['#3fb950' if v > 0 else '#f85149' for v in m_data.values]
    m_fig = go.Figure(go.Bar(
        x=m_data.index.strftime('%y/%m'), y=m_data.values * 100,
        marker_color=m_colors
    ))
    m_fig.update_layout(template='plotly_dark', height=200, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(m_fig, use_container_width=True)

    # (6) 거래 이력 테이블 스타일링
    st.subheader("Trade History")
    if os.path.exists(HISTORY_FILE):
        df_hist = pd.read_csv(HISTORY_FILE)
        def style_row(row):
            if row['signal'] == 'EXIT': return ['background-color: #441111'] * len(row)
            if row['signal'] == 'RE-ENTER': return ['background-color: #114411'] * len(row)
            return [''] * len(row)
        st.dataframe(df_hist.style.apply(style_row, axis=1), use_container_width=True)
    else:
        st.write("No history found.")

# 6. 메인 입구
if __name__ == "__main__":
    if os.getenv('GITHUB_ACTIONS') == 'true':
        asyncio.run(run_trading())
    else:
        run_dashboard()
