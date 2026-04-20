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
import warnings

# [수정] telegram 라이브러리 예외 처리
try:
    from telegram import Bot
except ImportError:
    Bot = None

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
# 3. 데이터 엔진 & 신호 판단
# ==========================================
def get_market_data():
    try:
        # [최적화] 1년치 Raw 데이터를 한 번에 가져옴
        spy_raw = yf.download(SIGNAL_TICKER, period='1y', progress=False, auto_adjust=True)
        vix_raw = yf.download('^VIX', period='1y', progress=False, auto_adjust=True)
        
        if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
        if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
        
        if spy_raw.empty or vix_raw.empty:
            return pd.DataFrame(), pd.Series(), pd.Series()
            
        spy_close = spy_raw['Close'].copy()
        vix_close = vix_raw['Close'].copy()
        monthly = spy_close.resample('ME').last().pct_change().dropna()
        # [최적화] 차트용으로 쓰기 위해 spy_raw(OHLC 포함)를 그대로 반환
        return spy_raw, monthly, vix_close
    except Exception:
        return pd.DataFrame(), pd.Series(), pd.Series()

def get_signal(spy_close, monthly, vix_close):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)
    else: state = {"in_market": True, "last_exit_price": 0}

    if spy_close.empty or len(spy_close) < 20:
        return "WAIT", "Not Enough Data", 0.0, state

    current_price = float(spy_close.iloc[-1])
    spy_daily_ret = (spy_close.iloc[-1] / spy_close.iloc[-2]) - 1
    vix_daily_ret = (vix_close.iloc[-1] / vix_close.iloc[-2]) - 1
    
    # [안정화] 3일 누적 수익률 계산 시 길이 검증
    spy_3day_cum_ret = (spy_close.iloc[-1] / spy_close.iloc[-4]) - 1 if len(spy_close) >= 4 else 0.0

    # 1. 긴급 탈출
    if vix_daily_ret >= 0.3: return "EXIT", f"VIX Spike (+{vix_daily_ret*100:.1f}%)", current_price, state
    if spy_daily_ret <= -0.03: return "EXIT", f"SPY Shock ({spy_daily_ret*100:.1f}%)", current_price, state
    if spy_3day_cum_ret <= -0.05: return "EXIT", f"3rd-Day Cum ({spy_3day_cum_ret*100:.1f}%)", current_price, state

    # 2. 보유/미보유 로직
    if state.get('in_market', True):
        recent_returns = monthly.tail(2).values
        consec_down = 0
        for ret in reversed(recent_returns):
            if ret < 0: consec_down += 1
            else: break
        if consec_down >= 2: return "EXIT", "Trend: 2 months down", current_price, state
        return "KEEP", "Uptrend Holding", current_price, state
    else:
        rebound = (current_price - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        vix_now = float(vix_close.iloc[-1]); vix_prev = float(vix_close.iloc[-2])
        vix_20d = vix_close.tail(20); vix_mean = float(vix_20d.mean()); vix_std = float(vix_20d.std())
        vix_upper = vix_mean + 2 * vix_std
        vix_reentry = (vix_now > vix_upper or vix_prev > vix_upper) and (vix_now < vix_prev * 0.95) and (vix_now < vix_20d.max() * 0.90) and (spy_daily_ret > 0)

        if vix_reentry: return "RE-ENTER", f"VIX Reversal ({vix_now:.1f})", current_price, state
        if rebound >= 0.02: return "RE-ENTER", "2% Rebound OK", current_price, state
        return "WAIT", f"Waiting ({rebound*100:.1f}%) | VIX {vix_now:.1f}", current_price, state

# ==========================================
# 4. 트레이딩 실행
# ==========================================
async def run_trading():
    trader = KIS_Trader()
    token = os.getenv('TELEGRAM_TOKEN'); chat_id = os.getenv('CHAT_ID')
    bot = Bot(token=token) if (Bot and token) else None
    
    if current_hour == 20:
        spy_ohlc, monthly, vix_close = get_market_data()
        if spy_ohlc.empty: return
        spy_close = spy_ohlc['Close']
        signal, reason, price, state = get_signal(spy_close, monthly, vix_close)
        
        exec_msg = "Holding"
        current_holding_qty = trader.get_holdings(TRADE_TICKER)
        if signal in ["KEEP", "RE-ENTER"] and current_holding_qty == 0:
            balance = trader.get_balance(); cur_p = trader.get_current_price(TRADE_TICKER)
            if cur_p > 0:
                qty = int((balance * 0.95) / cur_p)
                if qty >= 1:
                    res = trader.send_order(TRADE_TICKER, qty, "BUY")
                    if res.get('rt_cd') == '0':
                        exec_msg = f"BUY SUCCESS ({qty} shares)"
                        with open(STATE_FILE, 'w') as f: json.dump({"in_market": True, "last_exit_price": 0}, f)
        elif signal == "EXIT" and current_holding_qty > 0:
            res = trader.send_order(TRADE_TICKER, current_holding_qty, "SELL")
            if res.get('rt_cd') == '0':
                exec_msg = "EXIT SUCCESS"
                with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": price}, f)
        if bot: await bot.send_message(chat_id=chat_id, text=f"[20:00 Report] {signal}: {exec_msg}")

    elif current_hour == 1:
        spy_intraday = yf.download(SIGNAL_TICKER, period='1d', interval='5m', progress=False, auto_adjust=True)
        if not spy_intraday.empty:
            if isinstance(spy_intraday.columns, pd.MultiIndex): spy_intraday.columns = spy_intraday.columns.get_level_values(0)
            spy_ret = (float(spy_intraday['Close'].iloc[-1]) / float(spy_intraday['Open'].iloc[0])) - 1
            if spy_ret <= -0.03:
                current_holding_qty = trader.get_holdings(TRADE_TICKER)
                if current_holding_qty > 0:
                    trader.send_order(TRADE_TICKER, current_holding_qty, "SELL")
                    if bot: await bot.send_message(chat_id=chat_id, text=f"🚨 [EMERGENCY] SPY Shock. Sold All.")

# ==========================================
# 5. 스트림릿 대시보드
# ==========================================
def run_dashboard():
    st.set_page_config(page_title="SP500 Watchtower v2.9.4", layout="wide")
    st.title(f"🛡️ {TRADE_TICKER} Watchtower")
    
    # [최적화] 데이터 한 번만 다운로드
    spy_ohlc, monthly, vix_close = get_market_data()
    if spy_ohlc.empty or vix_close.empty:
        st.error("❌ Data load failed.")
        return

    spy_close = spy_ohlc['Close']
    signal, reason, price, state = get_signal(spy_close, monthly, vix_close)
    
    # 상단 지표
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Position", "IN" if state.get('in_market') else "OUT")
    with c2: st.metric("Signal", signal)
    with c3: st.metric("SPY Price", f"${price:.2f}")
    with c4: st.metric("Daily Ret", f"{((spy_close.iloc[-1]/spy_close.iloc[-2])-1)*100:+.2f}%")
    with c5: st.metric("VIX Value", f"{vix_close.iloc[-1]:.2f}")

    if signal == "KEEP": st.success(f"[OK] {reason}")
    elif signal == "EXIT": st.error(f"[EMERGENCY] {reason}")
    else: st.info(f"[INFO] {reason}")

    # [최적화] 다운로드 없이 위에서 받은 spy_ohlc 재사용 (인덱스 정렬)
    common_idx = spy_ohlc.index.intersection(vix_close.index)
    # 최근 6개월치만 슬라이싱 (약 126거래일)
    ohlc_plot = spy_ohlc.loc[common_idx].tail(126)
    vix_plot = vix_close.loc[common_idx].tail(126)

    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=ohlc_plot.index, open=ohlc_plot['Open'], high=ohlc_plot['High'], low=ohlc_plot['Low'], close=ohlc_plot['Close'], name='SPY'), row=1, col=1)
    
    v_colors = ['#f85149' if v > 25 else '#d29922' if v > 18 else '#3fb950' for v in vix_plot.values]
    fig.add_trace(go.Bar(x=vix_plot.index, y=vix_plot.values, name='VIX', marker_color=v_colors), row=2, col=1)
    fig.add_trace(go.Bar(x=ohlc_plot.index, y=ohlc_plot['Volume'], name='Volume', marker_color='#58a6ff', opacity=0.7), row=3, col=1)
    fig.update_layout(template='plotly_dark', height=600, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(HISTORY_FILE):
        st.subheader("History Logs")
        df_hist = pd.read_csv(HISTORY_FILE)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

# ==========================================
# 6. 진입점 (환경별 분기)
# ==========================================
if os.getenv('GITHUB_ACTIONS') == 'true':
    asyncio.run(run_trading())
else:
    run_dashboard()
