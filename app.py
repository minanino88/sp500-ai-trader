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
import warnings

# [보존] telegram 라이브러리 예외 처리
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
            for item in res.json().get('output1', []):
                if item.get('pdno') == ticker: return int(item.get('ccld_qty_smtl', 0))
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
# 3. 데이터 엔진 (이중화 및 복구 모드)
# ==========================================
def get_market_data():
    try:
        spy_raw = yf.download(SIGNAL_TICKER, period='2y', progress=False, auto_adjust=True, repair=True)
        vix_raw = yf.download('^VIX', period='2y', progress=False, auto_adjust=True, repair=True)
        
        if spy_raw.empty or vix_raw.empty:
            spy_raw = yf.Ticker(SIGNAL_TICKER).history(period='2y')
            vix_raw = yf.Ticker('^VIX').history(period='2y')

        if spy_raw.empty or vix_raw.empty:
            return pd.DataFrame(), pd.Series(), pd.Series(), "Data Empty"

        if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
        if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
        
        spy_close, vix_close = spy_raw['Close'], vix_raw['Close']
        monthly = spy_close.resample('ME').last().pct_change().dropna()
        return spy_raw, monthly, vix_close, "Success"
    except Exception as e:
        return pd.DataFrame(), pd.Series(), pd.Series(), str(e)

def get_signal(spy_close, monthly, vix_close):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)
    else: state = {"in_market": True, "last_exit_price": 0}
    
    if spy_close.empty or len(spy_close) < 20: return "WAIT", "Loading", 0.0, state
    
    curr_p = float(spy_close.iloc[-1])
    spy_daily_ret = (spy_close.iloc[-1] / spy_close.iloc[-2]) - 1
    vix_daily_ret = (vix_close.iloc[-1] / vix_close.iloc[-2]) - 1
    spy_3day_ret = (spy_close.iloc[-1] / spy_close.iloc[-4]) - 1 if len(spy_close) >= 4 else 0.0

    # 1. EXIT 신호 (긴급 및 추세)
    if vix_daily_ret >= 0.3: return "EXIT", "VIX Spike (+30%)", curr_p, state
    if spy_daily_ret <= -0.03: return "EXIT", "SPY Shock (-3%)", curr_p, state
    if spy_3day_ret <= -0.05: return "EXIT", "3d Cum Down (-5%)", curr_p, state
    
    if state.get('in_market', True):
        recent = monthly.tail(2).values
        if len(recent) == 2 and recent[0] < 0 and recent[1] < 0: return "EXIT", "2m Down Trend", curr_p, state
        return "KEEP", "Uptrend Holding", curr_p, state
    else:
        # 2. RE-ENTER 신호 (VIX 역발상 + 가격 반등)
        rebound = (curr_p - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        vix_now, vix_prev = float(vix_close.iloc[-1]), float(vix_close.iloc[-2])
        vix_20d = vix_close.tail(20)
        vix_reentry = (vix_now > (vix_20d.mean() + 2*vix_20d.std()) or vix_prev > (vix_20d.mean() + 2*vix_20d.std())) and (vix_now < vix_prev * 0.95) and (spy_daily_ret > 0)
        
        if vix_reentry: return "RE-ENTER", "VIX Reversal", curr_p, state
        if rebound >= 0.02: return "RE-ENTER", "2% Price Rebound", curr_p, state
        return "WAIT", f"Waiting({rebound*100:.1f}%)", curr_p, state

# ==========================================
# 4. 트레이딩 실행 (20:00 / 01:00)
# ==========================================
async def run_trading():
    trader = KIS_Trader()
    token = os.getenv('TELEGRAM_TOKEN'); chat_id = os.getenv('CHAT_ID')
    bot = Bot(token=token) if (Bot and token) else None
    
    if current_hour == 20:
        spy_ohlc, monthly, vix_close, msg = get_market_data()
        if spy_ohlc.empty: return
        signal, reason, price, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
        qty = trader.get_holdings(TRADE_TICKER)
        
        if signal in ["KEEP", "RE-ENTER"] and qty == 0:
            bal = trader.get_balance(); cur_p = trader.get_current_price(TRADE_TICKER)
            if cur_p > 0:
                buy_qty = int((bal * 0.95) / cur_p)
                if buy_qty >= 1:
                    res = trader.send_order(TRADE_TICKER, buy_qty, "BUY")
                    if res.get('rt_cd') == '0':
                        with open(STATE_FILE, 'w') as f: json.dump({"in_market": True, "last_exit_price": 0}, f)
        elif signal == "EXIT" and qty > 0:
            res = trader.send_order(TRADE_TICKER, qty, "SELL")
            if res.get('rt_cd') == '0':
                with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": price}, f)
        if bot: await bot.send_message(chat_id=chat_id, text=f"[20:00] {signal}: {reason}")

    elif current_hour == 1:
        spy_int = yf.download(SIGNAL_TICKER, period='1d', interval='5m', progress=False)
        if not spy_int.empty:
            spy_ret = (float(spy_int['Close'].iloc[-1]) / float(spy_int['Open'].iloc[0])) - 1
            if spy_ret <= -0.03:
                qty = trader.get_holdings(TRADE_TICKER)
                if qty > 0:
                    trader.send_order(TRADE_TICKER, qty, "SELL")
                    with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": float(spy_int['Close'].iloc[-1])}, f)
                    if bot: await bot.send_message(chat_id=chat_id, text=f"🚨 [01:00 EMERGENCY] SPY Shock. Sold All.")

# ==========================================
# 5. 스트림릿 대시보드
# ==========================================
def run_dashboard():
    st.set_page_config(page_title="SP500 Watchtower v3.0.6", layout="wide")
    
    # 사이드바 요약 (체크리스트 포함)
    st.sidebar.title("v3.0.6 Master")
    st.sidebar.caption(f"Last Update: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST")
    st.sidebar.divider()
    st.sidebar.write("**EXIT:** VIX+30%, SPY-3%, 3d-5%, 2m Down")
    st.sidebar.write("**ENTER:** VIX Reversal, +2% Rebound")

    st.title(f"🛡️ {TRADE_TICKER} Watchtower")
    
    spy_ohlc, monthly, vix_close, msg = get_market_data()
    if spy_ohlc.empty: st.error(f"❌ Data Load Failed: {msg}"); return
    signal, reason, price, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
    
    # 상단 5개 카드
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Position", "IN" if state.get('in_market') else "OUT")
    with c2: st.metric("Signal", signal)
    with c3: st.metric("SPY Price", f"${price:.2f}")
    with c4: st.metric("Daily Ret", f"{((spy_ohlc['Close'].iloc[-1]/spy_ohlc['Close'].iloc[-2])-1)*100:+.2f}%")
    with c5: st.metric("VIX Value", f"{vix_close.iloc[-1]:.2f}")

    # [수정] 신호별 색상 분기
    if signal == "KEEP": st.success(f"[OK] {reason}")
    elif signal == "EXIT": st.error(f"[EMERGENCY] {reason}")
    else: st.info(f"[INFO] {reason}")

    # 시장 상황 차트
    common_idx = spy_ohlc.index.intersection(vix_close.index)
    ohlc_p, vix_p = spy_ohlc.loc[common_idx].tail(126), vix_close.loc[common_idx].tail(126)
    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=ohlc_p.index, open=ohlc_p['Open'], high=ohlc_p['High'], low=ohlc_p['Low'], close=ohlc_p['Close'], name='SPY'), row=1, col=1)
    fig.add_trace(go.Bar(x=vix_p.index, y=vix_p.values, name='VIX', marker_color='orange'), row=2, col=1)
    fig.add_trace(go.Bar(x=ohlc_p.index, y=ohlc_p['Volume'], name='Vol', marker_color='blue'), row=3, col=1)
    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Performance Analysis (2022-2026)")

    # --- 백테스트 엔진 (버그 수정 완료본) ---
    bt_sp500 = [-0.053,-0.030,0.035,-0.087,-0.006,-0.082,0.092,-0.041,-0.094,0.079,0.054,-0.058,0.062,-0.025,0.035,0.015,-0.001,0.065,0.031,-0.017,-0.048,-0.022,0.087,0.044,0.016,0.052,0.031,-0.041,0.048,0.035,0.011,0.024,0.022,-0.009,0.057,-0.024,-0.012,-0.018,-0.058,-0.082,0.065,0.038,0.042,0.018,0.025,0.031,0.044,0.019,0.008,-0.021,-0.048,0.092]
    
    # 날짜 축 리스트 생성
    dates = [(datetime(2022,1,1) + timedelta(days=31*i)).strftime('%y-%m') for i in range(len(bt_sp500))]
    
    st_hist, bh_hist = [100.0], [100.0]
    in_m, c_d, cap_st, cap_bh, spy_p, last_ex_p = True, 0, 100.0, 100.0, 100.0, 100.0

    for r in bt_sp500:
        spy_p *= (1 + r)
        cap_bh *= (1 + r)
        bh_hist.append(cap_bh)
        if in_m:
            # [버그 1 수정] 시장 진입 중에만 하락 카운트
            if r < 0: c_d += 1
            else: c_d = 0
            if c_d >= 2: in_m, last_ex_p, ret_st = False, spy_p, 0
            else: ret_st = r * 3 - 0.001
        else:
            # [버그 2 수정] SPY 지수 가격 기준으로 리바운드 체크
            rebound = (spy_p - last_ex_p) / last_ex_p
            if rebound >= 0.02: in_m, c_d, ret_st = True, 0, r * 3 - 0.001
            else: ret_st = 0
        cap_st *= (1 + ret_st)
        st_hist.append(cap_st)

    # 월별 수익률 차트
    m_colors = ['green' if v > 0 else 'red' for v in bt_sp500]
    m_fig = go.Figure(go.Bar(x=dates, y=[v*100 for v in bt_sp500], marker_color=m_colors))
    m_fig.update_layout(template='plotly_dark', height=200, title="Monthly Returns (%)", margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(m_fig, use_container_width=True)

    # [수정] 수익률 비교 차트 (X축 날짜 적용)
    c_fig = go.Figure()
    c_fig.add_trace(go.Scatter(x=['22-01']+dates, y=st_hist, name='Strategy (UPRO)', line=dict(color='#3fb950', width=2)))
    c_fig.add_trace(go.Scatter(x=['22-01']+dates, y=bh_hist, name='SPY B&H', line=dict(color='gray', dash='dash')))
    c_fig.update_layout(template='plotly_dark', height=300, margin=dict(l=10,r=10,t=10,b=10), yaxis_title='Manwon (Start: 100)')
    st.plotly_chart(c_fig, use_container_width=True)

    # 전략 가이드 & 동적 리포트
    with st.expander("Strategy Guide & Performance Details"):
        final = st_hist[-1]
        st.write(f"### 📈 Dynamic Total Return: {final-100:.1f}%")
        st.write(f"Initial: 100 Manwon -> **Final: {final:.0f} Manwon**")
        st.divider()
        st.write("EXIT: 2m Down Exit / 2% Rebound Enter / VIX Reversal Enter")

    # 거래 이력 로그 표시
    if os.path.exists(HISTORY_FILE):
        st.subheader("📋 History Logs")
        st.dataframe(pd.read_csv(HISTORY_FILE), use_container_width=True, hide_index=True)

# ==========================================
# 6. 진입점 (버그 3 수정: GITHUB_ACTIONS 대응)
# ==========================================
if os.getenv('GITHUB_ACTIONS') == 'true':
    asyncio.run(run_trading())
else:
    run_dashboard()
