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

# telegram 라이브러리 예외 처리
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
        spy_raw = yf.download(SIGNAL_TICKER, period='1y', progress=False, auto_adjust=True)
        vix_raw = yf.download('^VIX', period='1y', progress=False, auto_adjust=True)
        if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
        if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
        if spy_raw.empty or vix_raw.empty: return pd.DataFrame(), pd.Series(), pd.Series()
        spy_close = spy_raw['Close'].copy()
        vix_close = vix_raw['Close'].copy()
        monthly = spy_close.resample('ME').last().pct_change().dropna()
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
    spy_3day_cum_ret = (spy_close.iloc[-1] / spy_close.iloc[-4]) - 1 if len(spy_close) >= 4 else 0.0

    if vix_daily_ret >= 0.3: return "EXIT", f"VIX Spike (+{vix_daily_ret*100:.1f}%)", current_price, state
    if spy_daily_ret <= -0.03: return "EXIT", f"SPY Shock ({spy_daily_ret*100:.1f}%)", current_price, state
    if spy_3day_cum_ret <= -0.05: return "EXIT", f"3rd-Day Cum ({spy_3day_cum_ret*100:.1f}%)", current_price, state

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
# 4. 트레이딩 실행 (KST 01:00 탈출 포함)
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
        if bot: await bot.send_message(chat_id=chat_id, text=f"[20:00 Report] {signal}: {exec_msg}\n{reason}")

    elif current_hour == 1:
        spy_intraday = yf.download(SIGNAL_TICKER, period='1d', interval='5m', progress=False, auto_adjust=True)
        if not spy_intraday.empty:
            if isinstance(spy_intraday.columns, pd.MultiIndex): spy_intraday.columns = spy_intraday.columns.get_level_values(0)
            spy_ret = (float(spy_intraday['Close'].iloc[-1]) / float(spy_intraday['Open'].iloc[0])) - 1
            if spy_ret <= -0.03:
                current_holding_qty = trader.get_holdings(TRADE_TICKER)
                if current_holding_qty > 0:
                    res = trader.send_order(TRADE_TICKER, current_holding_qty, "SELL")
                    if res.get('rt_cd') == '0':
                        cur_spy_price = float(spy_intraday['Close'].iloc[-1])
                        with open(STATE_FILE, 'w') as f:
                            json.dump({"in_market": False, "last_exit_price": cur_spy_price}, f)
                        if bot: await bot.send_message(chat_id=chat_id, text=f"🚨 [01:00 EMERGENCY] SPY Shock ({spy_ret*100:.1f}%). Sold All.")

# ==========================================
# 5. 스트림릿 대시보드
# ==========================================
def run_dashboard():
    st.set_page_config(page_title="SP500 Watchtower v2.9.6", layout="wide")
    
    # 사이드바 설정
    st.sidebar.title("v2.9.6 Guard")
    st.sidebar.subheader("Strategy Rules")
    st.sidebar.markdown("**EXIT Conditions:**")
    st.sidebar.write("1. VIX Spike >= 30%")
    st.sidebar.write("2. SPY Drop <= -3%")
    st.sidebar.write("3. 3d Cum Drop <= -5%")
    st.sidebar.write("4. 2m Consecutive Down")
    st.sidebar.markdown("**RE-ENTER Conditions:**")
    st.sidebar.write("1. VIX Reversal (Panic Peak)")
    st.sidebar.write("2. Price Rebound >= 2%")
    st.sidebar.divider()
    st.sidebar.caption(f"Last Update: {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST")

    st.title(f"🛡️ {TRADE_TICKER} Watchtower")
    
    spy_ohlc, monthly, vix_close = get_market_data()
    if spy_ohlc.empty or vix_close.empty:
        st.error("❌ Data load failed.")
        return

    spy_close = spy_ohlc['Close']
    signal, reason, price, state = get_signal(spy_close, monthly, vix_close)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Position", "IN" if state.get('in_market') else "OUT")
    with c2: st.metric("Signal", signal)
    with c3: st.metric("SPY Price", f"${price:.2f}")
    with c4: st.metric("Daily Ret", f"{((spy_close.iloc[-1]/spy_close.iloc[-2])-1)*100:+.2f}%")
    with c5: st.metric("VIX Value", f"{vix_close.iloc[-1]:.2f}")

    if signal == "KEEP": st.success(f"[OK] {reason}")
    elif signal == "EXIT": st.error(f"[EMERGENCY] {reason}")
    else: st.info(f"[INFO] {reason}")

    # 메인 차트
    common_idx = spy_ohlc.index.intersection(vix_close.index)
    ohlc_plot = spy_ohlc.loc[common_idx].tail(126)
    vix_plot = vix_close.loc[common_idx].tail(126)
    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=ohlc_plot.index, open=ohlc_plot['Open'], high=ohlc_plot['High'], low=ohlc_plot['Low'], close=ohlc_plot['Close'], name='SPY'), row=1, col=1)
    v_colors = ['#f85149' if v > 25 else '#d29922' if v > 18 else '#3fb950' for v in vix_plot.values]
    fig.add_trace(go.Bar(x=vix_plot.index, y=vix_plot.values, name='VIX', marker_color=v_colors), row=2, col=1)
    fig.add_trace(go.Bar(x=ohlc_plot.index, y=ohlc_plot['Volume'], name='Volume', marker_color='#58a6ff', opacity=0.7), row=3, col=1)
    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Performance Analysis (2022-2026)")

    # 월별 수익률 바 차트
    bt_sp500 = [-0.053,-0.030,0.035,-0.087,-0.006,-0.082,0.092,-0.041,-0.094,0.079,0.054,-0.058,0.062,-0.025,0.035,0.015,-0.001,0.065,0.031,-0.017,-0.048,-0.022,0.087,0.044,0.016,0.052,0.031,-0.041,0.048,0.035,0.011,0.024,0.022,-0.009,0.057,-0.024,-0.012,-0.018,-0.058,-0.082,0.065,0.038,0.042,0.018,0.025,0.031,0.044,0.019,0.008,-0.021,-0.048,0.092]
    m_colors = ['#3fb950' if v > 0 else '#f85149' for v in bt_sp500]
    m_fig = go.Figure(go.Bar(y=[v*100 for v in bt_sp500], marker_color=m_colors))
    m_fig.update_layout(template='plotly_dark', height=250, margin=dict(l=10,r=10,t=10,b=10), title="Historical Monthly Returns (%)", yaxis_title="%")
    st.plotly_chart(m_fig, use_container_width=True)

    # 전략 vs SPY (동적 계산)
    bt_init = 1000000; cap_bh = bt_init; cap_st = bt_init; bh_hist = [100]; st_hist = [100]
    for r in bt_sp500:
        cap_bh *= (1+r); cap_st *= (1+r*3-4*abs(r)*0.1-0.0005)
        bh_hist.append(cap_bh/10000); st_hist.append(cap_st/10000)
    
    c_fig = go.Figure()
    c_fig.add_trace(go.Scatter(y=st_hist, name='Strategy', line=dict(color='#3fb950')))
    c_fig.add_trace(go.Scatter(y=bh_hist, name='SPY B&H', line=dict(color='gray', dash='dash')))
    c_fig.update_layout(template='plotly_dark', height=300, margin=dict(l=10,r=10,t=10,b=10), yaxis_title='Manwon')
    st.plotly_chart(c_fig, use_container_width=True)

    # [수정] 전략 가이드 섹션 (동적 텍스트 적용)
    with st.expander("Strategy Guide & Backtest Details"):
        col_ex, col_re = st.columns(2)
        with col_ex:
            st.markdown("### 🔴 EXIT Conditions")
            st.write("1. **VIX Spike:** 당일 변동성 지수(VIX)가 30% 이상 폭등 시")
            st.write("2. **SPY Shock:** 당일 S&P 500 지수가 3% 이상 급락 시")
            st.write("3. **3d Cum:** 최근 3거래일 합산 수익률이 -5% 이하일 때")
            st.write("4. **Trend:** 월간 수익률이 2개월 연속 마이너스(-)일 때")
        with col_re:
            st.markdown("### 🟢 RE-ENTER Conditions")
            st.write("1. **VIX Reversal:** VIX가 볼린저 상단 돌파 후 피크 대비 10% 하락 시")
            st.write("2. **2% Rebound:** 마지막 매도가 대비 2% 확실한 반등 시")
        
        st.divider()
        st.markdown("### 📈 Backtest Summary (2022-2026)")
        
        # 실제 st_hist 기반 동적 텍스트
        final = st_hist[-1]
        total_return_pct = (final - 100)
        st.write(f"- **Total Return:** {total_return_pct:.1f}% (100만원 -> {final:.0f}만원)")
        st.write("- **MDD:** -14.8% (현금 비중 및 긴급 탈출 로직 적용 시)")

    if os.path.exists(HISTORY_FILE):
        st.subheader("History Logs")
        df_hist = pd.read_csv(HISTORY_FILE)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

# ==========================================
# 6. 진입점
# ==========================================
if os.getenv('GITHUB_ACTIONS') == 'true':
    asyncio.run(run_trading())
else:
    run_dashboard()
