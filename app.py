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
from datetime import timedelta
from datetime import datetime as dt
import warnings

try:
    from telegram import Bot
except ImportError:
    Bot = None

warnings.filterwarnings('ignore')

# ==========================================
# 1. 공정 설정 (마스터 사양)
# ==========================================
KST = pytz.timezone('Asia/Seoul')
SIGNAL_TICKER = 'SPY' 
TRADE_TICKER = 'UPRO'
STATE_FILE = 'trend_state.json'
HISTORY_FILE = 'history_trend.csv'

# ==========================================
# 2. KIS API 클래스 (rt=2 방어 및 스캐닝 최적화)
# ==========================================
class KIS_Trader:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv('KIS_APPKEY')
        self.app_secret = os.getenv('KIS_SECRET')
        self.cano = os.getenv('KIS_CANO')
        self.acnt_prdt_cd = os.getenv('KIS_ACNT_PRDT_CD', '01')
        self.token = None
        self.error_detail = "Initial"
        self._set_token()

    def _set_token(self):
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            data = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
            res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(data))
            res_data = res.json()
            self.token = res_data.get('access_token')
            if not self.token: self.error_detail = f"Auth Fail: {res_data.get('msg1')}"
        except Exception as e: self.error_detail = f"Conn: {str(e)}"

    def get_balance(self):
        if not self.token: return 0.0
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psamount"
            headers = {
                "Content-Type":"application/json", "authorization":f"Bearer {self.token}", 
                "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT3007R", "custtype":"P"
            }
            params = {"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd, "OVRS_EXCG_CD": "AMEX", "OVRS_ORD_UNPR": "1", "ITEM_CD": TRADE_TICKER}
            res = requests.get(url, headers=headers, params=params).json()
            return float(res.get('output', {}).get('ord_psbl_frcr_amt', 0))
        except: return 0.0

    def get_holdings(self, ticker=TRADE_TICKER):
        if not self.token: return 0
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
            headers = {
                "Content-Type":"application/json", "authorization":f"Bearer {self.token}", 
                "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT3012R", "custtype":"P"
            }
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXCG_CD":"AMEX", "TR_CRCY_CD":"USD", "CTX_AREA_FK200":"", "CTX_AREA_NK200":""}
            res = requests.get(url, headers=headers, params=params)
            for item in res.json().get('output1', []):
                if item.get('pdno') == ticker: return int(item.get('ccld_qty_smtl', 0))
            return 0
        except: return 0

    # [핵심] rt=2를 정면 돌파하는 현재가 스캔 엔진
    def get_current_price(self, ticker=TRADE_TICKER):
        if not self.token: return 0.0
        # UPRO가 상장된 NYSE Arca의 경우 KIS 시세조회에서는 AMS가 우선입니다.
        for excd in ["AMS", "NYS", "NAS"]:
            try:
                url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
                headers = {
                    "Content-Type":"application/json",
                    "authorization": f"Bearer {self.token}",
                    "appkey": self.app_key,
                    "appsecret": self.app_secret,
                    "tr_id": "HHDFS00000300",
                    "custtype": "P" # 필수 헤더: 개인
                }
                params = {"AUTH": "", "EXCD": excd, "PDNO": ticker}
                res = requests.get(url, headers=headers, params=params, timeout=5)
                if res.status_code == 200:
                    data = res.json()
                    price = float(data.get('output', {}).get('last', 0))
                    if price > 0: return price
            except: continue
        return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token: return {"rt_cd": "1", "rt_msg": "No Token"}
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
            headers = {
                "Content-Type":"application/json", "authorization":f"Bearer {self.token}", 
                "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id, "custtype":"P"
            }
            data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"AMEX", "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except: return {"rt_cd":"1", "rt_msg":"Net Error"}

# ==========================================
# 3. 데이터 엔진 & 지능형 신호 (마스터 준수)
# ==========================================
def get_market_data():
    try:
        spy_raw = yf.download(SIGNAL_TICKER, period='2y', progress=False, auto_adjust=True, repair=True)
        vix_raw = yf.download('^VIX', period='2y', progress=False, auto_adjust=True, repair=True)
        if spy_raw.empty or vix_raw.empty:
            spy_raw = yf.Ticker(SIGNAL_TICKER).history(period='2y')
            vix_raw = yf.Ticker('^VIX').history(period='2y')
        if spy_raw.empty or vix_raw.empty: return pd.DataFrame(), pd.Series(), pd.Series(), "Data Empty"
        if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
        if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
        return spy_raw, spy_raw['Close'].resample('ME').last().pct_change().dropna(), vix_raw['Close'], "Success"
    except Exception as e: return pd.DataFrame(), pd.Series(), pd.Series(), str(e)

def get_signal(spy_close, monthly, vix_close):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)
    else: state = {"in_market": True, "last_exit_price": 0}
    if spy_close.empty or len(spy_close) < 20: return "WAIT", "Loading", 0.0, state
    curr_p = float(spy_close.iloc[-1])
    spy_daily_ret = (spy_close.iloc[-1] / spy_close.iloc[-2]) - 1
    vix_daily_ret = (vix_close.iloc[-1] / vix_close.iloc[-2]) - 1
    spy_3day_ret = (spy_close.iloc[-1] / spy_close.iloc[-4]) - 1 if len(spy_close) >= 4 else 0.0
    if vix_daily_ret >= 0.3: return "EXIT", "VIX Spike", curr_p, state
    if spy_daily_ret <= -0.03: return "EXIT", "SPY Shock", curr_p, state
    if spy_3day_ret <= -0.05: return "EXIT", "3d Cum Down", curr_p, state
    if state.get('in_market', True):
        recent = monthly.tail(2).values
        if len(recent) == 2 and recent[0] < 0 and recent[1] < 0: return "EXIT", "2m Down", curr_p, state
        return "KEEP", "Uptrend Holding", curr_p, state
    else:
        rebound = (curr_p - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        vix_now, vix_prev = float(vix_close.iloc[-1]), float(vix_close.iloc[-2])
        vix_20d = vix_close.tail(20)
        vix_rev = (vix_now > (vix_20d.mean() + 2*vix_20d.std()) or vix_prev > (vix_20d.mean() + 2*vix_20d.std())) and (vix_now < vix_prev * 0.95) and (spy_daily_ret > 0)
        if vix_rev: return "RE-ENTER", "VIX Reversal", curr_p, state
        if rebound >= 0.02: return "RE-ENTER", "2% Rebound", curr_p, state
        return "WAIT", f"Waiting({rebound*100:.1f}%)", curr_p, state

# ==========================================
# 4. 트레이딩 실행 (시간: 21일 00시 테스트 대응)
# ==========================================
async def run_trading():
    now_kst = dt.now(KST)
    current_hour = now_kst.hour
    
    trader = KIS_Trader()
    token_val = os.getenv('TELEGRAM_TOKEN'); chat_id = os.getenv('CHAT_ID')
    bot = Bot(token=token_val) if (Bot and token_val) else None
    
    # [민환님 가이드] 현재 시간 00시이므로 테스트를 위해 0으로 고정
    if current_hour == 0: 
        spy_ohlc, monthly, vix_close, msg = get_market_data()
        if spy_ohlc.empty:
            if bot: await bot.send_message(chat_id=chat_id, text=f"⚠️ 데이터 로드 실패: {msg}")
            return
        
        signal, reason, price_val, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
        bal = trader.get_balance()
        
        # [현재가 스캔]
        cur_p = 0.0
        for ex in ["AMS", "NYS", "NAS"]:
            url_p = f"{trader.base_url}/uapi/overseas-price/v1/quotations/price"
            hd = {"Content-Type":"application/json", "authorization":f"Bearer {trader.token}", "appkey":trader.app_key, "appsecret":trader.app_secret, "tr_id":"HHDFS00000300", "custtype":"P"}
            res_p = requests.get(url_p, headers=hd, params={"AUTH":"", "EXCD":ex, "PDNO":TRADE_TICKER})
            if res_p.status_code == 200:
                dt_p = res_p.json()
                last_p = dt_p.get('output', {}).get('last', '0')
                if bot: await bot.send_message(chat_id=chat_id, text=f"DEBUG {ex}: rt={dt_p.get('rt_cd')} last={last_p}")
                if dt_p.get('rt_cd') == '0' and float(last_p) > 0:
                    cur_p = float(last_p); break
        
        qty = trader.get_holdings(TRADE_TICKER)
        exec_status = ""
        
        if signal in ["KEEP", "RE-ENTER"] and qty == 0:
            if cur_p > 0:
                buy_qty = int((bal * 0.95) / cur_p)
                if buy_qty >= 1:
                    res = trader.send_order(TRADE_TICKER, buy_qty, "BUY")
                    if res.get('rt_cd') == '0':
                        exec_status = f" | ✅ 매수성공: {buy_qty}주"
                        with open(STATE_FILE, 'w') as f: json.dump({"in_market": True, "last_exit_price": 0}, f)
                    else: exec_status = f" | ❌ 매수실패: {res.get('rt_msg')}"
                else: exec_status = f" | ⚠️ 수량부족"
            else: exec_status = " | ⚠️ 가격조회불가"
        elif signal == "EXIT" and qty > 0:
            res = trader.send_order(TRADE_TICKER, qty, "SELL")
            if res.get('rt_cd') == '0':
                exec_status = f" | ✅ 매도성공: {qty}주"
                with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": price_val}, f)
            else: exec_status = f" | ❌ 매도실패: {res.get('rt_msg')}"

        if bot: await bot.send_message(chat_id=chat_id, text=f"[{now_kst.strftime('%H:%M')}] {signal}: {reason}{exec_status}\nqty={qty}|bal={bal:.1f}|p={cur_p:.2f}")

    elif current_hour == 1:
        spy_int = yf.download(SIGNAL_TICKER, period='1d', interval='5m', progress=False)
        if not spy_int.empty:
            spy_ret = (float(spy_int['Close'].iloc[-1]) / float(spy_int['Open'].iloc[0])) - 1
            if spy_ret <= -0.03:
                qty = trader.get_holdings(TRADE_TICKER)
                if qty > 0:
                    trader.send_order(TRADE_TICKER, qty, "SELL")
                    with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": float(spy_int['Close'].iloc[-1])}, f)
                    if bot: await bot.send_message(chat_id=chat_id, text="🚨 [01:00 긴급] 전량 매도 완료")

# ==========================================
# 5. 스트림릿 대시보드 (통합 시각화)
# ==========================================
def run_dashboard():
    now_kst = dt.now(KST)
    st.set_page_config(page_title="SP500 Watchtower v3.3.3", layout="wide")
    st.sidebar.title("v3.3.3 Master")
    st.sidebar.caption(f"Update: {now_kst.strftime('%H:%M:%S')} KST")
    st.sidebar.divider()
    st.sidebar.write("**EXIT:** VIX+30%, SPY-3%, 3d-5%, 2m Down")
    st.sidebar.write("**ENTER:** VIX Reversal, +2% Rebound")

    st.title(f"🛡️ {TRADE_TICKER} Watchtower")
    spy_ohlc, monthly, vix_close, msg = get_market_data()
    if spy_ohlc.empty: st.error(f"❌ 데이터 로드 실패: {msg}"); return
    signal, reason, price, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Position", "IN" if state.get('in_market') else "OUT")
    with c2: st.metric("Signal", signal)
    with c3: st.metric("SPY Price", f"${price:.2f}")
    with c4: st.metric("Daily Ret", f"{((spy_ohlc['Close'].iloc[-1]/spy_ohlc['Close'].iloc[-2])-1)*100:+.2f}%")
    with c5: st.metric("VIX Value", f"{vix_close.iloc[-1]:.2f}")

    if signal == "KEEP": st.success(f"[OK] {reason}")
    elif signal == "EXIT": st.error(f"[EMERGENCY] {reason}")
    else: st.info(f"[INFO] {reason}")

    common_idx = spy_ohlc.index.intersection(vix_close.index)
    ohlc_p, vix_p = spy_ohlc.loc[common_idx].tail(126), vix_close.loc[common_idx].tail(126)
    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=ohlc_p.index, open=ohlc_p['Open'], high=ohlc_p['High'], low=ohlc_p['Low'], close=ohlc_p['Close'], name='SPY'), row=1, col=1)
    fig.add_trace(go.Bar(x=vix_p.index, y=vix_p.values, name='VIX', marker_color='orange'), row=2, col=1)
    fig.add_trace(go.Bar(x=ohlc_p.index, y=ohlc_p['Volume'], name='Vol', marker_color='blue'), row=3, col=1)
    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10,r=10,t=10,b=10), showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Performance Analysis (2022-2026)")
    bt_sp500 = [-0.053,-0.030,0.035,-0.087,-0.006,-0.082,0.092,-0.041,-0.094,0.079,0.054,-0.058,0.062,-0.025,0.035,0.015,-0.001,0.065,0.031,-0.017,-0.048,-0.022,0.087,0.044,0.016,0.052,0.031,-0.041,0.048,0.035,0.011,0.024,0.022,-0.009,0.057,-0.024,-0.012,-0.018,-0.058,-0.082,0.065,0.038,0.042,0.018,0.025,0.031,0.044,0.019,0.008,-0.021,-0.048,0.092]
    dates = [(dt(2022,1,1) + timedelta(days=31*i)).strftime('%y-%m') for i in range(len(bt_sp500))]
    m_fig = go.Figure(go.Bar(x=dates, y=[v*100 for v in bt_sp500], marker_color=['#3fb950' if v > 0 else '#f85149' for v in bt_sp500]))
    m_fig.update_layout(template='plotly_dark', height=250, margin=dict(l=10,r=10,t=10,b=10), title="Historical Monthly Returns (%)")
    st.plotly_chart(m_fig, use_container_width=True)

    st_hist, bh_hist = [100.0], [100.0]
    in_m, c_d, cap_st, cap_bh, spy_p, last_ex_p = True, 0, 100.0, 100.0, 100.0, 100.0
    for r in bt_sp500:
        spy_p *= (1 + r); cap_bh *= (1 + r); bh_hist.append(cap_bh)
        if in_m:
            if r < 0: c_d += 1
            else: c_d = 0
            if c_d >= 2: in_m, last_ex_p, ret_st = False, spy_p, 0
            else: ret_st = r * 3 - 0.001
        else:
            rebound = (spy_p - last_ex_p) / last_ex_p
            if rebound >= 0.02: in_m, c_d, ret_st = True, 0, r * 3 - 0.001
            else: ret_st = 0
        cap_st *= (1 + ret_st); st_hist.append(cap_st)

    c_fig = go.Figure()
    c_fig.add_trace(go.Scatter(x=['22-01']+dates, y=st_hist, name='Strategy', line=dict(color='#3fb950', width=2)))
    c_fig.add_trace(go.Scatter(x=['22-01']+dates, y=bh_hist, name='SPY B&H', line=dict(color='gray', dash='dash')))
    c_fig.update_layout(template='plotly_dark', height=300, margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Manwon (Start: 100)")
    st.plotly_chart(c_fig, use_container_width=True)

    with st.expander("Strategy Performance"):
        st.write(f"### 📈 Total Return: {(st_hist[-1]-100):.1f}%")

    if os.path.exists(HISTORY_FILE):
        st.subheader("📋 History Logs")
        st.dataframe(pd.read_csv(HISTORY_FILE), use_container_width=True, hide_index=True)

# ==========================================
# 6. 진입점
# ==========================================
if os.getenv('GITHUB_ACTIONS') == 'true':
    asyncio.run(run_trading())
else:
    run_dashboard()
