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
# 1. 공정 설정 (마스터 사양 보존)
# ==========================================
KST = pytz.timezone('Asia/Seoul')
SIGNAL_TICKER = 'SPY' 
TRADE_TICKER = 'UPRO'
STATE_FILE = 'trend_state.json'
HISTORY_FILE = 'history_trend.csv'

# ==========================================
# 2. KIS API 클래스 (민환님 검증 규격으로 원복)
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
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT3007R", "custtype":"P"}
            params = {"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd, "OVRS_EXCG_CD": "AMEX", "OVRS_ORD_UNPR": "1", "ITEM_CD": TRADE_TICKER}
            res = requests.get(url, headers=headers, params=params).json()
            return float(res.get('output', {}).get('ord_psbl_frcr_amt', 0))
        except: return 0.0

    def get_holdings(self, ticker=TRADE_TICKER):
        if not self.token: return 0
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":"JTTT3012R", "custtype":"P"}
            params = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXCG_CD":"AMEX", "TR_CRCY_CD":"USD", "CTX_AREA_FK200":"", "CTX_AREA_NK200":""}
            res = requests.get(url, headers=headers, params=params)
            for item in res.json().get('output1', []):
                if item.get('pdno') == ticker: return int(float(item.get('ccld_qty_smtl', 0)))
            return 0
        except: return 0

    def get_current_price(self, ticker=TRADE_TICKER):
        try:
            df = yf.download(ticker, period='1d', interval='1m', progress=False)
            if not df.empty: return float(df['Close'].iloc[-1])
            return 0.0
        except: return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token: return {"rt_cd": "1", "rt_msg": "No Token"}
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "TTTT1002U" if side == "BUY" else "TTTT1006U"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id, "custtype":"P"}
            
            clean_qty = str(int(float(qty)))
            
            # [규격 원복] 민환님 가이드: 시장가는 가격 "0" + ORD_DVSN "00" 조합이 가장 안전
            data = {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "OVRS_EXCG_CD": "AMEX",
                "PDNO": ticker,
                "ORD_QTY": clean_qty,
                "OVRS_ORD_UNPR": "0",         # 가격 0원 고정
                "ORD_SVR_DVSN_CD": "0",
                "ORD_DVSN": "00"              # 시장가(00)
            }
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except Exception as e: return {"rt_cd": "1", "rt_msg": str(e)}

# ==========================================
# 3. 데이터 엔진 (민환님표 2중 Flattening)
# ==========================================
def get_market_data():
    try:
        spy_raw = yf.download(SIGNAL_TICKER, period='2y', progress=False, auto_adjust=True, repair=True)
        vix_raw = yf.download('^VIX', period='2y', progress=False, auto_adjust=True, repair=True)
        if spy_raw.empty: return pd.DataFrame(), pd.Series(), pd.Series(), "Data Empty"
        
        # [1단계] MultiIndex 레벨 압착
        if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
        if isinstance(vix_raw.columns, pd.MultiIndex): vix_raw.columns = vix_raw.columns.get_level_values(0)
        
        # [2단계] 민환님 권장: 명시적 컬럼 선택 및 Copy로 인덱스 무결성 확보
        spy_raw = spy_raw[['Open','High','Low','Close','Volume']].copy()
        spy_close = spy_raw['Close'].squeeze()
        vix_close = vix_raw['Close'].squeeze()
        
        monthly = spy_close.resample('ME').last().pct_change().dropna()
        return spy_raw, monthly, vix_close, "Success"
    except Exception as e: return pd.DataFrame(), pd.Series(), pd.Series(), str(e)

def get_signal(spy_close, monthly, vix_close):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f: state = json.load(f)
    else: state = {"in_market": True, "last_exit_price": 0}
    if spy_close.empty or len(spy_close) < 20: return "WAIT", "Loading", 0.0, state
    
    curr_p = float(spy_close.iloc[-1])
    spy_daily_ret = float((spy_close.iloc[-1] / spy_close.iloc[-2]) - 1)
    vix_daily_ret = float((vix_close.iloc[-1] / vix_close.iloc[-2]) - 1)
    spy_3day_ret = float((spy_close.iloc[-1] / spy_close.iloc[-4]) - 1) if len(spy_close) >= 4 else 0.0
    
    if vix_daily_ret >= 0.3 or spy_daily_ret <= -0.03 or spy_3day_ret <= -0.05:
        return "EXIT", "Shock Trigger", curr_p, state
    
    if state.get('in_market', True):
        recent = monthly.tail(2).values
        if len(recent) == 2 and recent[0] < 0 and recent[1] < 0: return "EXIT", "2m Down", curr_p, state
        return "KEEP", "Holding", curr_p, state
    else:
        rebound = (curr_p - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        vix_now, vix_prev = float(vix_close.iloc[-1]), float(vix_close.iloc[-2])
        vix_20d = vix_close.tail(20)
        vix_mean, vix_std = float(vix_20d.mean()), float(vix_20d.std())
        vix_rev = (vix_now > (vix_mean + 2*vix_std) or vix_prev > (vix_mean + 2*vix_std)) and (vix_now < vix_prev * 0.95) and (spy_daily_ret > 0)
        
        if vix_rev: return "RE-ENTER", "VIX Reversal", curr_p, state
        if rebound >= 0.02: return "RE-ENTER", "2% Rebound", curr_p, state
        return "WAIT", f"Waiting({rebound*100:.1f}%)", curr_p, state

# ==========================================
# 4. 트레이딩 실행
# ==========================================
async def run_trading():
    now_kst = dt.now(KST); current_hour = now_kst.hour
    trader = KIS_Trader(); token_v = os.getenv('TELEGRAM_TOKEN'); chat_id = os.getenv('CHAT_ID')
    bot = Bot(token=token_v) if (Bot and token_v) else None
    
    if current_hour == 20: 
        spy_ohlc, monthly, vix_close, msg = get_market_data()
        if spy_ohlc.empty: return
        signal, reason, price_val, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
        bal = trader.get_balance(); cur_p = trader.get_current_price(TRADE_TICKER)
        qty = trader.get_holdings(TRADE_TICKER)
        
        exec_status = ""
        if signal in ["KEEP", "RE-ENTER"] and qty == 0 and cur_p > 0:
            buy_qty = int((bal * 0.95) / cur_p)
            if buy_qty >= 1:
                res_ord = trader.send_order(TRADE_TICKER, buy_qty, "BUY")
                if res_ord.get('rt_cd') == '0':
                    exec_status = f" | ✅ 매수성공: {buy_qty}주"
                    with open(STATE_FILE, 'w') as f: json.dump({"in_market": True, "last_exit_price": 0}, f)
                    pd.DataFrame([{"Date": now_kst.strftime("%Y-%m-%d %H:%M"), "Action": "BUY", "Qty": buy_qty, "Price": cur_p}]).to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)
                else: exec_status = f" | ❌ 매수실패: {str(res_ord)[:200]}"
        elif signal == "EXIT" and qty > 0:
            res_ord = trader.send_order(TRADE_TICKER, qty, "SELL")
            if res_ord.get('rt_cd') == '0':
                exec_status = f" | ✅ 매도성공: {qty}주"
                with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": price_val}, f)
                pd.DataFrame([{"Date": now_kst.strftime("%Y-%m-%d %H:%M"), "Action": "SELL", "Qty": qty, "Price": cur_p}]).to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)
            else: exec_status = f" | ❌ 매도실패: {str(res_ord)[:200]}"

        if bot: await bot.send_message(chat_id=chat_id, text=f"[{now_kst.strftime('%H:%M')}] {signal}: {reason}{exec_status}")

# ==========================================
# 5. 스트림릿 대시보드 (디버깅용 출력 추가)
# ==========================================
def run_dashboard():
    now_kst = dt.now(KST)
    st.set_page_config(page_title="SP500 Watchtower v3.6.9", layout="wide")
    
    st.sidebar.title("v3.6.9 Debug Master")
    st.sidebar.caption(f"Update: {now_kst.strftime('%H:%M:%S')} KST")
    
    spy_ohlc, monthly, vix_close, msg = get_market_data()
    
    # [민환님 요청] 스트림릿 디버깅용 임시 출력
    with st.expander("🛠️ 데이터 엔진 상태 점검 (Debug)", expanded=False):
        st.write(f"메시지: {msg}")
        st.write(f"SPY Shape: {spy_ohlc.shape}")
        st.write(f"컬럼 구조: {spy_ohlc.columns.tolist()}")
        if not spy_ohlc.empty:
            st.write("샘플 데이터:", spy_ohlc.tail(3))

    if spy_ohlc.empty: st.error("Data Load Fail"); return
    
    signal, reason, price, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
    
    # 5개 지표 카드
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Position", "IN" if state.get('in_market') else "OUT")
    with c2: st.metric("Signal", signal)
    with c3: st.metric("SPY Price", f"${price:.2f}")
    with c4: st.metric("Daily Ret", f"{((spy_ohlc['Close'].iloc[-1]/spy_ohlc['Close'].iloc[-2])-1)*100:+.2f}%")
    with c5: st.metric("VIX Value", f"{vix_close.iloc[-1]:.2f}")

    if signal == "KEEP": st.success(f"[OK] {reason}")
    elif signal == "EXIT": st.error(f"[EMERGENCY] {reason}")
    else: st.info(f"[INFO] {reason}")

    # 3단 통합 차트
    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=spy_ohlc.tail(126).index, open=spy_ohlc.tail(126)['Open'], high=spy_ohlc.tail(126)['High'], low=spy_ohlc.tail(126)['Low'], close=spy_ohlc.tail(126)['Close'], name='SPY'), row=1, col=1)
    fig.add_trace(go.Bar(x=vix_close.tail(126).index, y=vix_close.tail(126).values, name='VIX', marker_color='orange'), row=2, col=1)
    fig.add_trace(go.Bar(x=spy_ohlc.tail(126).index, y=spy_ohlc.tail(126)['Volume'], name='Vol', marker_color='blue'), row=3, col=1)
    fig.update_layout(template='plotly_dark', height=600, margin=dict(l=10,r=10,t=10,b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    bt_sp500 = [-0.053,-0.030,0.035,-0.087,-0.006,-0.082,0.092,-0.041,-0.094,0.079,0.054,-0.058,0.062,-0.025,0.035,0.015,-0.001,0.065,0.031,-0.017,-0.048,-0.022,0.087,0.044,0.016,0.052,0.031,-0.041,0.048,0.035,0.011,0.024,0.022,-0.009,0.057,-0.024,-0.012,-0.018,-0.058,-0.082,0.065,0.038,0.042,0.018,0.025,0.031,0.044,0.019,0.008,-0.021,-0.048,0.092]
    dates = [(dt(2022,1,1) + timedelta(days=31*i)).strftime('%y-%m') for i in range(len(bt_sp500))]
    
    # 월별 수익률 바 차트
    m_fig = go.Figure(go.Bar(x=dates, y=[v*100 for v in bt_sp500], marker_color=['#3fb950' if v > 0 else '#f85149' for v in bt_sp500]))
    m_fig.update_layout(template='plotly_dark', height=250, title="Monthly Returns (%)")
    st.plotly_chart(m_fig, use_container_width=True)

    # 수익률 곡선
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
    
    st.plotly_chart(go.Figure().add_trace(go.Scatter(x=['22-01']+dates, y=st_hist, name='Strategy', line=dict(color='#3fb950'))).add_trace(go.Scatter(x=['22-01']+dates, y=bh_hist, name='SPY B&H', line=dict(color='gray', dash='dash'))).update_layout(template='plotly_dark', height=300, yaxis_title="Equity (Base 100)"), use_container_width=True)

    if os.path.exists(HISTORY_FILE):
        st.subheader("📋 Trade Logs")
        st.dataframe(pd.read_csv(HISTORY_FILE).tail(10), use_container_width=True, hide_index=True)

# ==========================================
# 6. 진입점
# ==========================================
if os.getenv('GITHUB_ACTIONS') == 'true':
    asyncio.run(run_trading())
else:
    run_dashboard()
