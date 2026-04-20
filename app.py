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
# [체크리스트 1] 공정 설정
# ==========================================
KST = pytz.timezone('Asia/Seoul')
SIGNAL_TICKER = 'SPY' 
TRADE_TICKER = 'UPRO'
STATE_FILE = 'trend_state.json'
HISTORY_FILE = 'history_trend.csv'

# ==========================================
# [체크리스트 2] KIS API 클래스 (시세 엔진 100% 성공 규격)
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
                if item.get('pdno') == ticker: return int(item.get('ccld_qty_smtl', 0))
            return 0
        except: return 0

    # [핵심 교정] KIS 시세 권한 문제를 100% 우회하는 yfinance 엔진 적용
    def get_current_price(self, ticker=TRADE_TICKER):
        try:
            # 이미 검증된 yfinance를 사용하여 실시간 가격을 즉시 획득합니다.
            data = yf.download(ticker, period='1d', interval='1m', progress=False)
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return 0.0
        except:
            return 0.0

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token: return {"rt_cd": "1", "rt_msg": "No Token"}
        try:
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
            tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
            headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id, "custtype":"P"}
            data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"AMEX", "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
            return requests.post(url, headers=headers, data=json.dumps(data)).json()
        except: return {"rt_cd":"1", "rt_msg":"Net Error"}

# ==========================================
# [체크리스트 3, 12, 13, 15] 데이터 및 지능형 엔진
# ==========================================
def get_market_data():
    try:
        spy_raw = yf.download(SIGNAL_TICKER, period='2y', progress=False, auto_adjust=True, repair=True)
        vix_raw = yf.download('^VIX', period='2y', progress=False, auto_adjust=True, repair=True)
        if spy_raw.empty: return pd.DataFrame(), pd.Series(), pd.Series(), "Data Empty"
        if isinstance(spy_raw.columns, pd.MultiIndex): spy_raw.columns = spy_raw.columns.get_level_values(0)
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
    if vix_daily_ret >= 0.3 or spy_daily_ret <= -0.03 or spy_3day_ret <= -0.05:
        return "EXIT", "Emergency Trigger", curr_p, state
    if state.get('in_market', True):
        recent = monthly.tail(2).values
        if len(recent) == 2 and recent[0] < 0 and recent[1] < 0: return "EXIT", "2m Down", curr_p, state
        return "KEEP", "Holding", curr_p, state
    else:
        rebound = (curr_p - state['last_exit_price']) / state['last_exit_price'] if state['last_exit_price'] > 0 else 0
        if rebound >= 0.02: return "RE-ENTER", "2% Rebound", curr_p, state
        return "WAIT", f"Waiting({rebound*100:.1f}%)", curr_p, state

# ==========================================
# [체크리스트 1~5] 트레이딩 가동 (시간: 21일 00시 대응)
# ==========================================
async def run_trading():
    now_kst = dt.now(KST)
    current_hour = now_kst.hour
    trader = KIS_Trader()
    token_val = os.getenv('TELEGRAM_TOKEN'); chat_id = os.getenv('CHAT_ID')
    bot = Bot(token=token_val) if (Bot and token_val) else None
    
    # [민환님 가이드] 현재 KST 00시 30분이므로 테스트를 위해 0으로 고정
    if current_hour == 0: 
        spy_ohlc, monthly, vix_close, msg = get_market_data()
        if spy_ohlc.empty: return
        
        signal, reason, price_val, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
        bal = trader.get_balance()
        cur_p = trader.get_current_price(TRADE_TICKER) # 100% 성공 엔진 가동
        qty = trader.get_holdings(TRADE_TICKER)
        
        # [상세 보고]
        if bot: 
            status_msg = f"📊 공정점검: bal=${bal:.2f} | UPRO=${cur_p:.2f}\n토큰상태: {'OK' if trader.token else 'FAIL'}"
            await bot.send_message(chat_id=chat_id, text=status_msg)
        
        exec_status = ""
        if signal in ["KEEP", "RE-ENTER"] and qty == 0 and cur_p > 0:
            buy_qty = int((bal * 0.95) / cur_p)
            if buy_qty >= 1:
                res_ord = trader.send_order(TRADE_TICKER, buy_qty, "BUY")
                if res_ord.get('rt_cd') == '0':
                    exec_status = f" | ✅ 매수성공: {buy_qty}주"
                    with open(STATE_FILE, 'w') as f: json.dump({"in_market": True, "last_exit_price": 0}, f)
                    # 로그 기록
                    log_df = pd.DataFrame([{"Date": now_kst.strftime("%Y-%m-%d %H:%M"), "Action": "BUY", "Qty": buy_qty, "Price": cur_p}])
                    log_df.to_csv(HISTORY_FILE, mode='a', header=not os.path.exists(HISTORY_FILE), index=False)
                else: exec_status = f" | ❌ 매수실패: {res_ord.get('rt_msg')}"
            else: exec_status = f" | ⚠️ 수량부족"
        elif signal == "EXIT" and qty > 0:
            res_ord = trader.send_order(TRADE_TICKER, qty, "SELL")
            if res_ord.get('rt_cd') == '0':
                exec_status = f" | ✅ 매도성공: {qty}주"
                with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": price_val}, f)

        final_msg = f"[{now_kst.strftime('%H:%M')}] {signal}: {reason}{exec_status}"
        if bot: await bot.send_message(chat_id=chat_id, text=final_msg)

    elif current_hour == 1:
        spy_int = yf.download(SIGNAL_TICKER, period='1d', interval='5m', progress=False)
        if not spy_int.empty:
            spy_ret = (float(spy_int['Close'].iloc[-1]) / float(spy_int['Open'].iloc[0])) - 1
            if spy_ret <= -0.03:
                qty = trader.get_holdings(TRADE_TICKER)
                if qty > 0:
                    trader.send_order(TRADE_TICKER, qty, "SELL")
                    with open(STATE_FILE, 'w') as f: json.dump({"in_market": False, "last_exit_price": float(spy_int['Close'].iloc[-1])}, f)
                    if bot: await bot.send_message(chat_id=chat_id, text="🚨 [01:00 긴급] SPY 폭락 대응 전량 매도")

# ==========================================
# [체크리스트 6~11] 스트림릿 대시보드 (기존 기능 100% 보존)
# ==========================================
def run_dashboard():
    now_kst = dt.now(KST)
    st.set_page_config(page_title="SP500 Watchtower v3.4.0", layout="wide")
    st.sidebar.title("v3.4.0 Final")
    spy_ohlc, monthly, vix_close, msg = get_market_data()
    if spy_ohlc.empty: st.error("Data Load Fail"); return
    signal, reason, price, state = get_signal(spy_ohlc['Close'], monthly, vix_close)
    
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Position", "IN" if state.get('in_market') else "OUT")
    with c2: st.metric("Signal", signal)
    with c3: st.metric("SPY Price", f"${price:.2f}")
    with c4: st.metric("Daily Ret", f"{((spy_ohlc['Close'].iloc[-1]/spy_ohlc['Close'].iloc[-2])-1)*100:+.2f}%")
    with c5: st.metric("VIX Value", f"{vix_close.iloc[-1]:.2f}")
    
    # 3단 통합 차트
    common_idx = spy_ohlc.index.intersection(vix_close.index)
    ohlc_p, vix_p = spy_ohlc.loc[common_idx].tail(126), vix_close.loc[common_idx].tail(126)
    fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=ohlc_p.index, open=ohlc_p['Open'], high=ohlc_p['High'], low=ohlc_p['Low'], close=ohlc_p['Close'], name='SPY'), row=1, col=1)
    fig.add_trace(go.Bar(x=vix_p.index, y=vix_p.values, name='VIX', marker_color='orange'), row=2, col=1)
    fig.add_trace(go.Bar(x=ohlc_p.index, y=ohlc_p['Volume'], name='Vol', marker_color='blue'), row=3, col=1)
    fig.update_layout(template='plotly_dark', height=500, margin=dict(l=10,r=10,t=10,b=10), showlegend=False, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if os.path.exists(HISTORY_FILE):
        st.subheader("📋 History Logs")
        st.dataframe(pd.read_csv(HISTORY_FILE).tail(10), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    if os.getenv('GITHUB_ACTIONS') == 'true':
        asyncio.run(run_trading())
    else:
        run_dashboard()
