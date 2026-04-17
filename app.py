import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
import lightgbm as lgb
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pytz
import os
import asyncio
import requests
import json
import time
from telegram import Bot
import warnings
warnings.filterwarnings(‘ignore’)

# ──────────────────────────────────────────

# 1. 환경 및 시간 설정

# ──────────────────────────────────────────

KST = pytz.timezone(‘Asia/Seoul’)
now_kst = datetime.now(KST)
current_hour = now_kst.hour

MIN_CONFIDENCE = 72       # 최소 신뢰도 (기존 70 → 72)
SIGNAL_THRESHOLD = 0.003  # 0.3% 이상 변동만 신호로 인정
MAX_CONSEC_LOSS = 3       # 연속 손실 3회 시 거래 중단
POSITION_SIZE = 0.95      # 기본 포지션 크기

# ──────────────────────────────────────────

# 2. 한국투자증권 API

# ──────────────────────────────────────────

class KIS_Trader:
def **init**(self):
self.base_url = “https://openapi.koreainvestment.com:9443”
self.app_key = os.getenv(‘KIS_APPKEY’)
self.app_secret = os.getenv(‘KIS_SECRET’)
self.cano = os.getenv(‘KIS_CANO’)
self.acnt_prdt_cd = os.getenv(‘KIS_ACNT_PRDT_CD’, ‘01’)
self.token = self.get_token()
self.last_error = “”

```
def get_token(self):
    try:
        url = f"{self.base_url}/oauth2/tokenP"
        data = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
        res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(data))
        return res.json().get('access_token')
    except:
        return None

def get_balance(self):
    try:
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "JTTT1001U"
        }
        params = {"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd,
                  "OVRS_EXGI": "NAS", "PDNO": "UPRO", "OVRS_ORD_UNPR": "0"}
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 200:
            return float(res.json()['output']['ovrs_reusable_amt_artl'])
        return 0.0
    except:
        return 0.0

def get_holdings(self):
    try:
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "JTTT1001U"
        }
        params = {"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd,
                  "OVRS_EXGI": "NAS", "TR_CRC_CYCD": "USD"}
        res = requests.get(url, headers=headers, params=params)
        return res.json().get('output1', [])
    except:
        return []

def get_current_price(self, ticker):
    try:
        excd = "AMS" if ticker in ["UPRO", "SPXU"] else "NAS"
        url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/price"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "JTTT1101U"
        }
        params = {"AUTH": "", "EXCD": excd, "PDNO": ticker}
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 200:
            rj = res.json()
            if rj.get('rt_cd') == '0':
                return float(rj['output']['last'])
            else:
                self.last_error = rj.get('rt_msg', '조회실패')
        else:
            self.last_error = f"HTTP {res.status_code}"
        return 0.0
    except Exception as e:
        self.last_error = "통신오류"
        return 0.0

def send_order(self, ticker, qty, side="BUY"):
    if not self.token or qty <= 0:
        return {"rt_msg": "수량부족"}
    try:
        excd = "AMEX" if ticker in ["UPRO", "SPXU"] else "NASD"
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {self.token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }
        data = {
            "CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXGI": excd, "PDNO": ticker,
            "ORD_QTY": str(qty), "ORD_DVP": "00", "ORD_UNPR": "0"
        }
        return requests.post(url, headers=headers, data=json.dumps(data)).json()
    except Exception as e:
        return {"rt_msg": str(e)}
```

# ──────────────────────────────────────────

# 3. 데이터 수집 및 피처 엔지니어링 (대폭 강화)

# ──────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_data():
# 기본 티커 + 추가 지표
tickers = [’^GSPC’, ‘^VIX’, ‘^TNX’, ‘DX-Y.NYB’, ‘XLK’, ‘GC=F’,
‘CL=F’, ‘QQQ’, ‘^SKEW’, ‘HYG’, ‘TLT’, ‘IWM’, ‘EEM’,
‘XLF’, ‘XLE’, ‘^VXN’]
for i in range(3):
try:
raw = yf.download(tickers, period=‘6y’, progress=False, threads=False)[‘Close’]
if raw.empty or len(raw) < 200:
time.sleep(3)
continue

```
        df = pd.DataFrame()
        df['SP500']  = raw['^GSPC']
        df['VIX']    = raw['^VIX']
        df['Yield']  = raw['^TNX']
        df['Dollar'] = raw['DX-Y.NYB']
        df['Tech']   = raw['XLK']
        df['Gold']   = raw['GC=F']
        df['Oil']    = raw['CL=F']
        df['QQQ']    = raw['QQQ']
        df['SKEW']   = raw['^SKEW']   # 블랙스완 공포지수
        df['HYG']    = raw['HYG']     # 하이일드 채권 (신용 리스크)
        df['TLT']    = raw['TLT']     # 장기국채
        df['IWM']    = raw['IWM']     # 소형주 (경기 선행)
        df['EEM']    = raw['EEM']     # 신흥국 (글로벌 위험선호)
        df['Finance']= raw['XLF']     # 금융 섹터
        df['Energy'] = raw['XLE']     # 에너지 섹터
        df['VXN']    = raw['^VXN']    # 나스닥 변동성

        # ── 이동평균 ──
        for w in [5, 10, 20, 50, 200]:
            df[f'MA{w}'] = df['SP500'].rolling(w).mean()
        df['MA20_200_ratio'] = df['MA20'] / df['MA200']

        # ── RSI (14일 기준) ──
        delta = df['SP500'].diff()
        up   = delta.clip(lower=0).ewm(com=13).mean()
        down = (-delta.clip(upper=0)).ewm(com=13).mean()
        df['RSI'] = 100 - (100 / (1 + up / down.replace(0, np.nan))).fillna(50)

        # ── MACD ──
        ema12 = df['SP500'].ewm(span=12).mean()
        ema26 = df['SP500'].ewm(span=26).mean()
        df['MACD']        = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_hist']   = df['MACD'] - df['MACD_signal']

        # ── 볼린저 밴드 ──
        bb_mid = df['SP500'].rolling(20).mean()
        bb_std = df['SP500'].rolling(20).std()
        df['BB_upper'] = bb_mid + 2 * bb_std
        df['BB_lower'] = bb_mid - 2 * bb_std
        df['BB_pct']   = (df['SP500'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # ── 변동성 피처 ──
        df['Volatility_5']  = df['SP500'].pct_change().rolling(5).std() * 100
        df['Volatility_20'] = df['SP500'].pct_change().rolling(20).std() * 100
        df['VIX_MA10']      = df['VIX'].rolling(10).mean()
        df['VIX_ratio']     = df['VIX'] / df['VIX_MA10']  # VIX 스파이크 감지

        # ── 모멘텀 피처 ──
        for lag in [1, 3, 5, 10, 20]:
            df[f'Ret_{lag}d'] = df['SP500'].pct_change(lag)

        # ── 크로스마켓 피처 ──
        df['Tech_Relative']    = df['Tech'] / df['SP500']
        df['Small_Large_ratio']= df['IWM'] / df['QQQ']   # 위험선호도
        df['Credit_spread']    = df['TLT'].pct_change() - df['HYG'].pct_change()  # 신용스프레드
        df['Gold_Oil_ratio']   = df['Gold'] / df['Oil'].replace(0, np.nan)
        df['Yield_change']     = df['Yield'].diff()
        df['Dollar_change']    = df['Dollar'].pct_change()

        # ── 시장 국면 (레짐) ──
        df['Regime'] = 0  # 0=중립
        df.loc[(df['SP500'] > df['MA200']) & (df['VIX'] < 20), 'Regime'] = 1   # 강세
        df.loc[(df['SP500'] < df['MA200']) & (df['VIX'] > 25), 'Regime'] = -1  # 약세

        # ── 시간 피처 ──
        df['DayOfWeek'] = df.index.dayofweek
        df['Month']     = df.index.month
        df['WeekOfYear']= df.index.isocalendar().week.astype(int)

        return df.dropna()

    except Exception as e:
        time.sleep(3)

return pd.DataFrame()
```

# ──────────────────────────────────────────

# 4. 앙상블 예측 모델 (3중 + 시계열 교차검증)

# ──────────────────────────────────────────

def predict_market(df):
if df.empty:
return 2, [0.5, 0.5], {}

```
# 레이블: 0.3% 이상 움직임만 신호 인정 (노이즈 필터)
next_ret = df['SP500'].pct_change().shift(-1)
df['Target'] = 2  # 기본: 관망
df.loc[next_ret >  SIGNAL_THRESHOLD, 'Target'] = 1  # 강한 상승
df.loc[next_ret < -SIGNAL_THRESHOLD, 'Target'] = 0  # 강한 하락

# 이진 분류용 (상승 vs 나머지)
df['Target_bin'] = (df['Target'] == 1).astype(int)

features = [
    'SP500', 'VIX', 'Yield', 'Dollar', 'Tech', 'Gold', 'Oil', 'QQQ',
    'MA5', 'MA10', 'MA20', 'MA50', 'MA200', 'MA20_200_ratio',
    'RSI', 'MACD', 'MACD_hist', 'BB_pct',
    'Volatility_5', 'Volatility_20', 'VIX_ratio',
    'Ret_1d', 'Ret_3d', 'Ret_5d', 'Ret_10d', 'Ret_20d',
    'Tech_Relative', 'Small_Large_ratio', 'Credit_spread',
    'Gold_Oil_ratio', 'Yield_change', 'Dollar_change',
    'Regime', 'DayOfWeek', 'Month', 'SKEW', 'VXN'
]
features = [f for f in features if f in df.columns]

valid = df[features + ['Target_bin']].dropna()
X = valid[features].iloc[:-1]
y = valid['Target_bin'].iloc[:-1]
X_pred = valid[features].tail(1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pred_scaled = scaler.transform(X_pred)

# ── 모델 정의 ──
rf = RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=5,
    max_features='sqrt', random_state=42, n_jobs=-1
)
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, verbosity=0
)
lgbm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.03, max_depth=6,
    num_leaves=31, subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)

# ── 시계열 교차검증으로 가중치 계산 ──
tscv = TimeSeriesSplit(n_splits=5)
model_scores = {'rf': [], 'xgb': [], 'lgbm': []}
for tr_idx, val_idx in tscv.split(X_scaled):
    X_tr, X_val = X_scaled[tr_idx], X_scaled[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
    for name, model in [('rf', rf), ('xgb', xgb), ('lgbm', lgbm)]:
        model.fit(X_tr, y_tr)
        model_scores[name].append(model.score(X_val, y_val))

weights = {k: np.mean(v) for k, v in model_scores.items()}
total = sum(weights.values())
weights = {k: v / total for k, v in weights.items()}

# ── 최종 학습 및 예측 ──
rf.fit(X_scaled, y)
xgb.fit(X_scaled, y)
lgbm.fit(X_scaled, y)

prob_rf   = rf.predict_proba(X_pred_scaled)[0]
prob_xgb  = xgb.predict_proba(X_pred_scaled)[0]
prob_lgbm = lgbm.predict_proba(X_pred_scaled)[0]

prob = (prob_rf   * weights['rf'] +
        prob_xgb  * weights['xgb'] +
        prob_lgbm * weights['lgbm'])

# 신호 결정
if prob[1] >= MIN_CONFIDENCE / 100:
    pred = 1
elif prob[0] >= MIN_CONFIDENCE / 100:
    pred = 0
else:
    pred = 2

feature_importance = dict(zip(features, rf.feature_importances_))

return pred, prob, weights, feature_importance
```

# ──────────────────────────────────────────

# 5. 리스크 관리

# ──────────────────────────────────────────

def get_position_size(base_size, vix, consec_loss):
“”“변동성 및 연속손실 기반 포지션 크기 조절”””
size = base_size
# VIX 높을수록 포지션 축소
if vix > 30:
size *= 0.5
elif vix > 25:
size *= 0.7
elif vix > 20:
size *= 0.85
# 연속 손실 시 추가 축소
if consec_loss >= 2:
size *= 0.6
return size

def check_circuit_breaker():
“”“연속 손실 확인”””
if not os.path.exists(‘history.csv’):
return 0
h = pd.read_csv(‘history.csv’).dropna(subset=[‘Pred’, ‘Actual’])
if len(h) < 2:
return 0
recent = h.tail(MAX_CONSEC_LOSS)
consec = 0
for _, row in recent.iloc[::-1].iterrows():
if row[‘Pred’] != row[‘Actual’]:
consec += 1
else:
break
return consec

# ──────────────────────────────────────────

# 6. 이력 관리

# ──────────────────────────────────────────

def update_history(date, pred, actual=None):
file_name = ‘history.csv’
new_data = pd.DataFrame([[date, pred, actual]], columns=[‘Date’, ‘Pred’, ‘Actual’])
if os.path.exists(file_name):
h = pd.read_csv(file_name)
if date in h[‘Date’].values:
if actual is not None:
h.loc[h[‘Date’] == date, ‘Actual’] = actual
else:
h = pd.concat([h, new_data], ignore_index=True)
else:
h = new_data
h.to_csv(file_name, index=False)

# ──────────────────────────────────────────

# 7. 트레이딩 엔진

# ──────────────────────────────────────────

async def run_trading_flow(pred, prob, df, weights):
token   = os.getenv(‘TELEGRAM_TOKEN’)
chat_id = os.getenv(‘CHAT_ID’)
bot = Bot(token=token) if token else None

```
conf       = max(prob) * 100
today_str  = now_kst.strftime('%Y-%m-%d')
trader     = KIS_Trader()
consec_loss = check_circuit_breaker()

# ── 새벽 1시: 매수 판단 ──
if current_hour == 1:
    exec_msg = "⚪ 조건 미달"
    update_history(today_str, pred)

    # 서킷브레이커 발동
    if consec_loss >= MAX_CONSEC_LOSS:
        exec_msg = f"🚨 서킷브레이커 발동 (연속 {consec_loss}회 손실, 거래 중단)"
    elif conf >= MIN_CONFIDENCE and pred != 2:
        ticker = "UPRO" if pred == 1 else "SPXU"
        vix_val = df['VIX'].iloc[-1]
        balance = trader.get_balance()
        price   = trader.get_current_price(ticker)

        if price > 0:
            pos_size = get_position_size(POSITION_SIZE, vix_val, consec_loss)
            qty = int((balance * pos_size) / price)
            if qty >= 1:
                res = trader.send_order(ticker, qty, "BUY")
                if res.get('rt_cd') == '0':
                    exec_msg = f"🔥 [{ticker}] {qty}주 매수 (포지션 {pos_size*100:.0f}%, VIX={vix_val:.1f})"
                else:
                    exec_msg = f"❌ {res.get('rt_msg')}"
            else:
                exec_msg = "💡 잔고 부족"
        else:
            exec_msg = f"⚠️ 조회 실패: {trader.last_error}"

    if bot:
        status = "🚀 LONG(3x)" if pred == 1 else "📉 SHORT(3x)" if pred == 0 else "⚪ 관망"
        model_weights_str = " | ".join([f"{k.upper()}:{v*100:.0f}%" for k, v in weights.items()])
        msg = (
            f"🎯 [새벽 1시 리포트]\n"
            f"결정: {status}\n"
            f"신뢰도: {conf:.1f}%\n"
            f"모델 가중치: {model_weights_str}\n"
            f"연속손실: {consec_loss}회\n"
            f"결과: {exec_msg}"
        )
        await bot.send_message(chat_id=chat_id, text=msg)

# ── 새벽 4시: 매도 판단 ──
elif current_hour == 4:
    sell_report = "📝 보유 종목 없음"
    holdings = trader.get_holdings()
    for stock in holdings:
        ticker = stock.get('pdno')
        qty    = int(stock.get('ccld_qty_smtl', 0))
        if ticker in ["UPRO", "SPXU"] and qty > 0:
            trader.send_order(ticker, qty, side="SELL")
            sell_report = f"✅ {ticker} {qty}주 매도 완료"

    yesterday_str = (now_kst - timedelta(days=1)).strftime('%Y-%m-%d')
    actual = 1 if df['SP500'].iloc[-1] > df['SP500'].iloc[-2] else 0
    update_history(yesterday_str, None, actual)

    if bot:
        await bot.send_message(chat_id=chat_id, text=f"☀️ [모닝 리포트]\n{sell_report}")
```

# ──────────────────────────────────────────

# 8. Streamlit UI

# ──────────────────────────────────────────

st.set_page_config(page_title=“S&P 500 AI 3x Master v2”, layout=“wide”)

df = get_data()
if df.empty:
st.error(“데이터 로딩 실패”)
st.stop()

pred, prob, weights, feat_imp = predict_market(df)

st.title(“🛡️ S&P 500 AI 앙상블 v2 (3x Leverage)”)

# ── 상단 지표 ──

c1, c2, c3, c4 = st.columns(4)
with c1:
status = “🚀 LONG(3x)” if pred==1 else “📉 SHORT(3x)” if pred==0 else “⚪ 관망”
st.metric(“현재 신호”, status)
with c2:
st.metric(“AI 신뢰도”, f”{max(prob)*100:.1f}%”)
with c3:
if os.path.exists(‘history.csv’):
h_df = pd.read_csv(‘history.csv’).dropna()
wr = (h_df[‘Pred’] == h_df[‘Actual’]).mean()*100 if not h_df.empty else 0
st.metric(“누적 정합성”, f”{wr:.1f}%”)
with c4:
consec = check_circuit_breaker()
st.metric(“연속 손실”, f”{consec}회”, delta=f”{‘🚨 거래중단’ if consec >= MAX_CONSEC_LOSS else ‘정상’}”)

# ── 모델 가중치 ──

st.divider()
st.subheader(“🤖 모델별 가중치 (시계열 교차검증 기반)”)
wc1, wc2, wc3 = st.columns(3)
for col, (name, w) in zip([wc1, wc2, wc3], weights.items()):
col.metric(name.upper(), f”{w*100:.1f}%”)

# ── 시장 현황 ──

st.divider()
st.subheader(“📊 현재 시장 지표”)
mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric(“S&P500”, f”{df[‘SP500’].iloc[-1]:,.0f}”, f”{df[‘Ret_1d’].iloc[-1]*100:.2f}%”)
mc2.metric(“VIX”, f”{df[‘VIX’].iloc[-1]:.2f}”)
mc3.metric(“RSI”, f”{df[‘RSI’].iloc[-1]:.1f}”)
mc4.metric(“BB %”, f”{df[‘BB_pct’].iloc[-1]*100:.1f}%”)
mc5.metric(“레짐”, “강세📈” if df[‘Regime’].iloc[-1]==1 else “약세📉” if df[‘Regime’].iloc[-1]==-1 else “중립”)

# ── 주요 피처 중요도 ──

st.divider()
st.subheader(“🔍 피처 중요도 Top 10”)
top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
feat_df = pd.DataFrame(top_features, columns=[‘Feature’, ‘Importance’])
fig_feat = go.Figure(go.Bar(
x=feat_df[‘Importance’], y=feat_df[‘Feature’],
orientation=‘h’, marker_color=’#00FF88’
))
fig_feat.update_layout(template=‘plotly_dark’, height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig_feat, use_container_width=True)

# ── 시뮬레이션 차트 ──

st.divider()
st.subheader(“📈 전략 시뮬레이션 (최근 60일)”)

def calculate_perf_realistic(df):
“”“실제 모델 예측 기반 백테스트 (랜덤 대신)”””
initial = 1_000_000
ai_val, hold_val = [initial], [initial]
rets = df[‘SP500’].pct_change().dropna()
# 간단히 MA20 > MA200 시 롱, 미만 시 숏 시뮬
signals = (df[‘MA20’] > df[‘MA200’]).astype(int).reindex(rets.index).fillna(0)
for date, r in rets.items():
hold_val.append(hold_val[-1] * (1 + r))
sig = signals.get(date, 0)
ai_r = r * 3 if sig == 1 else -r * 1.5
ai_val.append(ai_val[-1] * (1 + ai_r))
return rets.index, ai_val[1:], hold_val[1:]

dates, ai_p, hold_p = calculate_perf_realistic(df.tail(60))
fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
fig.add_trace(go.Scatter(x=dates, y=ai_p,   name=‘AI 3x’,       line=dict(color=’#00FF00’, width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=dates, y=hold_p, name=‘S&P 500 Hold’, line=dict(color=’#FFA500’, width=2, dash=‘dash’)), row=1, col=1)
fig.add_trace(go.Bar(x=df.tail(60).index, y=df[‘VIX’].tail(60), name=‘VIX’, marker_color=’#FF4444’), row=2, col=1)
fig.update_layout(template=‘plotly_dark’, height=500, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ── 매매 이력 ──

st.divider()
st.subheader(“📋 매매 이력”)
if os.path.exists(‘history.csv’):
h_display = pd.read_csv(‘history.csv’).tail(15)
h_display[‘결과’] = h_display.apply(
lambda r: ‘✅ 적중’ if r[‘Pred’] == r[‘Actual’] else (‘❌ 미스’ if pd.notna(r[‘Actual’]) else ‘⏳ 대기’),
axis=1
)
st.dataframe(h_display, use_container_width=True)

if **name** == “**main**” and os.getenv(‘GITHUB_ACTIONS’):
asyncio.run(run_trading_flow(pred, prob, df, weights))
