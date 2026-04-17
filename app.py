import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import streamlit as st
import os
import asyncio
import requests
import json
from telegram import Bot
from datetime import datetime
import pytz

# 1. 환경 및 시간 설정
KST = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(KST)
current_hour = now_kst.hour

class KIS_Trader:
    def __init__(self):
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv('KIS_APPKEY')
        self.app_secret = os.getenv('KIS_SECRET')
        self.cano = os.getenv('KIS_CANO')
        self.acnt_prdt_cd = os.getenv('KIS_ACNT_PRDT_CD', '01')
        self.token = self.get_token()

    def get_token(self):
        try:
            url = f"{self.base_url}/oauth2/tokenP"
            data = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret}
            res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(data))
            return res.json().get('access_token')
        except: return None

    def get_balance(self):
        """해외주식 주문가능 달러(USD) 조회"""
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psbl-order"
        headers = {
            "Content-Type": "application/json", "authorization": f"Bearer {self.token}",
            "appkey": self.app_key, "appsecret": self.app_secret, "tr_id": "JTTT1001U"
        }
        params = {"CANO": self.cano, "ACNT_PRDT_CD": self.acnt_prdt_cd, "OVRS_EXGI": "NASD", "PDNO": "SPY", "OVRS_ORD_UNPR": "0"}
        res = requests.get(url, headers=headers, params=params)
        return float(res.json()['output']['ovrs_reusable_amt_artl'])

    def get_current_price(self, ticker):
        """해외주식 현재가 조회"""
        url = f"{self.base_url}/uapi/overseas-stock/v1/quotations/price"
        headers = {
            "Content-Type": "application/json", "authorization": f"Bearer {self.token}",
            "appkey": self.app_key, "appsecret": self.app_secret, "tr_id": "JTTT1101U"
        }
        params = {"AUTH": "", "EXCD": "NASD", "PDNO": ticker}
        res = requests.get(url, headers=headers, params=params)
        return float(res.json()['output']['last'])

    def send_order(self, ticker, qty, side="BUY"):
        if not self.token: return {"rt_msg": "토큰 발급 실패"}
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = "JTTT1002U" if side == "BUY" else "JTTT1006U"
        headers = {"Content-Type":"application/json", "authorization":f"Bearer {self.token}", "appkey":self.app_key, "appsecret":self.app_secret, "tr_id":tr_id}
        data = {"CANO":self.cano, "ACNT_PRDT_CD":self.acnt_prdt_cd, "OVRS_EXGI":"NASD", "PDNO":ticker, "ORD_QTY":str(qty), "ORD_DVP":"00", "ORD_UNPR":"0"}
        return requests.post(url, headers=headers, data=json.dumps(data)).json()

# --- 모델 및 예측 로직 생략 (기존 70% 임계값 유지) ---

async def execution_logic(pred, prob):
    trader = KIS_Trader()
    conf = max(prob) * 100
    
    if current_hour == 0 and conf >= 70:
        ticker = "SPY" if pred == 1 else "SH"
        
        try:
            # 실시간 잔고 및 현재가 조회
            usd_balance = trader.get_balance()
            curr_price = trader.get_current_price(ticker)
            
            # [변경점] 100만 원 고정이 아닌 '전체 잔고' 기반 수량 계산
            # 슬리피지와 수수료를 고려해 잔고의 95%만 사용합니다.
            qty = int((usd_balance * 0.95) / curr_price)
            
            if qty >= 1:
                res = trader.send_order(ticker, qty, "BUY")
                rt_msg = res.get('rt_msg', '응답 없음')
                return f"🔥 [풀-베팅 실행] {ticker} {qty}주 매수 주문! (잔고: ${usd_balance:.2f} 활용)"
            else:
                return f"💡 잔고 부족: 현재 ${usd_balance:.2f}로 {ticker} 1주를 살 수 없습니다."
        except Exception as e:
            return f"⚠️ 오류 발생: {e}"
            
    return f"💡 [대기] 신뢰도 {conf:.1f}%로 조건 미달"

# --- 메인 실행부 (텔레그램 연동 포함) ---
