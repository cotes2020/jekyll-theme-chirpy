---
title: Python으로 주식 데이터 수집 및 분석하기
date: 2026-03-01 10:00:00 +0900
categories: [데이터분석, 금융]
tags: [python, 주식, 데이터분석, pandas, finance]
---

## yfinance로 주식 데이터 수집

```python
import yfinance as yf
import pandas as pd

# 삼성전자 데이터 수집
ticker = yf.Ticker("005930.KS")
df = ticker.history(period="1y")

print(df.head())
```

## 이동평균선 계산

```python
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA60"] = df["Close"].rolling(window=60).mean()

# 골든크로스 신호
df["signal"] = 0
df.loc[df["MA20"] > df["MA60"], "signal"] = 1
df.loc[df["MA20"] < df["MA60"], "signal"] = -1
```

## RSI 계산

```python
def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df["RSI"] = calculate_rsi(df["Close"])
```

## 변동성 분석

```python
# 일간 수익률
df["returns"] = df["Close"].pct_change()

# 연간 변동성
annual_volatility = df["returns"].std() * (252 ** 0.5)
print(f"연간 변동성: {annual_volatility:.2%}")

# 샤프 비율 (무위험수익률 3% 가정)
risk_free_rate = 0.03
annual_return = df["returns"].mean() * 252
sharpe = (annual_return - risk_free_rate) / annual_volatility
print(f"샤프 비율: {sharpe:.2f}")
```

## PostgreSQL에 저장

```python
import asyncpg

async def save_to_db(df: pd.DataFrame):
    conn = await asyncpg.connect(DATABASE_URL)
    records = [
        (row.Index.date(), row.Open, row.High, row.Low, row.Close, row.Volume)
        for row in df.itertuples()
    ]
    await conn.executemany(
        "INSERT INTO stock_prices (date, open, high, low, close, volume) "
        "VALUES ($1, $2, $3, $4, $5, $6) ON CONFLICT DO NOTHING",
        records
    )
    await conn.close()
```

## 자동화 스케줄링

```python
import schedule
import time

def daily_update():
    # 매일 오후 4시 데이터 수집
    print("데이터 업데이트 중...")
    # ... 수집 로직

schedule.every().day.at("16:00").do(daily_update)

while True:
    schedule.run_pending()
    time.sleep(60)
```
