"""
01_data_collection.py
KOSPI 유니버스 종목 마스터, 가격 데이터, 벤치마크 수집 → SQLite 저장
"""
import sqlite3
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"
START_DATE = (datetime.today() - timedelta(days=365*3)).strftime("%Y-%m-%d")
END_DATE = datetime.today().strftime("%Y-%m-%d")

UNIVERSE = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("207940", "삼성바이오로직스"),
    ("005380", "현대차"),
    ("068270", "셀트리온"),
    ("051910", "LG화학"),
    ("006400", "삼성SDI"),
    ("035420", "NAVER"),
    ("035720", "카카오"),
    ("000270", "기아"),
    ("105560", "KB금융"),
    ("055550", "신한지주"),
    ("096770", "SK이노베이션"),
    ("003550", "LG"),
    ("032830", "삼성생명"),
    ("086790", "하나금융지주"),
    ("003490", "대한항공"),
    ("018260", "삼성에스디에스"),
    ("011200", "HMM"),
    ("028260", "삼성물산"),
    ("066570", "LG전자"),
    ("017670", "SK텔레콤"),
    ("030200", "KT"),
    ("011170", "롯데케미칼"),
    ("034730", "SK"),
    ("024110", "기업은행"),
    ("000810", "삼성화재"),
    ("090430", "아모레퍼시픽"),
    ("010130", "고려아연"),
    ("047050", "포스코인터내셔널"),
    ("012330", "현대모비스"),
    ("051900", "LG생활건강"),
    ("009150", "삼성전기"),
    ("001040", "CJ"),
    ("003670", "포스코퓨처엠"),
    ("000100", "유한양행"),
    ("271560", "오리온"),
    ("097950", "CJ제일제당"),
    ("282330", "BGF리테일"),
    ("005490", "POSCO홀딩스"),
]


def get_db():
    return sqlite3.connect(DB_PATH)


def fetch_price_fdr(code, start, end, retries=3):
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader(code, start, end)
        if df is None or df.empty:
            return None
        df = df[["Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.columns = ["close", "volume"]
        df["code"] = code
        df["date"] = df.index.strftime("%Y-%m-%d")
        return df[["code", "date", "close", "volume"]].reset_index(drop=True)
    except Exception as e:
        print(f"  [FDR 오류] {code}: {e}")
        return None


def fetch_benchmark_naver(count=800):
    """NAVER 금융 KOSPI 지수 실제 데이터 수집"""
    import re
    try:
        url = f"https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&timeframe=day&count={count}&requestType=0"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible)"}, timeout=15)
        items = re.findall(r'<item data="([^"]+)"', r.text)
        records = []
        for item in items:
            parts = item.split("|")
            if len(parts) >= 5:
                ds = parts[0]
                close = float(parts[4])
                records.append({"date": f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}", "close": close})
        if records:
            return pd.DataFrame(records)
    except Exception as e:
        print(f"  [NAVER 벤치마크 오류]: {e}")
    return None


def main():
    print("=" * 60)
    print("01. 데이터 수집 시작")
    print(f"기간: {START_DATE} ~ {END_DATE}")
    print(f"유니버스: {len(UNIVERSE)}개 종목")
    print("=" * 60)

    con = get_db()

    # 종목 마스터
    master_df = pd.DataFrame(UNIVERSE, columns=["code", "name"])
    master_df["market"] = "KOSPI"
    master_df.to_sql("stock_master", con, if_exists="replace", index=False)
    print(f"[완료] stock_master: {len(master_df)}개 종목 저장")

    # 가격 데이터
    all_prices = []
    for code, name in UNIVERSE:
        print(f"  수집 중: {name} ({code})", end=" ")
        df = fetch_price_fdr(code, START_DATE, END_DATE)
        if df is not None and not df.empty:
            all_prices.append(df)
            print(f"→ {len(df)}일")
        else:
            print("→ 실패, 더미 데이터 생성")
            dates = pd.date_range(START_DATE, END_DATE, freq="B")
            np.random.seed(int(code) % 9999)
            prices = 50000 * np.cumprod(1 + np.random.normal(0.0003, 0.015, len(dates)))
            volumes = np.random.randint(100000, 5000000, len(dates))
            dummy = pd.DataFrame({
                "code": code,
                "date": dates.strftime("%Y-%m-%d"),
                "close": prices.astype(int),
                "volume": volumes
            })
            all_prices.append(dummy)
        time.sleep(0.3)

    if all_prices:
        price_df = pd.concat(all_prices, ignore_index=True)
        price_df.to_sql("price_data", con, if_exists="replace", index=False)
        print(f"[완료] price_data: {len(price_df)}행 저장")

    # 벤치마크 (KOSPI) — NAVER 금융 실제 데이터
    print("  벤치마크(KOSPI) 수집 중 (NAVER 금융)...", end=" ")
    bm = fetch_benchmark_naver(count=900)
    if bm is not None and not bm.empty:
        # 수집 기간 필터
        bm = bm[bm["date"] >= START_DATE]
        bm.to_sql("benchmark_price", con, if_exists="replace", index=False)
        print(f"→ {len(bm)}일 (실제 KOSPI, 최근: {bm.iloc[-1]['close']:.2f}pt)")
    else:
        print("→ 실패, 더미 데이터 생성")
        dates = pd.date_range(START_DATE, END_DATE, freq="B")
        np.random.seed(42)
        kospi = 2500 * np.cumprod(1 + np.random.normal(0.0002, 0.01, len(dates)))
        bm = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": kospi.astype(int)})
        bm.to_sql("benchmark_price", con, if_exists="replace", index=False)

    con.close()
    print("\n[01] 데이터 수집 완료\n")


if __name__ == "__main__":
    main()
