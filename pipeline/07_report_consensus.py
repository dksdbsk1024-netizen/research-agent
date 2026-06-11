"""
07_report_consensus.py
NAVER 금융에서 FnGuide 컨센서스 실제 데이터 수집
 - 목표주가 컨센서스 (애널리스트 평균)
 - 투자의견 점수 (4.0~5.0=매수, 3.0=중립, 1.0~2.0=매도)
 - 상승여력, 추정PER/EPS
실패 시 proxy로 fallback
"""
import sqlite3
import pandas as pd
import numpy as np
import requests
import re
import time
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://finance.naver.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}


def get_db():
    return sqlite3.connect(DB_PATH)


def minmax_score(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100


# ── 실제 컨센서스 수집 (NAVER 금융 / FnGuide) ────────────────────────────────

def fetch_naver_consensus(code, current_price):
    """
    NAVER 금융 coinfo 페이지에서 FnGuide 컨센서스 데이터 수집
    - 목표주가 컨센서스 (애널리스트 평균)
    - 투자의견 점수 및 텍스트
    - 추정 PER/EPS
    """
    url = f"https://finance.naver.com/item/coinfo.naver?code={code}&target=agree"
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        text = r.text

        # 목표주가 컨센서스
        tp_m = re.search(r'목표주가.*?<em>([\d,]+)</em>', text, re.DOTALL)
        target_price = int(tp_m.group(1).replace(",", "")) if tp_m else None

        # 투자의견 점수 (4.04매수, 3.5중립 등)
        op_m = re.search(r'<em>([\d\.]+)</em>(매수|매도|중립|강매수|강매도)', text)
        op_score = float(op_m.group(1)) if op_m else None
        op_text = op_m.group(2) if op_m else None

        # 추정PER, 추정EPS (컨센서스 기반)
        est_per_m = re.search(r'추정PER.*?<em[^>]*>([\d\.]+)</em>배', text, re.DOTALL)
        est_eps_m = re.search(r'추정PER.*?l\s*<em[^>]*>([\d,]+)</em>원', text, re.DOTALL)
        est_per = float(est_per_m.group(1)) if est_per_m else None
        est_eps = int(est_eps_m.group(1).replace(",", "")) if est_eps_m else None

        # 상승여력 계산
        upside = None
        if target_price and current_price and current_price > 0:
            upside = (target_price - current_price) / current_price * 100

        # 투자의견 → 감성점수 변환 (4.5~5=+1, 4~4.5=+0.7, 3~4=0, 2~3=-0.5, 1~2=-1)
        sentiment = 0.0
        if op_score is not None:
            if op_score >= 4.5:
                sentiment = 1.0
            elif op_score >= 4.0:
                sentiment = 0.7
            elif op_score >= 3.0:
                sentiment = 0.0
            elif op_score >= 2.0:
                sentiment = -0.5
            else:
                sentiment = -1.0

        return {
            "target_price": target_price,
            "opinion_score": op_score,
            "opinion": op_text,
            "est_per": est_per,
            "est_eps": est_eps,
            "upside_ratio": round(upside, 2) if upside is not None else None,
            "sentiment_score": sentiment,
            "data_type": "real" if target_price else "proxy",
        }

    except Exception as e:
        return {"data_type": "proxy", "error": str(e)}


def fetch_report_count_naver(code, days=365):
    """NAVER 모바일 API로 종목코드 기반 최근 N일 리포트 수 수집"""
    from datetime import datetime, timedelta
    cutoff = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")

    mobile_headers = {**HEADERS, "Referer": "https://m.stock.naver.com/"}
    url = f"https://m.stock.naver.com/api/research/stock/{code}"
    try:
        r = requests.get(url, headers=mobile_headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        count = sum(1 for item in data if item.get("writeDate", "") >= cutoff)
        return count if count > 0 else None
    except Exception:
        return None


# ── 전체 종목 수집 ────────────────────────────────────────────────────────────

def collect_all_consensus(score_df, price_df):
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["code", "date"])

    records = []
    success_count = 0

    for _, row in score_df.iterrows():
        code = row["code"]
        name = row.get("name", code)

        # 현재가
        sub = price_df[price_df["code"] == code].set_index("date").sort_index()
        current_price = float(sub["close"].iloc[-1]) if not sub.empty else 0

        print(f"  [{code}] {name} 수집 중...", end=" ")

        # 실제 컨센서스 수집
        cons = fetch_naver_consensus(code, current_price)
        time.sleep(0.5)

        # 리포트 수
        report_count = fetch_report_count_naver(code, days=365)
        time.sleep(0.3)

        data_type = cons.get("data_type", "proxy")
        target_price = cons.get("target_price")
        upside = cons.get("upside_ratio")

        if data_type == "real" and target_price:
            success_count += 1
            print(f"✅ 목표주가={target_price:,}원, 의견={cons.get('opinion','N/A')}({cons.get('opinion_score','N/A')}), "
                  f"상승여력={upside:.1f}%, 리포트={report_count}건")
        else:
            # proxy fallback
            high_52w = float(sub["close"].tail(252).max()) if not sub.empty else current_price
            target_price = round(high_52w * 1.1)
            upside = (target_price - current_price) / current_price * 100 if current_price else 0
            ret_3m = row.get("ret_3m", 0)
            ret_6m = row.get("ret_6m", 0)
            cons = {
                "target_price": target_price,
                "opinion": "중립",
                "opinion_score": 3.0,
                "upside_ratio": round(upside, 2),
                "sentiment_score": np.clip((ret_3m * 0.6 + ret_6m * 0.4) / 20, -1, 1),
                "data_type": "proxy",
            }
            report_count = max(1, int(row.get("liquidity_score", 50) / 10))
            print(f"⚠️  proxy 사용 (목표주가={target_price:,}원)")

        records.append({
            "code": code,
            "name": name,
            "current_price": round(current_price, 0),
            "target_price": cons.get("target_price"),
            "upside_ratio": cons.get("upside_ratio"),
            "opinion": cons.get("opinion"),
            "opinion_score": cons.get("opinion_score"),
            "est_per": cons.get("est_per"),
            "est_eps": cons.get("est_eps"),
            "report_count": report_count if report_count else 0,
            "sentiment_score": cons.get("sentiment_score", 0),
            "data_type": cons.get("data_type", "proxy"),
        })

    print(f"\n  실제 컨센서스 수집: {success_count}/{len(score_df)}개 종목")
    return pd.DataFrame(records)


# ── 리서치 팩터 점수 산출 ────────────────────────────────────────────────────

def compute_research_scores(df):
    df = df.copy()

    # 상승여력 점수 (높을수록 good)
    upside = pd.to_numeric(df["upside_ratio"], errors="coerce").fillna(0)
    df["target_score"] = minmax_score(upside)

    # 투자의견 점수 (5=강매수, 1=강매도)
    op = pd.to_numeric(df["opinion_score"], errors="coerce").fillna(3.0)
    df["opinion_score_norm"] = minmax_score(op)

    # 리포트 수 점수
    rc = pd.to_numeric(df["report_count"], errors="coerce").fillna(0)
    df["report_score"] = minmax_score(rc)

    # 감성 점수
    sent = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0)
    df["sentiment_score_norm"] = minmax_score(sent)

    # 통합 리서치 점수
    df["research_score"] = (
        df["target_score"] * 0.40        # 목표주가 상승여력 (가장 중요)
        + df["opinion_score_norm"] * 0.35  # 투자의견
        + df["report_score"] * 0.10       # 리포트 수
        + df["sentiment_score_norm"] * 0.15 # 감성
    )

    return df


def build_research_enhanced_score(score_df, consensus_df):
    merged = score_df.merge(
        consensus_df[["code", "research_score", "upside_ratio", "opinion", "opinion_score", "data_type"]],
        on="code", how="left"
    )
    merged["research_score"] = merged["research_score"].fillna(50.0)

    # 리서치 강화 점수: 퀀트 70% + 실제 리서치 30%
    merged["research_enhanced_score"] = (
        merged["quant_score"] * 0.70
        + merged["research_score"] * 0.30
    )
    merged["research_enhanced_rank"] = merged["research_enhanced_score"].rank(ascending=False).astype(int)
    return merged


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("07. 리포트/컨센서스 팩터 생성 시작 (NAVER 금융 실제 데이터)")
    print("=" * 60)

    con = get_db()
    score_df = pd.read_sql("SELECT * FROM score_result_all", con)
    price_df = pd.read_sql("SELECT * FROM price_data", con)

    # 실제 컨센서스 수집
    consensus_df = collect_all_consensus(score_df, price_df)

    # 리서치 점수 산출
    consensus_df = compute_research_scores(consensus_df)

    consensus_df.to_sql("research_consensus_factor_all", con, if_exists="replace", index=False)
    real_count = (consensus_df["data_type"] == "real").sum()
    print(f"\n[완료] research_consensus_factor_all: {len(consensus_df)}개 종목 "
          f"(실제={real_count}개, proxy={len(consensus_df)-real_count}개)")

    # 리서치 강화 점수 계산
    enhanced_df = build_research_enhanced_score(score_df, consensus_df)
    enhanced_df.to_sql("score_result_research_enhanced_all", con, if_exists="replace", index=False)
    print(f"[완료] score_result_research_enhanced_all 저장")

    print("\n리서치 강화 상위 10개:")
    top10 = enhanced_df.nsmallest(10, "research_enhanced_rank")[
        ["research_enhanced_rank", "code", "name", "quant_score", "research_score",
         "upside_ratio", "opinion", "research_enhanced_score"]
    ]
    print(top10.to_string(index=False))

    con.close()
    print("\n[07] 리포트/컨센서스 팩터 완료\n")


if __name__ == "__main__":
    main()
