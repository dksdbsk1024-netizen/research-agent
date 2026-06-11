"""
01b_fundamental_data.py
실제 재무 데이터 수집 → SQLite 저장
 - KRX 공시 기반 PER/PBR/EPS/BPS/배당수익률 (pykrx)
 - DART OpenAPI 기반 ROE/영업이익률/부채비율 (API키 있을 때)
"""
import sqlite3
import pandas as pd
import numpy as np
import requests
import os
import time
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "data/quant_project.db"
DART_API_KEY = os.environ.get("DART_API_KEY", "")


def get_db():
    return sqlite3.connect(DB_PATH)


# ── pykrx: KRX 실제 PER/PBR/EPS/BPS ────────────────────────────────────────

def fetch_krx_fundamental_pykrx(codes, ref_date=None):
    """pykrx로 종목별 PER/PBR/EPS/BPS/배당수익률 수집"""
    from pykrx import stock as pyk
    if ref_date is None:
        from datetime import datetime, timedelta
        # 최근 거래일 (D-1 기준)
        ref_date = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")

    records = []
    print(f"  KRX 재무지표 수집 기준일: {ref_date}")
    for code in codes:
        try:
            df = pyk.get_market_fundamental_by_date(ref_date, ref_date, code)
            if df is not None and not df.empty and len(df.columns) > 0:
                row = df.iloc[-1]
                records.append({
                    "code": code,
                    "BPS": float(row.get("BPS", 0) or 0),
                    "PER": float(row.get("PER", 0) or 0),
                    "PBR": float(row.get("PBR", 0) or 0),
                    "EPS": float(row.get("EPS", 0) or 0),
                    "DIV": float(row.get("DIV", 0) or 0),
                    "DPS": float(row.get("DPS", 0) or 0),
                    "source": "pykrx_krx",
                })
                print(f"    [{code}] PER={row.get('PER','N/A')}, PBR={row.get('PBR','N/A')}")
            else:
                records.append({"code": code, "source": "pykrx_empty"})
            time.sleep(0.1)
        except Exception as e:
            print(f"    [{code}] pykrx 오류: {e}")
            records.append({"code": code, "source": "error"})

    return pd.DataFrame(records)


def fetch_krx_fundamental_naver(codes):
    """NAVER 금융 종목 상세 페이지에서 PER/PBR/ROE/EPS 스크래핑"""
    import re
    records = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Referer": "https://finance.naver.com/",
    }
    patterns = {
        "PER": r'PER\(배\)</strong>.*?<td[^>]*>\s*([\d\.,\-N/A]+)\s*</td>',
        "PBR": r'PBR\(배\)</strong>.*?<td[^>]*>\s*([\d\.,\-N/A]+)\s*</td>',
        "ROE": r'ROE\(.*?\)</strong>.*?<td[^>]*>\s*([\-\d\.,N/A]+)\s*</td>',
        "EPS": r'EPS\(.*?\)</strong>.*?<td[^>]*>\s*([\-\d\.,N/A]+)\s*</td>',
    }

    def safe_float(s):
        if s is None:
            return None
        s = s.replace(",", "").strip()
        try:
            return float(s)
        except Exception:
            return None

    for code in codes:
        try:
            url = f"https://finance.naver.com/item/main.naver?code={code}"
            r = requests.get(url, headers=headers, timeout=12)
            text = r.text

            row = {"code": code, "source": "naver_scrape"}
            for name, pat in patterns.items():
                m = re.search(pat, text, re.DOTALL)
                row[name] = safe_float(m.group(1)) if m else None

            records.append(row)
            non_null = sum(1 for k in ["PER", "PBR", "ROE", "EPS"] if row.get(k) is not None)
            print(f"    [{code}] PER={row['PER']}, PBR={row['PBR']}, ROE={row['ROE']}, EPS={row['EPS']} ({non_null}/4)")
            time.sleep(0.4)
        except Exception as e:
            print(f"    [{code}] NAVER 오류: {e}")
            records.append({"code": code, "source": "error"})

    return pd.DataFrame(records)


# ── DART OpenAPI: ROE/영업이익률/부채비율 ────────────────────────────────────

def fetch_dart_corp_codes(api_key):
    """DART 기업코드 매핑 파일 다운로드"""
    import zipfile, io
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    try:
        r = requests.get(url, params={"crtfc_key": api_key}, timeout=30)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open("CORPCODE.xml") as f:
                import xml.etree.ElementTree as ET
                tree = ET.parse(f)
                root = tree.getroot()
                records = []
                for item in root.findall("list"):
                    corp_code = item.findtext("corp_code", "")
                    stock_code = item.findtext("stock_code", "").strip()
                    corp_name = item.findtext("corp_name", "")
                    if stock_code:
                        records.append({"dart_code": corp_code, "code": stock_code, "name": corp_name})
                return pd.DataFrame(records)
    except Exception as e:
        print(f"  DART 기업코드 오류: {e}")
        return pd.DataFrame()


def fetch_dart_financials(api_key, dart_code, year="2024", report_code="11011"):
    """DART 단일 기업 재무제표 수집 (report_code: 11011=사업보고서, 11012=반기, 11013=1분기, 11014=3분기)"""
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": dart_code,
        "bsns_year": year,
        "reprt_code": report_code,
        "fs_div": "CFS",  # 연결재무제표 우선
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        if data.get("status") != "000":
            # 연결 없으면 개별로 재시도
            params["fs_div"] = "OFS"
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
        if data.get("status") == "000" and data.get("list"):
            return pd.DataFrame(data["list"])
    except Exception as e:
        print(f"    DART 재무 오류: {e}")
    return pd.DataFrame()


def parse_dart_financials(df):
    """DART 재무제표에서 주요 지표 추출"""
    if df.empty:
        return {}

    def get_amount(account_nm_list):
        for name in account_nm_list:
            row = df[df["account_nm"].str.contains(name, na=False)]
            if not row.empty:
                val_str = row.iloc[0].get("thstrm_amount", "0") or "0"
                try:
                    return float(val_str.replace(",", ""))
                except Exception:
                    return 0
        return 0

    revenue = get_amount(["매출액", "수익(매출액)", "영업수익"])
    operating_profit = get_amount(["영업이익"])
    net_income = get_amount(["당기순이익"])
    total_equity = get_amount(["자본총계"])
    total_assets = get_amount(["자산총계"])
    total_debt = get_amount(["부채총계"])

    result = {}
    if revenue and revenue != 0:
        result["operating_margin"] = operating_profit / revenue * 100
    if total_equity and total_equity != 0:
        result["roe"] = net_income / total_equity * 100
        result["roa"] = net_income / total_assets * 100 if total_assets else None
        result["debt_ratio"] = total_debt / total_equity * 100 if total_debt else None

    return result


def fetch_all_dart(api_key, codes, corp_map):
    """전체 종목 DART 재무지표 수집"""
    from datetime import datetime
    current_year = datetime.today().year
    # 최신 사업보고서 (전년도 기준)
    target_year = str(current_year - 1)

    records = []
    for code in codes:
        row = corp_map[corp_map["code"] == code]
        if row.empty:
            print(f"    [{code}] DART 기업코드 없음")
            records.append({"code": code, "source": "dart_no_code"})
            continue

        dart_code = row.iloc[0]["dart_code"]
        print(f"    [{code}] DART 수집 중 (dart_code={dart_code}, year={target_year})")
        fin_df = fetch_dart_financials(api_key, dart_code, year=target_year)
        parsed = parse_dart_financials(fin_df)

        if parsed:
            parsed["code"] = code
            parsed["source"] = "dart"
            records.append(parsed)
            roe_v = parsed.get("roe")
            opm_v = parsed.get("operating_margin")
            if roe_v is not None and isinstance(roe_v, (int, float)):
                print(f"      ROE={roe_v:.2f}%, 영업이익률={opm_v:.2f}%" if isinstance(opm_v, (int, float)) else f"      ROE={roe_v:.2f}%")
            else:
                print(f"      파싱 결과: {parsed}")
        else:
            records.append({"code": code, "source": "dart_empty"})
        time.sleep(0.5)

    return pd.DataFrame(records)


# ── 스코어 재계산 ─────────────────────────────────────────────────────────────

def recompute_scores_with_fundamentals(score_df, fund_df):
    """실제 재무지표(NAVER + DART)를 반영해 스코어 재계산"""
    merged = score_df.merge(fund_df, on="code", how="left")

    def minmax(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(50.0, index=series.index)
        return (series - mn) / (mx - mn) * 100

    def to_num(col):
        return pd.to_numeric(merged[col], errors="coerce") if col in merged.columns else None

    # ── 밸류에이션: PER (낮을수록 저평가) + PBR 결합 ─────────────────────
    per = to_num("PER")
    pbr = to_num("PBR")
    if per is not None:
        per_c = per.replace(0, np.nan).clip(1, 200)
        val_score = 100 - minmax(per_c.fillna(per_c.median()))
        if pbr is not None:
            pbr_c = pbr.replace(0, np.nan).clip(0.1, 20)
            pbr_score = 100 - minmax(pbr_c.fillna(pbr_c.median()))
            val_score = val_score * 0.6 + pbr_score * 0.4
        has_val = per.notna() & (per > 0)
        merged.loc[has_val, "valuation_score"] = val_score[has_val]
        print(f"  PER/PBR 기반 밸류에이션 점수 업데이트: {has_val.sum()}개 종목")

    # ── 수익성: DART ROE(우선) + 영업이익률 결합 ─────────────────────────
    # DART ROE (더 정확한 연결재무제표 기반)
    dart_df_local = pd.DataFrame()
    try:
        import sqlite3
        con_tmp = sqlite3.connect(DB_PATH)
        dart_df_local = pd.read_sql("SELECT * FROM dart_financials", con_tmp)
        con_tmp.close()
    except Exception:
        pass

    if not dart_df_local.empty and "roe" in dart_df_local.columns:
        merged = merged.merge(
            dart_df_local[["code", "roe", "operating_margin", "debt_ratio"]].rename(
                columns={"roe": "dart_roe", "operating_margin": "dart_opm", "debt_ratio": "dart_debt"}
            ),
            on="code", how="left"
        )
        dart_roe = pd.to_numeric(merged["dart_roe"], errors="coerce")
        dart_opm = pd.to_numeric(merged.get("dart_opm"), errors="coerce") if "dart_opm" in merged.columns else None
        dart_debt = pd.to_numeric(merged.get("dart_debt"), errors="coerce") if "dart_debt" in merged.columns else None

        has_dart = dart_roe.notna() & (dart_roe.abs() > 0.01)
        if has_dart.sum() > 0:
            roe_c = dart_roe.clip(-50, 100)
            prof_score = minmax(roe_c.fillna(roe_c.median()))
            # 영업이익률 추가 반영 (있을 때)
            if dart_opm is not None:
                opm_c = dart_opm.clip(-50, 100).replace(0, np.nan)
                # 이상치(금융사 수천%) 제거
                opm_c = opm_c.clip(opm_c.quantile(0.05), opm_c.quantile(0.95))
                opm_score = minmax(opm_c.fillna(opm_c.median()))
                has_opm = dart_opm.notna() & dart_opm.between(-50, 100)
                prof_score = np.where(has_opm, prof_score * 0.6 + opm_score * 0.4, prof_score)
                prof_score = pd.Series(prof_score, index=merged.index)
            merged.loc[has_dart, "profitability_score"] = prof_score[has_dart]
            print(f"  DART ROE+영업이익률 기반 수익성 점수 업데이트: {has_dart.sum()}개 종목")

        # 안정성: 부채비율 반영 (낮을수록 안정)
        if dart_debt is not None:
            debt_c = dart_debt.replace(0, np.nan).clip(0, 500)
            has_debt = dart_debt.notna() & (dart_debt > 0)
            # 금융사(부채비율 매우 높음) 제외
            reasonable_debt = has_debt & (dart_debt < 500)
            if reasonable_debt.sum() > 0:
                stab_debt_score = 100 - minmax(debt_c.fillna(debt_c.median()))
                # 기존 안정성 점수와 50:50 결합
                merged.loc[reasonable_debt, "stability_score"] = (
                    merged.loc[reasonable_debt, "stability_score"] * 0.5
                    + stab_debt_score[reasonable_debt] * 0.5
                )
                print(f"  DART 부채비율 기반 안정성 점수 보완: {reasonable_debt.sum()}개 종목")
    else:
        # DART 없으면 NAVER ROE 사용
        naver_roe = to_num("ROE")
        if naver_roe is not None:
            has_roe = naver_roe.notna() & (naver_roe.abs() > 0.01)
            if has_roe.sum() > 0:
                roe_c = naver_roe.clip(-50, 100)
                merged.loc[has_roe, "profitability_score"] = minmax(roe_c.fillna(roe_c.median()))[has_roe]
                print(f"  NAVER ROE 기반 수익성 점수 업데이트: {has_roe.sum()}개 종목")

    # ── 퀀트 점수 재계산 ─────────────────────────────────────────────────
    merged["quant_score"] = (
        merged["profitability_score"] * 0.20
        + merged["valuation_score"] * 0.20
        + merged["momentum_score"] * 0.25
        + merged["stability_score"] * 0.20
        + merged["liquidity_score"] * 0.15
    )
    merged["rank"] = merged["quant_score"].rank(ascending=False).astype(int)

    return merged


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01b. 실제 재무 데이터 수집 시작")
    print("=" * 60)

    con = get_db()
    master_df = pd.read_sql("SELECT * FROM stock_master", con)
    score_df = pd.read_sql("SELECT * FROM score_result_all", con)
    codes = master_df["code"].tolist()

    # ── Step 1: NAVER 금융 스크래핑 (PER/PBR/ROE) ──────────────────────
    print("\n[Step 1] NAVER 금융 PER/PBR/ROE 수집")
    naver_df = fetch_krx_fundamental_naver(codes)

    # ── Step 2: pykrx KRX 재무지표 (보완) ──────────────────────────────
    print("\n[Step 2] pykrx KRX 재무지표 보완 수집")
    try:
        pykrx_df = fetch_krx_fundamental_pykrx(codes)
        # pykrx와 NAVER 병합 (pykrx 우선, 없으면 NAVER 값 사용)
        fund_df = naver_df.copy()
        if not pykrx_df.empty and "PER" in pykrx_df.columns:
            for col in ["PER", "PBR", "EPS", "BPS", "DIV"]:
                if col in pykrx_df.columns:
                    fund_df = fund_df.merge(
                        pykrx_df[["code", col]].rename(columns={col: f"{col}_krx"}),
                        on="code", how="left"
                    )
                    # pykrx 값이 있고 0이 아니면 우선 사용
                    krx_col = f"{col}_krx"
                    if krx_col in fund_df.columns:
                        valid_mask = fund_df[krx_col].notna() & (fund_df[krx_col] != 0)
                        fund_df.loc[valid_mask, col] = fund_df.loc[valid_mask, krx_col]
    except Exception as e:
        print(f"  pykrx 오류 무시: {e}")
        fund_df = naver_df.copy()

    # ── Step 3: DART OpenAPI (ROE/영업이익률) ───────────────────────────
    if DART_API_KEY:
        print(f"\n[Step 3] DART OpenAPI 재무지표 수집 (API Key 감지)")
        corp_map = fetch_dart_corp_codes(DART_API_KEY)
        if not corp_map.empty:
            dart_df = fetch_all_dart(DART_API_KEY, codes, corp_map)
            dart_df.to_sql("dart_financials", con, if_exists="replace", index=False)
            print(f"  DART 재무 저장: {len(dart_df)}개 종목")
            # DART ROE로 NAVER ROE 보완
            if "roe" in dart_df.columns:
                fund_df = fund_df.merge(
                    dart_df[["code", "roe", "operating_margin", "debt_ratio"]].rename(
                        columns={"roe": "ROE_dart"}
                    ),
                    on="code", how="left"
                )
                dart_valid = fund_df["ROE_dart"].notna() & (fund_df["ROE_dart"].abs() > 0.01)
                fund_df.loc[dart_valid, "ROE"] = fund_df.loc[dart_valid, "ROE_dart"]
    else:
        print("\n[Step 3] DART API Key 없음 → 건너뜀")
        print("  (DART_API_KEY 환경변수 설정 또는 알려주시면 적용 가능)")

    # 저장
    fund_df.to_sql("fundamental_data", con, if_exists="replace", index=False)
    print(f"\n[완료] fundamental_data: {len(fund_df)}개 종목 저장")

    # ── Step 4: 스코어 재계산 ──────────────────────────────────────────
    print("\n[Step 4] 실제 재무지표 반영 스코어 재계산")
    updated_score = recompute_scores_with_fundamentals(score_df, fund_df)
    updated_score.to_sql("score_result_all", con, if_exists="replace", index=False)
    updated_score.to_sql("score_result", con, if_exists="replace", index=False)

    print("\n재무지표 반영 후 TOP 10:")
    cols = ["rank", "code", "name", "quant_score", "profitability_score", "valuation_score"]
    print(updated_score.nsmallest(10, "rank")[cols].to_string(index=False))

    con.close()
    print("\n[01b] 실제 재무 데이터 수집 완료\n")


if __name__ == "__main__":
    main()
