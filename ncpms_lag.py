"""
ncpms_lag.py — NCPMS 예찰 데이터로 self_lag 채우기
─────────────────────────────────────────────────────────────────────
잎집무늬마름병(논벼·벼관찰포) 피해율을 NCPMS OpenAPI 에서 받아
predict.py 의 self_lag1 / self_lag2 로 공급한다.

확정 사양 (실측 검증 완료):
  · SVC51 : 그 해 벼관찰포·논벼 기본조사 회차 목록 (8회: 6/1~9/16)
            → insectKey + 조사기준일자(inputStdrDatetm)
  · SVC52 : insectKey 로 시도별·병해별 상세
            → 잎집무늬마름병 '피해율' = inqireCnClCode "SF0011"
  · 단위   : merged_df.csv 잎집무늬마름병 컬럼과 1:1 동일 (0~1, 변환 X)
  · 지역   : sidoNm 풀네임 = KR_SIDO 키 → KR_SIDO[sidoNm]["region"]
  · self_lag 매칭: target_date 직전 2개 '조사기준일자' (회차번호 아님!)

캐시 불필요 (GitHub Actions 가 매 run 새 컨테이너 → 매번 fresh fetch).
지나간 회차 결과는 불변이고 데이터량이 작아(8회×15지역) 매일 받아도 무방.

사용 (predict.py main 에서):
    from ncpms_lag import build_lag_table, resolve_lags
    table = build_lag_table(date.today().year, api_key=os.getenv("NCPMS_API_KEY"))
    lag1, lag2 = resolve_lags(table, region, date.today())
"""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Optional

import requests

from kr_regions import KR_SIDO

# ─── 고정 상수 (벼 / 잎집무늬마름병 전용) ──────────────────────────
API_URL = "http://ncpms.rda.go.kr/npmsAPI/service"
PREDICTN_SPCHCKN_CODE = "00204"      # 벼관찰포
KNCR_CODE = "FC010101"               # 논벼
SHEATH_BLIGHT_CODE = "SF0011"        # 잎집무늬마름병(피해율) ← self_lag 그 자체
SERVICE_TYPE = "AA003"               # JSON

HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
HTTP_BACKOFF = 3.0

# 풀네임 sidoNm → 학습데이터 region 약칭 (KR_SIDO 활용, 서울/제주 = None)
SIDO_TO_REGION = {sido: info["region"] for sido, info in KR_SIDO.items()}


# ═══════════════════════════════════════════════════════════════════
#  HTTP
# ═══════════════════════════════════════════════════════════════════
def _request_with_retry(params: dict, verbose: bool = True) -> dict:
    """timeout / 5xx 시 지수 백오프 재시도. JSON dict 반환."""
    last_exc = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            r = requests.get(API_URL, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} server error", response=r)
            r.raise_for_status()
            # NCPMS 는 Content-Type 을 text/xml 로 잘못 줄 때가 있어 r.json() 대신 직접 파싱
            return r.json()
        except (requests.Timeout, requests.ConnectionError,
                requests.HTTPError, ValueError) as e:
            last_exc = e
            if attempt == HTTP_RETRIES:
                break
            wait = HTTP_BACKOFF * (2 ** (attempt - 1))
            if verbose:
                print(f"   ⏳ NCPMS 재시도 {attempt}/{HTTP_RETRIES} (대기 {wait:.0f}s): {e}")
            time.sleep(wait)
    raise last_exc


def _as_list(node) -> list:
    """응답의 list/item 노드를 항상 list 로 정규화 (단일 dict, 래핑 형태 모두 처리)."""
    if node is None:
        return []
    if isinstance(node, list):
        return node
    if isinstance(node, dict):
        # {"item": [...]} or {"item": {...}} 형태
        if "item" in node:
            return _as_list(node["item"])
        return [node]
    return []


# ═══════════════════════════════════════════════════════════════════
#  SVC51 — 회차 목록
# ═══════════════════════════════════════════════════════════════════
def fetch_survey_list(year: int, api_key: str, verbose: bool = True) -> list[dict]:
    """
    그 해 벼관찰포·논벼 기본조사 회차 목록.
    반환: [{"date": "2024-06-16", "insectKey": "...", "round": 2}, ...]
          조사기준일자(date) 오름차순.
    """
    data = _request_with_retry({
        "apiKey": api_key,
        "serviceCode": "SVC51",
        "serviceType": SERVICE_TYPE,
        "searchExaminYear": str(year),
        "searchPredictnSpchcknCode": PREDICTN_SPCHCKN_CODE,
        "searchKncrCode": KNCR_CODE,
        "displayCount": 50,
        "startPoint": 1,
    }, verbose=verbose)

    svc = data.get("service", data)
    items = _as_list(svc.get("list"))
    surveys = []
    for it in items:
        ds = str(it.get("inputStdrDatetm", "")).strip()   # "20240616"
        ikey = str(it.get("insectKey", "")).strip()
        if len(ds) != 8 or not ikey:
            continue
        surveys.append({
            "date": f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}",
            "insectKey": ikey,
            "round": int(it.get("examinTmrd", 0) or 0),
        })
    surveys.sort(key=lambda x: x["date"])
    if verbose:
        print(f"   SVC51: {year}년 회차 {len(surveys)}개 "
              f"({surveys[0]['date']} ~ {surveys[-1]['date']})" if surveys
              else f"   SVC51: {year}년 회차 0개")
    return surveys


# ═══════════════════════════════════════════════════════════════════
#  SVC52 — 시도별 잎집무늬마름병 피해율
# ═══════════════════════════════════════════════════════════════════
def fetch_sido_detail(insect_key: str, api_key: str, verbose: bool = True) -> dict[str, float]:
    """
    insectKey 의 시도별 상세 → 잎집무늬마름병 피해율(SF0011)만 추출.
    반환: {region약칭: 피해율}  (예: {"전북": 0.025, "전남": 0.044, ...})
          서울/제주 등 region=None 인 시도는 제외.
    """
    data = _request_with_retry({
        "apiKey": api_key,
        "serviceCode": "SVC52",
        "serviceType": SERVICE_TYPE,
        "insectKey": insect_key,
    }, verbose=verbose)

    svc = data.get("service", data)
    items = _as_list(svc.get("structList"))
    out: dict[str, float] = {}
    for it in items:
        if str(it.get("inqireCnClCode", "")).strip() != SHEATH_BLIGHT_CODE:
            continue
        sido_full = str(it.get("sidoNm", "")).strip()
        region = SIDO_TO_REGION.get(sido_full)
        if region is None:           # 서울/제주/미지정 → skip
            continue
        try:
            out[region] = float(it.get("inqireValue", 0) or 0)
        except (TypeError, ValueError):
            out[region] = 0.0
    return out


# ═══════════════════════════════════════════════════════════════════
#  통합: 날짜별·지역별 피해율 테이블
# ═══════════════════════════════════════════════════════════════════
def build_lag_table(year: int, api_key: str, verbose: bool = True,
                    sleep_between: float = 0.5) -> dict[str, dict[str, float]]:
    """
    그 해 모든 회차의 시도별 잎집무늬마름병 피해율 테이블.
    반환: {"2024-06-16": {"전북": 0.0, "전남": 0.01, ...}, "2024-07-01": {...}, ...}
          (조사기준일자 → {region: 피해율})

    실패한 회차는 건너뜀(부분 실패 허용). api_key 없으면 빈 dict.
    """
    if not api_key:
        if verbose:
            print("   ⚠️ NCPMS_API_KEY 없음 → self_lag 빈 테이블 (D=0 동작)")
        return {}

    try:
        surveys = fetch_survey_list(year, api_key, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"   ⚠️ SVC51 실패 → self_lag 빈 테이블: {e}")
        return {}

    table: dict[str, dict[str, float]] = {}
    for sv in surveys:
        try:
            rates = fetch_sido_detail(sv["insectKey"], api_key, verbose=verbose)
            table[sv["date"]] = rates
        except Exception as e:
            if verbose:
                print(f"   ⚠️ SVC52 실패 (회차 {sv['date']}, 건너뜀): {e}")
        time.sleep(sleep_between)    # rate limit 완화

    if verbose:
        print(f"   NCPMS self_lag 테이블: {len(table)}개 회차 적재")
    return table


def resolve_lags(table: dict[str, dict[str, float]],
                 region: Optional[str],
                 target_date) -> tuple[float, float]:
    """
    target_date 직전 2개 '조사기준일자' 의 (지역 피해율) → (self_lag1, self_lag2).

    self_lag1 = target_date 보다 앞선 가장 최근 회차의 피해율
    self_lag2 = 그 전 회차의 피해율
    회차 부족·지역 없음·시즌 전 → 해당 값 0.0.

    회차번호(examinTmrd)가 아니라 '날짜' 로 매칭하므로
    predict.py STANDARD_SURVEY_MD(7개) 와 NCPMS(8개, 6/1 포함) 불일치에 안전.
    """
    if not table or region is None:
        return 0.0, 0.0

    if isinstance(target_date, datetime):
        target_date = target_date.date()
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    # target_date 보다 '엄격히 앞선' 조사일자만, 최신순
    past_dates = sorted(
        (d for d in table.keys() if date.fromisoformat(d) < target_date),
        reverse=True,
    )

    def _rate(d: str) -> float:
        return float(table.get(d, {}).get(region, 0.0))

    lag1 = _rate(past_dates[0]) if len(past_dates) >= 1 else 0.0
    lag2 = _rate(past_dates[1]) if len(past_dates) >= 2 else 0.0
    return lag1, lag2


# ═══════════════════════════════════════════════════════════════════
#  단독 실행 — 진단/검증
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys, json

    api_key = os.getenv("NCPMS_API_KEY")
    if not api_key:
        sys.exit("환경변수 NCPMS_API_KEY 가 필요합니다.")

    year = int(sys.argv[1]) if len(sys.argv) > 1 else date.today().year - 1
    print(f"═══ NCPMS self_lag 테이블 빌드 (year={year}) ═══")
    table = build_lag_table(year, api_key, verbose=True)

    print("\n[회차별 전북/전남/세종 피해율]")
    for d in sorted(table):
        r = table[d]
        print(f"  {d}: 전북={r.get('전북',0):.3f}  전남={r.get('전남',0):.3f}  세종={r.get('세종',0):.3f}")

    # resolve_lags 예시: 8/20 기준 (직전=8/16, 그전=8/1)
    for region in ["전북", "전남", "세종"]:
        l1, l2 = resolve_lags(table, region, date(year, 8, 20))
        print(f"\n  resolve_lags({region}, {year}-08-20) → lag1={l1:.3f}, lag2={l2:.3f}")
