"""
predict.py (v3 - 11차 회의록 사양 반영)
─────────────────────────────────────────────────────────────────────────────
EBM 모델로 광역단체별 잎집무늬마름병 위험도 예측 (운영용).

v3 변경 (회의록 11차):
  - P_EBM = clip(ŷ / y_p95_train, 0, 1)         (회의록 9p)
  - D     = clip(self_lag1 / y_p95_train, 0, 1) (회의록 9p)
  - Risk Score = α × P_EBM + (1-α) × (D × E), α=0.7  (회의록 9p, 15p)
  - Grade 경계 b1=0.03, b2=0.16, b3=0.35           (회의록 15p)
  - Override: P_EBM≥0.85 → ≥2, D≥0.8 ∧ E≥0.7 → 3   (회의록 18p)
  - 상황 메시지 (4-quadrant)                       (회의록 21p)
  - 주요 원인 top-3 (self_lag 제외)                (회의록 22p)
  - self_lag NaN → 0                              (회의록 16p)
  - E (CTM 환경 점수) 외부 주입 (--e 옵션, 제진님 담당)

v2 → v3 호환성:
  기존 v2 인자 (sido, --date, --lag1, --lag2)는 그대로 작동.
  단, --e 안 주면 D×E=0으로 처리되어 Risk Score=α×P_EBM 만 산출됨.

사용:
  # 오늘 기준 (forecast)
  python3 predict.py 전북특별자치도

  # 과거 검증 (archive, E 미지정)
  python3 predict.py 전북특별자치도 --date 2024-08-16 --lag1 0.036 --lag2 0.007

  # 과거 검증 (E 외부 주입)
  python3 predict.py 전북특별자치도 --date 2024-08-16 --lag1 0.036 --lag2 0.007 --e 0.45
"""

import argparse
import json
import os
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import firebase_admin
import joblib
import numpy as np
import pandas as pd
import requests
from firebase_admin import credentials

from kr_regions import KR_SIDO, get_sido_info, list_sido
from open_meteo_3_7_15 import OpenMeteoMultiWindowExtractor
from your_ctm_module import compute_e

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ═══════════════════════════════════════════════════════════════════════
#  상수
# ═══════════════════════════════════════════════════════════════════════
MODEL_DIR = Path(__file__).parent
TARGET = "잎집무늬마름병"
GDD_THRESHOLD = 25.0

STANDARD_SURVEY_MD = [(6, 16), (7, 1), (7, 16), (8, 1), (8, 16), (9, 1), (9, 16)]
SEASON_START_MD = (6, 1)

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo가 가끔 느리거나 일시적 502/504/timeout/429를 내므로 재시도 + 백오프
HTTP_TIMEOUT = 60          # 초
HTTP_RETRIES = 3           # 최대 시도 횟수
HTTP_BACKOFF = 5.0         # 초, 시도마다 지수 증가


def _request_with_retry(url, params, verbose=True):
    """timeout / 5xx / 429 시 지수 백오프로 재시도. 429는 Retry-After 헤더 존중."""
    import time
    last_exc = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            # 429 Too Many Requests: Retry-After 헤더가 있으면 그만큼 대기
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", HTTP_BACKOFF * (2 ** (attempt - 1))))
                if verbose:
                    print(f"   ⏳ 429 rate-limit (Retry-After={wait:.0f}s) — 시도 {attempt}/{HTTP_RETRIES}")
                if attempt == HTTP_RETRIES:
                    r.raise_for_status()
                time.sleep(min(wait, 60))    # 최대 60초까지만 대기
                continue
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} server error", response=r)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_exc = e
            if attempt == HTTP_RETRIES:
                break
            wait = HTTP_BACKOFF * (2 ** (attempt - 1))
            if verbose:
                print(f"   ⏳ Open-Meteo 재시도 {attempt}/{HTTP_RETRIES} (대기 {wait:.0f}s): {e}")
            time.sleep(wait)
    raise last_exc

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "relative_humidity_2m_mean", "dew_point_2m_mean",
    "wind_speed_10m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant",
    "precipitation_sum", "sunshine_duration", "shortwave_radiation_sum",
    "soil_temperature_0_to_7cm_mean", "soil_moisture_0_to_10cm_mean",
]

# ─── 회의록 11차 사양 ─────────────────────────────────────────────────
# y_p95_train: merged_df.csv 의 2014~2023 잎집무늬마름병 95%ile
# q33, q67   : train 양성값의 33/67%ile  (회의록 11p 입력값 요약)
Y_P95_TRAIN = 0.4537
Q33 = 0.0429
Q67 = 0.1866

ALPHA = 0.7                            # 회의록 15p (10차 합의)
B1, B2, B3 = 0.03, 0.16, 0.35          # 회의록 15p Grid Search 결과
P_EBM_OVERRIDE = 0.85                  # 회의록 18p
D_OVERRIDE = 0.8                       # 회의록 18p
E_OVERRIDE = 0.7                       # 회의록 18p

GRADE_NAME = {0: '안전', 1: '주의', 2: '경계', 3: '심각'}
GRADE_EMOJI = {0: '🟢', 1: '🟡', 2: '🟠', 3: '🔴'}

# 주요 원인 추출 시 제외 (회의록 22p)
LAG_EXCLUDE = {'self_lag1', 'self_lag2'}

# ─── 리포트 멘트 매트릭스 (리포트_멘트_매트릭스.xlsx) ────────────────────
# Sheet1 「변수_그룹_매핑」을 코드 dict 로 옮긴 것.
#   feature → (그룹, 그룹명, 윈도우(일), 단위, 위험방향, 짧은 한국어명)
# 그룹: A 기온 / B 습도 / C 강수 / D 풍속 / E 일사·일조
#       F 고온다습복합 / G 토양 / H 대기오염 / I lag·누적위험
FEATURE_GROUP = {
    # A 기온
    'temp_mean_3':       ('A', '기온',         3,  '°C',  'BOTH', '평균기온'),
    'temp_mean_7':       ('A', '기온',         7,  '°C',  'BOTH', '평균기온'),
    'temp_mean_15':      ('A', '기온',         15, '°C',  'BOTH', '평균기온'),
    'tmax_max_3':        ('A', '기온',         3,  '°C',  'UP',   '최고기온'),
    'tmax_max_7':        ('A', '기온',         7,  '°C',  'UP',   '최고기온'),
    'tmax_max_15':       ('A', '기온',         15, '°C',  'UP',   '최고기온'),
    'tmin_min_3':        ('A', '기온',         3,  '°C',  'DOWN', '최저기온'),
    'tmin_min_7':        ('A', '기온',         7,  '°C',  'DOWN', '최저기온'),
    'tmin_min_15':       ('A', '기온',         15, '°C',  'DOWN', '최저기온'),
    'hot_days_3':        ('A', '기온',         3,  '일',  'UP',   '고온일수'),
    'hot_days_7':        ('A', '기온',         7,  '일',  'UP',   '고온일수'),
    'hot_days_15':       ('A', '기온',         15, '일',  'UP',   '고온일수'),
    'dtr_mean_3':        ('A', '기온',         3,  '°C',  'UP',   '일교차'),
    'dtr_mean_7':        ('A', '기온',         7,  '°C',  'UP',   '일교차'),
    'dtr_mean_15':       ('A', '기온',         15, '°C',  'UP',   '일교차'),
    'soil_temp_mean_3':  ('A', '기온',         3,  '°C',  'UP',   '토양온도'),
    'soil_temp_mean_7':  ('A', '기온',         7,  '°C',  'UP',   '토양온도'),
    'soil_temp_mean_15': ('A', '기온',         15, '°C',  'UP',   '토양온도'),
    # B 습도
    'rh_mean_3':         ('B', '습도',         3,  '%',   'UP',   '상대습도'),
    'rh_mean_7':         ('B', '습도',         7,  '%',   'UP',   '상대습도'),
    'rh_mean_15':        ('B', '습도',         15, '%',   'UP',   '상대습도'),
    'dew_mean_3':        ('B', '습도',         3,  '°C',  'UP',   '이슬점'),
    'dew_mean_7':        ('B', '습도',         7,  '°C',  'UP',   '이슬점'),
    'dew_mean_15':       ('B', '습도',         15, '°C',  'UP',   '이슬점'),
    'vapor_mean_3':      ('B', '습도',         3,  'hPa', 'UP',   '수증기압'),
    'vapor_mean_7':      ('B', '습도',         7,  'hPa', 'UP',   '수증기압'),
    'vapor_mean_15':     ('B', '습도',         15, 'hPa', 'UP',   '수증기압'),
    'humid_days_3':      ('B', '습도',         3,  '일',  'UP',   '고습일수'),
    'humid_days_7':      ('B', '습도',         7,  '일',  'UP',   '고습일수'),
    'humid_days_15':     ('B', '습도',         15, '일',  'UP',   '고습일수'),
    # C 강수
    'rain_sum_3':        ('C', '강수',         3,  'mm',  'UP',   '누적강수'),
    'rain_sum_7':        ('C', '강수',         7,  'mm',  'UP',   '누적강수'),
    'rain_sum_15':       ('C', '강수',         15, 'mm',  'UP',   '누적강수'),
    'rainy_days_3':      ('C', '강수',         3,  '일',  'UP',   '강수일수'),
    'rainy_days_7':      ('C', '강수',         7,  '일',  'UP',   '강수일수'),
    'rainy_days_15':     ('C', '강수',         15, '일',  'UP',   '강수일수'),
    'heavy_rain_days_3': ('C', '강수',         3,  '일',  'UP',   '폭우일수'),
    'heavy_rain_days_7': ('C', '강수',         7,  '일',  'UP',   '폭우일수'),
    'heavy_rain_days_15':('C', '강수',         15, '일',  'UP',   '폭우일수'),
    # D 풍속
    'wind_mean_3':       ('D', '풍속',         3,  'm/s', 'DOWN', '평균풍속'),
    'wind_mean_7':       ('D', '풍속',         7,  'm/s', 'DOWN', '평균풍속'),
    'wind_mean_15':      ('D', '풍속',         15, 'm/s', 'DOWN', '평균풍속'),
    # E 일사/일조
    'sunshine_sum_3':    ('E', '일사/일조',    3,  'h',   'DOWN', '일조시간'),
    'sunshine_sum_7':    ('E', '일사/일조',    7,  'h',   'DOWN', '일조시간'),
    'sunshine_sum_15':   ('E', '일사/일조',    15, 'h',   'DOWN', '일조시간'),
    # F 고온다습복합
    'hot_humid_days_3':  ('F', '고온다습복합', 3,  '일',  'UP',   '고온다습일수'),
    'hot_humid_days_7':  ('F', '고온다습복합', 7,  '일',  'UP',   '고온다습일수'),
    'hot_humid_days_15': ('F', '고온다습복합', 15, '일',  'UP',   '고온다습일수'),
    # G 토양
    'soil_moisture_mean_3':  ('G', '토양',     3,  '',    'UP',   '토양수분'),
    'soil_moisture_mean_7':  ('G', '토양',     7,  '',    'UP',   '토양수분'),
    'soil_moisture_mean_15': ('G', '토양',     15, '',    'UP',   '토양수분'),
}

# 평년값 (mean baseline). 매트릭스 시트의 'val > baseline' 조건과
# [diff] 플레이스홀더 계산에 사용. 한국 여름철 기후값으로 추정.
BASELINES = {
    'rh_mean_3': 74, 'rh_mean_7': 74, 'rh_mean_15': 74,
    'rain_sum_3': 15, 'rain_sum_7': 35, 'rain_sum_15': 75,
    'sunshine_sum_3': 24, 'sunshine_sum_7': 56, 'sunshine_sum_15': 120,
    'temp_mean_3': 22, 'temp_mean_7': 22, 'temp_mean_15': 22,
    'soil_moisture_mean_3': 16, 'soil_moisture_mean_7': 16, 'soil_moisture_mean_15': 16,
    'soil_temp_mean_3': 20, 'soil_temp_mean_7': 20, 'soil_temp_mean_15': 20,
}

GROWTH_STAGE_DESC = {
    '이앙기':       '이제 막 모내기를 마친 이앙기',
    '분얼기':       '벼가 가장 약한 분얼기',
    '유수형성기':   '이삭이 만들어지는 유수형성기',
    '수잉기':       '이삭이 패는 수잉기',
    '출수기':       '벼가 이삭을 내는 출수기',
    '등숙기':       '낟알이 여무는 등숙기',
}

# 한글 변수명
FEATURE_KO = {
    'temp_mean_3': '최근 3일 평균기온', 'temp_mean_7': '최근 7일 평균기온', 'temp_mean_15': '최근 15일 평균기온',
    'rh_mean_3': '최근 3일 상대습도', 'rh_mean_7': '최근 7일 상대습도', 'rh_mean_15': '최근 15일 상대습도',
    'precip_sum_3': '최근 3일 누적강수', 'precip_sum_7': '최근 7일 누적강수', 'precip_sum_15': '최근 15일 누적강수',
    'rainy_days_3': '최근 3일 강수일수', 'rainy_days_7': '최근 7일 강수일수', 'rainy_days_15': '최근 15일 강수일수',
    'hot_days_3': '최근 3일 고온일수', 'hot_days_7': '최근 7일 고온일수', 'hot_days_15': '최근 15일 고온일수',
    'humid_days_3': '최근 3일 다습일수', 'humid_days_7': '최근 7일 다습일수', 'humid_days_15': '최근 15일 다습일수',
    'dew_mean_3': '최근 3일 이슬점', 'dew_mean_7': '최근 7일 이슬점', 'dew_mean_15': '최근 15일 이슬점',
    'vapor_mean_3': '최근 3일 수증기압', 'vapor_mean_7': '최근 7일 수증기압', 'vapor_mean_15': '최근 15일 수증기압',
    'wind_mean_3': '최근 3일 풍속', 'wind_mean_7': '최근 7일 풍속', 'wind_mean_15': '최근 15일 풍속',
    'sunshine_sum_3': '최근 3일 일조시간', 'sunshine_sum_7': '최근 7일 일조시간', 'sunshine_sum_15': '최근 15일 일조시간',
    'soil_temp_mean_3': '최근 3일 토양온도', 'soil_temp_mean_7': '최근 7일 토양온도', 'soil_temp_mean_15': '최근 15일 토양온도',
    'soil_moist_mean_3': '최근 3일 토양수분', 'soil_moist_mean_7': '최근 7일 토양수분', 'soil_moist_mean_15': '최근 15일 토양수분',
    'gdd_15': '15일 누적 적산온도', 'gdd_cum': '시즌 누적 적산온도',
    'year': '연도', 'month': '월', 'dayofyear': '일자', 'days_since_season_start': '시즌 경과일',
    'season_idx': '조사회차',
}


# ═══════════════════════════════════════════════════════════════════════
#  Weather fetch (v2 동일)
# ═══════════════════════════════════════════════════════════════════════
def _build_daily_df_from_response(d):
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "tmax": d.get("temperature_2m_max"), "tmin": d.get("temperature_2m_min"),
        "tmean": d.get("temperature_2m_mean"),
        "rh_mean": d.get("relative_humidity_2m_mean"),
        "dew_point": d.get("dew_point_2m_mean"),
        "wind_speed_mean": d.get("wind_speed_10m_mean"),
        "wind_speed_max": d.get("wind_speed_10m_max"),
        "wind_dir": d.get("wind_direction_10m_dominant"),
        "precipitation": d.get("precipitation_sum"),
        "sunshine_duration": d.get("sunshine_duration"),
        "solar_radiation": d.get("shortwave_radiation_sum"),
        "soil_temp": d.get("soil_temperature_0_to_7cm_mean"),
        "soil_moisture": d.get("soil_moisture_0_to_10cm_mean"),
    })


def fetch_30day_window(lat, lon, today=None, n_past=14, n_future=15, verbose=True):
    """
    today 기준 (n_past + 1 + n_future)일 = 기본 30일 daily 데이터.
    Open-Meteo FORECAST API 한 번 호출로 과거 + 오늘 + 예보를 모두 받음.

    반환: date 오름차순으로 정렬된 DataFrame (n_past+1+n_future 행).
    """
    if today is None:
        today = date.today()

    # past_days=N → [today-N, ..., today-1], forecast_days=M → [today, ..., today+M-1]
    past_days = max(0, n_past)
    forecast_days = max(1, n_future + 1)        # today 포함
    forecast_days = min(forecast_days, 16)      # API 한도
    past_days = min(past_days, 92)

    params = {
        "latitude": lat, "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "past_days": past_days, "forecast_days": forecast_days,
        "timezone": "auto", "wind_speed_unit": "ms",
    }
    if verbose:
        print(f"🌐 FORECAST → past {past_days}d + today + forecast {forecast_days - 1}d "
              f"(총 {past_days + forecast_days}일)")
    r = _request_with_retry(FORECAST_API, params, verbose=verbose)
    d = r.json()["daily"]
    df = (_build_daily_df_from_response(d)
          .sort_values("date").reset_index(drop=True))
    if verbose: print(f"✅ daily {len(df)}일 수신")
    return df


def _slice_window_ending_at(daily_df, pred_date, W):
    """pred_date 로 끝나는 W일 윈도우 (오름차순). 길이 부족 시 ValueError."""
    end_ts = pd.Timestamp(pred_date)
    start_ts = end_ts - pd.Timedelta(days=W - 1)
    win = (daily_df[(daily_df["date"] >= start_ts) & (daily_df["date"] <= end_ts)]
           .sort_values("date").reset_index(drop=True))
    if len(win) < W:
        raise ValueError(
            f"{pred_date} 기준 {W}일 윈도우에 {len(win)}일만 존재 "
            f"({start_ts.date()} ~ {end_ts.date()})"
        )
    return win


def fetch_daily_for_target_window(lat, lon, target_date, n_days=16, verbose=True):
    today = date.today()
    end_date = target_date + timedelta(days=n_days - 1)

    if target_date > today:
        raise ValueError(
            f"❌ target_date={target_date}는 미래입니다.\n"
            f"   Open Meteo forecast 는 (오늘 + 16일) 까지만 가능."
        )

    if end_date <= today - timedelta(days=5):
        use_api, url = "ARCHIVE", ARCHIVE_API
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": target_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": ",".join(DAILY_VARS),
            "timezone": "auto", "wind_speed_unit": "ms",
        }
    else:
        past_days = min(max(0, (today - target_date).days), 92)
        forecast_days = min(max(1, (end_date - today).days + 1), 16)
        use_api, url = "FORECAST", FORECAST_API
        params = {
            "latitude": lat, "longitude": lon,
            "daily": ",".join(DAILY_VARS),
            "past_days": past_days, "forecast_days": forecast_days,
            "timezone": "auto", "wind_speed_unit": "ms",
        }

    if verbose: print(f"🌐 {use_api} API → {target_date} ~ {end_date} ({n_days}일)")
    r = _request_with_retry(url, params, verbose=verbose)
    d = r.json()["daily"]
    df = _build_daily_df_from_response(d)
    df = df[(df["date"] >= pd.Timestamp(target_date)) &
            (df["date"] <= pd.Timestamp(end_date))].reset_index(drop=True)
    if len(df) < n_days:
        raise RuntimeError(f"{n_days}일 중 {len(df)}일만 수신")
    if verbose: print(f"✅ daily {len(df)}일 수신")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  gdd_cum 자동 계산 (v2 동일)
# ═══════════════════════════════════════════════════════════════════════
def _fetch_tmean_series(lat, lon, start, end, verbose=False):
    today = date.today()
    df_list = []
    archive_end = min(end, today - timedelta(days=7))
    if archive_end >= start:
        try:
            r = _request_with_retry(ARCHIVE_API, {
                "latitude": lat, "longitude": lon,
                "start_date": start.isoformat(), "end_date": archive_end.isoformat(),
                "daily": "temperature_2m_mean", "timezone": "auto",
            }, verbose=verbose)
            d = r.json()["daily"]
            df_list.append(pd.DataFrame({"date": pd.to_datetime(d["time"]), "tmean": d["temperature_2m_mean"]}))
        except Exception as e:
            if verbose: print(f"   archive fetch 실패: {e}")
    forecast_start = max(start, today - timedelta(days=92))
    if end >= forecast_start:
        past_days = min(max(0, (today - forecast_start).days), 92)
        forecast_days = min(max(1, (end - today).days + 1), 16)
        try:
            r = _request_with_retry(FORECAST_API, {
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_mean",
                "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto",
            }, verbose=verbose)
            d = r.json()["daily"]
            df_list.append(pd.DataFrame({"date": pd.to_datetime(d["time"]), "tmean": d["temperature_2m_mean"]}))
        except Exception as e:
            if verbose: print(f"   forecast fetch 실패: {e}")
    if not df_list:
        raise RuntimeError("기온 데이터 fetch 실패")
    df = pd.concat(df_list, ignore_index=True)
    df["tmean"] = pd.to_numeric(df["tmean"], errors="coerce")
    return (df.dropna(subset=["tmean"]).drop_duplicates(subset="date", keep="first")
              .sort_values("date").reset_index(drop=True))


def compute_season_idx(target_date):
    diffs = [abs((target_date - date(target_date.year, m, d)).days)
             for (m, d) in STANDARD_SURVEY_MD]
    return diffs.index(min(diffs)) + 1


def compute_gdd_cum(lat, lon, target_date, todays_gdd_15, verbose=True):
    season_start = date(target_date.year, *SEASON_START_MD)
    if target_date < season_start:
        return float(todays_gdd_15)
    past_surveys = [date(target_date.year, m, d) for (m, d) in STANDARD_SURVEY_MD
                    if date(target_date.year, m, d) < target_date]
    if not past_surveys:
        return float(todays_gdd_15)
    tdf = _fetch_tmean_series(lat, lon, season_start - timedelta(days=14),
                              past_surveys[-1], verbose=verbose)
    past_sum = 0.0
    for sd in past_surveys:
        lo, hi = pd.Timestamp(sd - timedelta(days=14)), pd.Timestamp(sd)
        win = tdf[(tdf["date"] >= lo) & (tdf["date"] <= hi)]
        if len(win) == 0:
            if verbose: print(f"   {sd} 데이터 없음 skip")
            continue
        tm15 = win["tmean"].mean()
        g = max(0.0, tm15 - GDD_THRESHOLD)
        past_sum += g
        if verbose: print(f"   {sd}: temp_mean_15={tm15:.2f}, gdd_15={g:.2f}")
    return float(past_sum + todays_gdd_15)


# ═══════════════════════════════════════════════════════════════════════
#  학습 노트북 전처리 재현 (v2 동일)
# ═══════════════════════════════════════════════════════════════════════
def _transform_group_median_imputer(df_in, imputer):
    df_out = df_in.copy()
    keys = list(zip(*[df_out[g] for g in imputer['group_cols']]))
    for col in imputer['numeric_cols']:
        if col not in df_out.columns:
            df_out[col] = np.nan
        if df_out[col].isnull().any():
            mapped = pd.Series(keys, index=df_out.index).map(imputer['group_medians'][col])
            df_out[col] = df_out[col].fillna(mapped).fillna(imputer['global_medians'][col])
    return df_out


def _transform_features_with_meta(df_in, meta):
    d = df_in.copy()
    for col in meta['flag_source_cols']:
        if col not in d.columns:
            d[col] = np.nan
        d[f'{col}_isnull'] = d[col].isnull().astype(int)
    d = _transform_group_median_imputer(d, meta['imputer'])
    d['지역'] = pd.Categorical(d['지역'], categories=meta['region_categories'])
    d = pd.get_dummies(d, columns=['지역'], prefix='region', dtype=int)
    X = d.drop(columns=[c for c in ['날짜', TARGET] if c in d.columns])
    return X.reindex(columns=meta['feature_cols'], fill_value=0)


# ═══════════════════════════════════════════════════════════════════════
#  회의록 사양 — 정규화 · Grade · 상황 · 주요 원인
# ═══════════════════════════════════════════════════════════════════════
def normalize_p_ebm(raw_pred):
    """회의록 9p: P_EBM = clip(ŷ / y_p95_train, 0, 1)"""
    return float(np.clip(raw_pred / Y_P95_TRAIN, 0, 1))


def normalize_d(self_lag1):
    """회의록 9p: D = clip(self_lag1 / y_p95_train, 0, 1)"""
    return float(np.clip(self_lag1 / Y_P95_TRAIN, 0, 1))


def compute_risk_score(p_ebm, d, e):
    """회의록 9p: Risk Score = α × P_EBM + (1-α) × (D × E)"""
    return ALPHA * p_ebm + (1 - ALPHA) * (d * e)


def score_to_grade(score):
    """회의록 11p, 15p: 경계 b1, b2, b3 매핑"""
    if score < B1: return 0
    if score < B2: return 1
    if score < B3: return 2
    return 3


def apply_override(grade, p_ebm, d, e):
    """회의록 18p: Override 안전장치"""
    g = grade
    overrides = []
    if p_ebm >= P_EBM_OVERRIDE:
        if g < 2:
            overrides.append(f"P_EBM≥{P_EBM_OVERRIDE} → Grade≥2")
        g = max(g, 2)
    if d >= D_OVERRIDE and e >= E_OVERRIDE:
        if g < 3:
            overrides.append(f"D≥{D_OVERRIDE} AND E≥{E_OVERRIDE} → Grade 3")
        g = 3
    return g, overrides


def situation_message(p_ebm, d, e):
    """회의록 21p: EBM 낮/높음 × Risk(D×E) 낮/높음의 4-quadrant"""
    p_high = p_ebm >= B2
    de_high = (d * e) >= B1
    if not p_high and not de_high:
        return '안전', "환경이 안정적이며 병 발생 확률은 낮지만 예의 주시할 필요가 있습니다."
    if not p_high and de_high:
        return '주의', "단기적인 고온다습한 환경에 의해 병 발생 가능성이 커지고 있습니다."
    if p_high and not de_high:
        return '경계', "환경은 괜찮으나 장기적으로 병 발생 기반 위험이 높아 지속적인 관찰 및 관리가 필요합니다."
    return '심각', "누적 위험과 고온다습 환경으로 전체적인 위험이 크게 증가했습니다. 즉시 방제가 필요합니다."


def _ko_name(feat):
    if feat.startswith('region_'):
        return f"{feat.replace('region_', '')} 지역 효과"
    if feat.endswith('_isnull'):
        return None
    return FEATURE_KO.get(feat, feat)


def top_features(ebm, X_row, k=3):
    """회의록 22p: EBM local explanation → self_lag 제외 top-k (절댓값 기준)"""
    exp = ebm.explain_local(X_row)
    data = exp.data(0)
    pairs = []
    for name, score in zip(data['names'], data['scores']):
        if name in LAG_EXCLUDE: continue
        ko = _ko_name(name)
        if ko is None: continue
        pairs.append((ko, float(score)))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:k]


def top_features_positive(ebm, X_row, k=3):
    """위험도를 '높이는' 방향(양의 기여)으로 작용하는 feature top-k.
    self_lag, region_*, *_isnull 제외."""
    exp = ebm.explain_local(X_row)
    data = exp.data(0)
    pairs = []
    for name, score in zip(data['names'], data['scores']):
        if name in LAG_EXCLUDE: continue
        if float(score) <= 0: continue
        ko = _ko_name(name)
        if ko is None: continue
        pairs.append((ko, float(score)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]


def feature_phrases(top_feats):
    """회의록 22p: 한글 문구"""
    out = []
    for name, score in top_feats:
        direction = "높이는" if score > 0 else "낮추는"
        out.append(f"**{name}**이/가 병 발생 위험도를 {direction} 방향으로 작용하고 있습니다.")
    return out


# ═══════════════════════════════════════════════════════════════════════
#  리포트 카드 렌더링 (리포트_멘트_매트릭스.xlsx 룰 구현)
# ═══════════════════════════════════════════════════════════════════════
def _growth_stage(target_date):
    """월·일 → 벼 생육시기 (한줄요약_매트릭스 시트 참조).
    리포트 예시(5월 26일 = 분얼기)에 맞춘 한국 표준 작기 기준."""
    md = (target_date.month, target_date.day)
    if md < (5, 16): return '이앙기'
    if md < (7, 6):  return '분얼기'
    if md < (7, 26): return '유수형성기'
    if md < (8, 16): return '수잉기'
    if md < (9, 6):  return '출수기'
    return '등숙기'


def _r_t(temp):
    """CTM r(T): Yan-Hunt 곡선 (Tmin=22, Topt=31, Tmax=35) — your_ctm_module.compute_e 와 동일."""
    t_min, t_opt, t_max = 22.0, 31.0, 35.0
    try:
        t = float(temp)
    except (TypeError, ValueError):
        return 0.0
    if not (t_min < t < t_max):
        return 0.0
    alpha = (t_max - t) / (t_max - t_opt)
    beta  = (t - t_min) / (t_opt - t_min)
    exp   = (t_opt - t_min) / (t_max - t_opt)
    return float(alpha * (beta ** exp))


def _f_rh(rh):
    """CTM f(RH): Sigmoid (RH50=96) — your_ctm_module.compute_e 와 동일."""
    try:
        h = float(rh)
    except (TypeError, ValueError):
        return 0.0
    g = float(np.clip(-0.5 * (h - 96.0), -50, 50))
    return float(1.0 / (1.0 + np.exp(g)))


def _render_climate_card(feature, value, score):
    """EBM_멘트_매트릭스 시트 row 단위 분기. 매칭 없으면 None."""
    if feature not in FEATURE_GROUP:
        return None
    grp, grp_name, n, unit, _dir, short_ko = FEATURE_GROUP[feature]
    v = float(value)
    code = title = msg = None

    if grp == 'A':
        if feature.startswith('temp_mean'):
            if 25 <= v <= 32:
                code, title = 'A_HOT_OPT', '기온이 평년보다 높아요'
                msg = f'최근 {n}일 평균 {v:.1f}°C — 균이 가장 잘 자라는 25~30°C 구간이에요.'
            elif v > 35:
                code, title = 'A_HOT_OUT', '기온이 매우 높아요'
                msg = f'기온이 {v:.1f}°C로 균이 활동하는 범위를 넘었어요. 기온이 내려오면 다시 주의가 필요해요.'
            elif v < 22:
                code, title = 'A_COLD', '기온이 낮아요'
                msg = f'기온이 {v:.1f}°C로 균이 활동하기 어려운 상태예요.'
        elif feature.startswith('hot_days') and v >= 3:
            code, title = 'A_HOTDAYS', '더운 날이 많았어요'
            msg = f'최근 {n}일 중 {int(v)}일이 30°C를 넘었어요. 균이 폭발적으로 번지기 좋은 날씨가 반복됐어요.'
        elif feature.startswith('dtr_mean') and v > 10:
            code, title = 'A_DTR', '일교차가 커요'
            msg = f'일교차가 {v:.1f}°C예요. 밤에 이슬이 많이 맺혀 잎집 습도가 올라가요.'
        elif feature.startswith('soil_temp_mean') and v > BASELINES.get(feature, 20):
            code, title = 'G_TEMP', '토양 온도가 높아요'
            msg = f'토양 온도 {v:.1f}°C — 균 활동이 지표면 가까이서도 활발해질 수 있어요.'
        elif feature.startswith('tmax_max') and v > 32:
            code, title = 'A_HOT_OUT', '최고기온이 매우 높아요'
            msg = f'최고기온이 {v:.1f}°C로 균이 활동하는 범위 가까이까지 올랐어요.'

    elif grp == 'B':
        if feature.startswith('rh_mean'):
            base = BASELINES.get(feature, 74)
            diff = abs(v - base)
            if v > base:
                code, title = 'B_RH_HIGH', '습도가 높아졌어요'
                msg = f'최근 {n}일 평균 {v:.0f}% — 평년보다 {diff:.0f}% 더 습해요.'
            else:
                code, title = 'B_RH_LOW', '습도가 낮아졌어요'
                msg = f'최근 {n}일 평균 {v:.0f}% — 평년보다 {diff:.0f}% 덜 습해요.'
        elif feature.startswith('dew_mean') and score > 0:
            code, title = 'B_DEW', '이슬이 많이 맺혀요'
            msg = f'이슬점이 {v:.1f}°C로 높아요. 밤사이 잎집 표면에 수분이 오래 남아요.'
        elif feature.startswith('vapor_mean') and score > 0:
            code, title = 'B_VAPOR', '대기가 습해요'
            msg = '대기 중 수증기가 많아요. 밀식된 논일수록 포기 사이 습도가 더 빠르게 쌓여요.'
        elif feature.startswith('humid_days') and v >= 2:
            code, title = 'B_HUMDAYS', '습한 날이 많았어요'
            msg = f'최근 {n}일 중 {int(v)}일이 습도 90% 이상이었어요. 균이 침입하기 충분한 수분이 계속 공급됐어요.'

    elif grp == 'C':
        if feature.startswith('rain_sum'):
            base = BASELINES.get(feature, 0)
            if v > base:
                code, title = 'C_RAINSUM', '강수가 많았어요'
                msg = f'최근 {n}일 누적 {v:.0f}mm — 잎집이 오랫동안 젖어 있으면서 균이 옮겨붙기 쉬워요.'
        elif feature.startswith('rainy_days') and v >= 3:
            code, title = 'C_RAINDAYS', '비 오는 날이 많았어요'
            msg = f'최근 {n}일 중 {int(v)}일 비가 왔어요. 잎집이 마를 틈이 없었어요.'
        elif feature.startswith('heavy_rain_days') and v >= 1:
            code, title = 'C_HEAVY', '강한 비가 잦았어요'
            msg = f'강한 비({int(v)}일)가 균핵을 논 전체로 퍼뜨렸을 수 있어요.'

    elif grp == 'D':
        if feature.startswith('wind_mean'):
            if v < 2:
                code, title = 'D_LOW', '통풍이 부족해요'
                msg = f'평균 풍속 {v:.1f}m/s — 바람이 거의 없어요. 포기 사이 공기가 순환되지 않아 습기가 쌓이기 쉬워요.'
            else:
                code, title = 'D_OK', '바람이 적당히 불고 있어요'
                msg = f'풍속 {v:.1f}m/s — 통풍이 되면서 습도 누적을 막아줘요.'

    elif grp == 'E':
        if feature.startswith('sunshine_sum'):
            base = BASELINES.get(feature, 0)
            if v < base:
                code, title = 'E_LOW', '흐린 날이 많았어요'
                msg = f'최근 {n}일 일조 {v:.0f}시간 — 잎집이 마를 시간이 부족해 습도가 유지됐어요.'

    elif grp == 'F':
        if v >= 2:
            code, title = 'F_COMP', '고온다습한 날이 반복됐어요'
            msg = f'최근 {n}일 중 {int(v)}일이 고온다습한 날씨였어요. 이 기간 동안 균이 잎집에 침입하기 가장 좋은 조건이 이어졌어요.'

    elif grp == 'G':
        base = BASELINES.get(feature, 0)
        if feature.startswith('soil_moisture') and v > base:
            code, title = 'G_MOIST', '토양이 습해요'
            msg = f'논 토양 수분이 {v:.1f}로 높아요. 포기 아랫부분 잎집 주변 습도도 덩달아 올라가요.'

    if code is None:
        return None
    return {
        'type': 'weather',
        'group': grp,
        'group_label': grp_name,
        'subtitle': short_ko,
        'title': title,
        'message': msg,
        'condition_code': code,
        'feature': feature,
        'value': v,
        'contribution': float(score),
        'window_days': n,
    }


def _select_climate_cards(ebm, X_row, k=3):
    """카드_선택_로직 Step1~4: 그룹 dedup → |score| 상위 k개 → 카드 렌더."""
    exp = ebm.explain_local(X_row)
    data = exp.data(0)
    group_top = {}
    for name, score in zip(data['names'], data['scores']):
        if name in LAG_EXCLUDE:        continue
        if name.startswith('region_'): continue
        if name.endswith('_isnull'):   continue
        if name not in FEATURE_GROUP:  continue
        grp = FEATURE_GROUP[name][0]
        if grp == 'I':                 continue
        if grp not in group_top or abs(score) > abs(group_top[grp][1]):
            group_top[grp] = (name, float(score))

    sorted_groups = sorted(group_top.values(), key=lambda x: abs(x[1]), reverse=True)
    cards = []
    for feat, sc in sorted_groups:
        if len(cards) >= k: break
        if feat not in X_row.columns:  continue
        card = _render_climate_card(feat, float(X_row[feat].iloc[0]), sc)
        if card:
            cards.append(card)
    return cards


def _select_lag_card(ebm, X_row, d, self_lag1, self_lag2):
    """카드_선택_로직 Step5: lag 카드 분기."""
    if d <= 0:
        return None
    exp = ebm.explain_local(X_row)
    data = exp.data(0)
    score_map = {name: float(s) for name, s in zip(data['names'], data['scores'])}
    lag1_score = score_map.get('self_lag1', 0.0)

    if self_lag1 > 0 and abs(lag1_score) >= 0.05:
        return {
            'type': 'lag',
            'group': 'I',
            'group_label': '누적 위험',
            'subtitle': '누적 위험',
            'title': '2주 전부터 위험이 쌓이고 있었어요',
            'message': '15일 전 이 논의 위험 점수가 이미 꽤 높았어요. 한 번 번지기 시작한 병은 시간이 갈수록 빠르게 누적돼요.',
            'condition_code': 'I_LAG1',
            'contribution': lag1_score,
        }
    if self_lag2 > 0 and self_lag1 == 0:
        return {
            'type': 'lag',
            'group': 'I',
            'group_label': '누적 위험',
            'subtitle': '누적 위험',
            'title': '이전 발병이 지금 위험도를 높이고 있어요',
            'message': '30일 전 발병 기록이 있어요. 균핵이 아직 논에 남아 있을 수 있어요.',
            'condition_code': 'I_LAG2',
            'contribution': float(score_map.get('self_lag2', 0.0)),
        }
    return None


def _select_ctm_card(d, e, risk_score, temp_mean_3, rh_mean_3):
    """카드_선택_로직 Step6 + CTM_멘트_매트릭스 시트."""
    if d <= 0 or risk_score <= 0:
        return None
    ctm_ratio = ((1 - ALPHA) * d * e) / risk_score
    if ctm_ratio < 0.2:
        return None

    r_t  = _r_t(temp_mean_3)
    f_rh = _f_rh(rh_mean_3)
    code = title = msg = None

    if r_t >= 0.7 and f_rh >= 0.7:
        code, title = 'CTM_BOTH_HIGH', '온도와 습도가 동시에 위험 구간이에요'
        msg = (f'지금 기온({float(temp_mean_3):.1f}°C)과 습도({float(rh_mean_3):.0f}%)가 '
               '균이 가장 빠르게 번지는 조건이에요. 이미 발병이 시작된 상황에서 '
               '이런 환경이 이어지면 하루 안에 급격히 확산돼요.')
    elif r_t >= 0.7 and f_rh < 0.5:
        code, title = 'CTM_TEMP_HIGH', '기온이 균 활동에 최적이에요'
        msg = (f'기온 {float(temp_mean_3):.1f}°C — 균이 가장 활발하게 움직이는 구간이에요. '
               '습도까지 오르면 확산 속도가 급격히 빨라져요.')
    elif r_t < 0.5 and f_rh >= 0.7:
        code, title = 'CTM_RH_HIGH', '습도가 임계값을 넘었어요'
        msg = (f'습도 {float(rh_mean_3):.0f}%로 균이 잎집에 침입하기에 충분한 수분이 확보된 상태예요. '
               '이미 발병 중인 논에서 더 빠르게 번질 수 있어요.')
    elif 0.4 <= r_t < 0.7 and 0.4 <= f_rh < 0.7:
        code, title = 'CTM_MID', '환경이 균 활동을 돕고 있어요'
        msg = ('온도와 습도가 균이 활동하기 시작하는 수준이에요. '
               '지금 발병 중이라면 조건이 더 나빠지기 전에 방제하세요.')

    if code is None:
        return None
    return {
        'type': 'ctm',
        'group': 'CTM',
        'group_label': 'CTM 지수',
        'subtitle': 'CTM 지수',
        'title': title,
        'message': msg,
        'condition_code': code,
        'r_T': float(r_t),
        'f_RH': float(f_rh),
        'ctm_ratio': float(ctm_ratio),
    }


def _render_summary(p_ebm, e, grade_trend, growth_stage, top_feat_subtitle):
    """한줄요약_매트릭스 시트의 6개 분기 + [생육시기_위험설명] 치환."""
    stage_desc = GROWTH_STAGE_DESC.get(growth_stage, growth_stage)
    top = top_feat_subtitle or '단기 기상'
    p_high = p_ebm >= 0.5
    e_high = e >= 0.5

    if not p_high and not e_high:
        return 'SUM_SAFE', f'전반적으로 안정적이에요. 다만 {growth_stage}인 만큼 꾸준한 관찰이 필요해요.'
    if not p_high and e_high:
        if grade_trend == 'up':
            return ('SUM_ENV',
                    f'{top} 등 단기 기상 조건이 나빠지면서 {stage_desc}에 위험 점수가 빠르게 올랐어요.')
        return ('SUM_ENV2',
                f'{top} 등 단기 기상 조건이 나빠지면서 {stage_desc}에 위험 점수가 높은 상태가 이어지고 있어요.')
    if p_high and not e_high:
        return ('SUM_ACCUM',
                f'당장 날씨는 괜찮지만, {growth_stage} 동안 쌓인 위험이 높아요. 지속적인 관찰이 필요해요.')
    if grade_trend == 'up':
        return ('SUM_BOTH',
                f'고온다습이 이어지고 {stage_desc}에 들어서면서 위험 점수가 빠르게 올랐어요.')
    return ('SUM_BOTH2',
            f'고온다습이 이어지고 {stage_desc}에서 위험 점수가 높은 상태가 이어지고 있어요.')


def _grade_change_label(prev_grade, curr_grade):
    """위험 등급 변화 라벨 (리포트 상단 '주의 → 심각' 영역)."""
    if prev_grade is None:
        return None
    if curr_grade > prev_grade:  direction, label = 'up',   '위험 등급이 상승했어요'
    elif curr_grade < prev_grade: direction, label = 'down', '위험 등급이 하락했어요'
    else:                         direction, label = 'same', '위험 등급이 유지되고 있어요'
    return {
        'from':      GRADE_NAME[prev_grade],
        'from_code': int(prev_grade),
        'to':        GRADE_NAME[curr_grade],
        'to_code':   int(curr_grade),
        'direction': direction,
        'label':     label,
    }


# ═══════════════════════════════════════════════════════════════════════
#  모델 캐싱 (joblib.load 1회만)
# ═══════════════════════════════════════════════════════════════════════
_MODEL_CACHE: dict = {}


def _load_models():
    if "ebm" not in _MODEL_CACHE:
        _MODEL_CACHE["meta"] = joblib.load(MODEL_DIR / "feature_preprocess_meta.pkl")
        _MODEL_CACHE["ebm"] = joblib.load(MODEL_DIR / "ebm_final.pkl")
    return _MODEL_CACHE["meta"], _MODEL_CACHE["ebm"]


# ═══════════════════════════════════════════════════════════════════════
#  슬라이딩 윈도우 16일 예측
# ═══════════════════════════════════════════════════════════════════════
def _predict_at_date(
    daily_df,
    pred_date,
    info,
    meta,
    ebm,
    past_gdd_sum=0.0,
    self_lag1=0.0,
    self_lag2=0.0,
):
    """daily_df 에서 pred_date 로 끝나는 3/7/15일 윈도우로 단일 예측."""
    # 1. 각 윈도우 (pred_date 로 끝나는 W일) feature 계산
    feats = {"지역": info["region"]}
    for W in (3, 7, 15):
        window = _slice_window_ending_at(daily_df, pred_date, W)
        feats.update(OpenMeteoMultiWindowExtractor._compute_window_block(window, W))

    # 2. 날짜·시즌 메타
    feats["날짜"] = pd.to_datetime(pred_date)
    feats["year"] = pred_date.year
    feats["month"] = pred_date.month
    feats["dayofyear"] = pred_date.timetuple().tm_yday
    season_start = date(pred_date.year, *SEASON_START_MD)
    feats["days_since_season_start"] = max(0, (pred_date - season_start).days)
    feats["season_idx"] = compute_season_idx(pred_date)
    feats["gdd_15"] = float(max(0.0, feats["temp_mean_15"] - GDD_THRESHOLD))
    feats["gdd_cum"] = float(past_gdd_sum) + feats["gdd_15"]
    feats["self_lag1"] = 0.0 if pd.isna(self_lag1) else float(self_lag1)
    feats["self_lag2"] = 0.0 if pd.isna(self_lag2) else float(self_lag2)

    # 3. EBM 예측
    df_row = pd.DataFrame([feats])
    X = _transform_features_with_meta(df_row, meta)
    raw_pred = float(np.clip(np.expm1(ebm.predict(X)[0]), 0, None))
    p_ebm = normalize_p_ebm(raw_pred)
    d = normalize_d(feats["self_lag1"])

    # 4. CTM 환경 점수 E = compute_e(temp_mean_3, rh_mean_3)
    e_used = float(np.clip(
        compute_e(feats.get("temp_mean_3"), feats.get("rh_mean_3")),
        0, 1,
    ))

    # 5. Risk Score + Grade + Override
    risk_score = compute_risk_score(p_ebm, d, e_used)
    base_grade = score_to_grade(risk_score)
    final_grade, overrides = apply_override(base_grade, p_ebm, d, e_used)
    season_label, situation = situation_message(p_ebm, d, e_used)

    # 6. 리포트 카드 (리포트_멘트_매트릭스.xlsx 룰)
    climate_cards = _select_climate_cards(ebm, X, k=3)
    lag_card = _select_lag_card(ebm, X, d, feats["self_lag1"], feats["self_lag2"])
    ctm_card = _select_ctm_card(d, e_used, risk_score,
                                feats.get("temp_mean_3"), feats.get("rh_mean_3"))

    cards = list(climate_cards)
    if lag_card: cards.append(lag_card)
    if ctm_card: cards.append(ctm_card)
    for i, c in enumerate(cards, 1):
        c["no"] = i

    growth_stage = _growth_stage(pred_date)
    top_subtitle = climate_cards[0]["subtitle"] if climate_cards else None
    ctm_ratio = ((1 - ALPHA) * d * e_used) / risk_score if risk_score > 0 else 0.0

    return {
        # ── 필수 필드 (사용자 요청) ──
        "기준_날짜": pred_date.isoformat(),
        "기준_날짜_표시": f"{pred_date.month}월 {pred_date.day}일",
        "위험_등급": GRADE_NAME[final_grade],
        "위험_등급_코드": int(final_grade),
        "위험도": situation,
        "Risk_score": float(risk_score),

        # ── 리포트 카드 (앱 상단 표시) ──
        "summary": None,                   # 상위 루프에서 grade_trend 산출 후 채움
        "summary_code": None,
        "growth_stage": growth_stage,
        "growth_stage_desc": GROWTH_STAGE_DESC.get(growth_stage, growth_stage),
        "grade_change": None,              # 상위 루프에서 prev_grade 와 비교 후 채움
        "cards": cards,

        # ── 부가 메트릭 ──
        "metrics": {
            "P_EBM": float(p_ebm),
            "D": float(d),
            "E": float(e_used),
            "raw_pred": float(raw_pred),
            "self_lag1": float(feats["self_lag1"]),
            "self_lag2": float(feats["self_lag2"]),
            "ctm_ratio": float(ctm_ratio),
            "r_T": _r_t(feats.get("temp_mean_3")),
            "f_RH": _f_rh(feats.get("rh_mean_3")),
            "override_triggered": overrides,
            "situation_label": season_label,
        },
        "inputs": {
            "temp_mean_3": float(feats["temp_mean_3"]),
            "rh_mean_3": float(feats["rh_mean_3"]),
            "temp_mean_15": float(feats["temp_mean_15"]),
            "gdd_15": float(feats["gdd_15"]),
            "gdd_cum": float(feats["gdd_cum"]),
            "season_idx": int(feats["season_idx"]),
        },

        # ── 기존 키 호환성 (이전 JSON 소비자를 위해 별칭 유지) ──
        "target_date": pred_date.isoformat(),
        "grade": int(final_grade),
        "grade_name": GRADE_NAME[final_grade],
        "Risk_Score": float(risk_score),
    }


def predict_window_series(
    sido,
    today=None,
    n_future=15,
    self_lag1=0.0,
    self_lag2=0.0,
    verbose=True,
):
    """
    오늘부터 +n_future 일까지 (총 n_future+1 회) 슬라이딩 윈도우 예측.

    데이터: today 기준 과거 14일 + 오늘 + 예보 (n_future)일 = 30일.
    각 예측 시점 t (= today..today+n_future) 에서:
      - 3 / 7 / 15일 윈도우 모두 'pred_date 로 끝나는' 윈도우 (backward).
      - CTM E = compute_e(temp_mean_3, rh_mean_3)  (윈도우 안의 값으로 매번 갱신)
      - 양의 방향 top-3 EBM feature, self_lag, Risk_Score 저장.
    """
    if today is None:
        today = date.today()
    elif isinstance(today, str):
        today = date.fromisoformat(today)
    elif isinstance(today, datetime):
        today = today.date()

    info = get_sido_info(sido)
    if info["region"] is None:
        raise ValueError(f"'{sido}' 학습 데이터에 없는 지역")

    if verbose:
        print(f"\n{'═'*64}")
        print(f"  {sido}  (today={today}, +{n_future}일 예측)")
        print(f"{'═'*64}")

    # 1. 30일 daily 데이터 한 번에 fetch
    daily_df = fetch_30day_window(
        info["lat"], info["lon"], today,
        n_past=14, n_future=n_future, verbose=verbose,
    )

    # 2. 모델 (1회 캐시)
    meta, ebm = _load_models()

    # 3. gdd_cum: 추가 API 호출 없이 0으로 근사 (각 pred_date 에서 gdd_15 만 사용)
    #    → 시즌 누적 적산온도는 daily 배치 운영상 외부 상태로 관리하는 게 적절.
    #    무리한 ARCHIVE-API 호출은 GitHub Actions IP의 Open-Meteo 무료 한도(429)와
    #    502/timeout 의 주된 원인이라 제거.
    past_gdd_sum = 0.0

    # 4. 16개 예측 시점 순회
    predictions = []
    prev_grade = None
    for offset in range(n_future + 1):     # 0..n_future
        pred_date = today + timedelta(days=offset)
        try:
            r = _predict_at_date(
                daily_df, pred_date, info, meta, ebm,
                past_gdd_sum=past_gdd_sum,
                self_lag1=self_lag1, self_lag2=self_lag2,
            )
            r["offset_days"] = offset

            # 인접 offset 간 등급 변화 산출 (offset 0 은 prev 없음 → null)
            curr_grade = r["위험_등급_코드"]
            r["grade_change"] = _grade_change_label(prev_grade, curr_grade)
            trend = (r["grade_change"]["direction"]
                     if r["grade_change"] else 'same')

            # 한 줄 요약 (등급 추세 반영)
            top_subtitle = r["cards"][0]["subtitle"] if r["cards"] else None
            summary_code, summary = _render_summary(
                r["metrics"]["P_EBM"], r["metrics"]["E"],
                trend, r["growth_stage"], top_subtitle,
            )
            r["summary_code"] = summary_code
            r["summary"] = summary

            predictions.append(r)
            prev_grade = curr_grade
            if verbose:
                print(f"  · +{offset:2d}일 ({pred_date}): "
                      f"Risk={r['Risk_score']:.4f}  "
                      f"E={r['metrics']['E']:.4f}  "
                      f"grade={curr_grade} {r['위험_등급']}")
        except Exception as e:
            predictions.append({
                "offset_days": offset,
                "기준_날짜": pred_date.isoformat(),
                "target_date": pred_date.isoformat(),
                "error": str(e),
            })
            if verbose: print(f"  · +{offset:2d}일 ({pred_date}): ❌ {e}")

    return {
        "시도": sido,
        "sido": sido,
        "region": info["region"],
        "base_date": today.isoformat(),
        "n_future": n_future,
        "predictions": predictions,
    }


# ═══════════════════════════════════════════════════════════════════════
#  메인 예측 함수 (단일 날짜 — CLI 호환용)
# ═══════════════════════════════════════════════════════════════════════
def predict(
    sido,
    target_date=None,
    self_lag1=0.0,
    self_lag2=0.0,
    e_value=None,                  # ★ 회의록 사양: CTM 환경 점수 (제진님)
    prior_gdd_cum=None,
    use_mock_weather=False,
    verbose=True,
):
    """target_date 가 과거면 archive, today 면 forecast 자동 사용."""
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)
    elif isinstance(target_date, datetime):
        target_date = target_date.date()

    today = date.today()
    info = get_sido_info(sido)
    if info["region"] is None:
        raise ValueError(f"'{sido}' 학습 데이터에 없는 지역")

    if verbose:
        print(f"\n{'═'*64}")
        print(f"  예측 시점: {target_date}  ({sido})  [today={today}]")
        print(f"{'═'*64}")

    # 회의록 16p: self_lag 결측 처리
    self_lag1 = 0.0 if pd.isna(self_lag1) else float(self_lag1)
    self_lag2 = 0.0 if pd.isna(self_lag2) else float(self_lag2)

    # ── Feature 빌드 ────────────────────────────────────────────
    if use_mock_weather:
        ex = OpenMeteoMultiWindowExtractor.from_region(sido, windows=(3, 7, 15))
        ex.fetch_data(use_mock_data=True)
        feats = ex.compute_all_windows()
    else:
        daily_df = fetch_daily_for_target_window(
            info["lat"], info["lon"], target_date, n_days=16, verbose=verbose)
        feats = {"지역": info["region"]}
        for W in (3, 7, 15):
            feats.update(OpenMeteoMultiWindowExtractor._compute_window_block(daily_df, W))

    feats["날짜"] = pd.to_datetime(target_date)
    feats["year"] = target_date.year
    feats["month"] = target_date.month
    feats["dayofyear"] = target_date.timetuple().tm_yday
    season_start = date(target_date.year, *SEASON_START_MD)
    feats["days_since_season_start"] = max(0, (target_date - season_start).days)
    feats["season_idx"] = compute_season_idx(target_date)
    feats["gdd_15"] = float(max(0.0, feats["temp_mean_15"] - GDD_THRESHOLD))

    if prior_gdd_cum is not None:
        feats["gdd_cum"] = float(prior_gdd_cum) + feats["gdd_15"]
    elif use_mock_weather:
        feats["gdd_cum"] = feats["gdd_15"]
    else:
        if verbose: print("📐 gdd_cum: 자동 계산 중...")
        feats["gdd_cum"] = compute_gdd_cum(
            info["lat"], info["lon"], target_date, feats["gdd_15"], verbose=verbose)

    feats["self_lag1"] = self_lag1
    feats["self_lag2"] = self_lag2

    df_row = pd.DataFrame([feats])
    meta, ebm = _load_models()
    X = _transform_features_with_meta(df_row, meta)

    # ── 회의록 사양 계산 ────────────────────────────────────────
    raw_pred = float(np.clip(np.expm1(ebm.predict(X)[0]), 0, None))
    p_ebm = normalize_p_ebm(raw_pred)
    d = normalize_d(self_lag1)

    if e_value is None:
        # CTM 환경 점수 E 자동 계산: compute_e(temp_mean_3, rh_mean_3)
        e_used = float(np.clip(
            compute_e(feats.get("temp_mean_3"), feats.get("rh_mean_3")),
            0, 1,
        ))
        e_provided = True
    else:
        e_used = float(np.clip(e_value, 0, 1))
        e_provided = True

    risk_score = compute_risk_score(p_ebm, d, e_used)
    base_grade = score_to_grade(risk_score)
    final_grade, overrides = apply_override(base_grade, p_ebm, d, e_used)
    season_label, situation = situation_message(p_ebm, d, e_used)
    feats_top3 = top_features(ebm, X, k=3)

    # ── 출력 (회의록 24p 스타일) ────────────────────────────────
    if verbose:
        emoji = GRADE_EMOJI[final_grade]
        name = GRADE_NAME[final_grade]
        print(f"\n{'═'*64}")
        print(f"  {emoji} {name} ({final_grade}) · {sido}  {target_date}")
        print(f"{'═'*64}")
        print(f"\n  ▶ 입력")
        print(f"    self_lag1 = {self_lag1:.4f}    self_lag2 = {self_lag2:.4f}")
        print(f"    temp_mean_15 = {feats['temp_mean_15']:.2f}°C    gdd_cum = {feats['gdd_cum']:.2f}")
        if not e_provided:
            print(f"    E (CTM 환경 점수) = 미지정 (--e 옵션 필요)")
        else:
            print(f"    E (CTM 환경 점수) = {e_used:.4f}")
        print(f"\n  ▶ EBM 출력 (회의록 9p)")
        print(f"    raw ŷ    = {raw_pred:.4f}  ({raw_pred*100:.2f}%)")
        print(f"    P_EBM    = {p_ebm:.4f}   (= clip({raw_pred:.4f} / {Y_P95_TRAIN}, 0, 1))")
        print(f"    D        = {d:.4f}   (= clip({self_lag1:.4f} / {Y_P95_TRAIN}, 0, 1))")
        beta = round(1 - ALPHA, 2)
        if e_provided:
            print(f"\n  ▶ Risk Score (α=0.7 고정)")
            print(f"    = {ALPHA}×{p_ebm:.4f} + {beta}×({d:.4f}×{e_used:.4f})")
            print(f"    = {risk_score:.4f}")
        else:
            print(f"\n  ▶ Risk Score (E 미지정 → D×E=0 가정)")
            print(f"    = {ALPHA}×{p_ebm:.4f} + {beta}×0 = {risk_score:.4f}")
        print(f"\n  ▶ Grade (b1={B1}, b2={B2}, b3={B3})")
        print(f"    {emoji} 예측 Grade = {final_grade} ({name})")
        for ov in overrides:
            print(f"    ⚡ Override 발동: {ov}")
        print(f"\n  ▶ 상황 [{season_label}]")
        print(f"    {situation}")
        print(f"\n  ▶ 주요 원인 (self_lag 제외 top-3)")
        for phrase in feature_phrases(feats_top3):
            print(f"    · {phrase}")
        print(f"\n{'═'*64}\n")

    return {
        "sido": sido,
        "region": info["region"],
        "target_date": target_date.isoformat(),
        # ── EBM 출력 (회의록 사양) ──
        "raw_pred": raw_pred,
        "P_EBM": p_ebm,
        "D": d,
        "E": e_used if e_provided else None,
        "Risk_Score": risk_score,
        # ── Grade ──
        "grade": final_grade,
        "grade_name": GRADE_NAME[final_grade],
        "override_triggered": overrides,
        # ── 상황 ──
        "situation_label": season_label,
        "situation_message": situation,
        # ── 주요 원인 ──
        "top_features": [{"name": n, "contribution": c} for n, c in feats_top3],
        "explanation_phrases": feature_phrases(feats_top3),
        # ── 입력 ──
        "inputs": {
            "self_lag1": self_lag1,
            "self_lag2": self_lag2,
            "season_idx": int(feats["season_idx"]),
            "gdd_15": float(feats["gdd_15"]),
            "gdd_cum": float(feats["gdd_cum"]),
            "temp_mean_15": float(feats["temp_mean_15"]),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  Firebase 초기화 (counter.py 설정 그대로, Firestore 제외)
# ═══════════════════════════════════════════════════════════════════════
FIREBASE_KEY_PATH = "firebase_service_account.json"


def init_firebase():
    """Firebase Admin SDK 초기화 (이미 초기화돼 있으면 skip)."""
    if firebase_admin._apps:
        return
    if not os.path.exists(FIREBASE_KEY_PATH):
        print(f"⚠️ Firebase 키 파일 없음 ({FIREBASE_KEY_PATH}) — 초기화 건너뜀")
        return
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)


# ═══════════════════════════════════════════════════════════════════════
#  배치: 전국 광역단체별 일일 예측 → Firebase Hosting용 JSON
# ═══════════════════════════════════════════════════════════════════════
def main():
    import time
    init_firebase()

    output_dir = os.path.join("public", "regions")
    os.makedirs(output_dir, exist_ok=True)

    target_sido_list = list_sido()
    print(f"🎯 오늘 연산할 광역단체 수: {len(target_sido_list)}개")

    # 모델 1회 미리 로드 (전 지역 공유)
    _load_models()

    for sido in target_sido_list:
        info = KR_SIDO[sido]
        # 학습 데이터에 없는 지역(region=None)은 건너뜀
        if info["region"] is None:
            print(f"⏭️ [{sido}] 학습 데이터 미포함 → skip")
            continue

        print(f"\n🌾 [{sido}] 16일 슬라이딩 윈도우 예측 시작...")
        try:
            result = predict_window_series(
                sido=sido, n_future=15, verbose=False,
            )
            result["updated_at"] = datetime.now().isoformat()

            file_path = os.path.join(output_dir, f"{sido}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4, default=str)

            n_ok = sum(1 for p in result["predictions"] if "error" not in p)
            n_total = len(result["predictions"])
            print(f"✅ [{sido}] {n_ok}/{n_total}일 예측 완료 → {file_path}")
        except Exception as e:
            print(f"❌ [{sido}] 예측 실패: {e}")

        # Open-Meteo rate limit 완화 (지역 사이 쉬는 시간)
        # GitHub Actions IP가 Open-Meteo 무료 한도를 공유하므로 다소 길게.
        time.sleep(3.0)

    print("\n🚀 모든 일일 배치 작업이 완료되었습니다!")


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════
def _cli():
    p = argparse.ArgumentParser(
        description="EBM 잎집무늬마름병 위험도 예측 (v3: 11차 회의록 사양)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python3 predict.py 전북특별자치도                                        # 오늘
  python3 predict.py 전북특별자치도 --date 2024-08-16                       # 과거
  python3 predict.py 전북특별자치도 --date 2024-08-16 --lag1 0.036          # 과거 + lag
  python3 predict.py 전북특별자치도 --date 2024-08-16 --lag1 0.036 --e 0.45 # 과거 + lag + E
""")
    p.add_argument("sido", help="광역단체 풀네임")
    p.add_argument("--date", default=None)
    p.add_argument("--lag1", type=float, default=0.0, help="직전 조사 피해율 (없으면 0)")
    p.add_argument("--lag2", type=float, default=0.0, help="그 전 조사 피해율 (없으면 0)")
    p.add_argument("--e", type=float, default=None,
                   help="CTM 환경 점수 E (제진님 산출, 0~1). 미지정 시 D×E=0.")
    p.add_argument("--prior-gdd-cum", type=float, default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--json", action="store_true", help="JSON만 출력 (--quiet 자동 적용)")
    args = p.parse_args()

    target_date = date.fromisoformat(args.date) if args.date else None
    verbose = not (args.quiet or args.json)
    result = predict(
        sido=args.sido, target_date=target_date,
        self_lag1=args.lag1, self_lag2=args.lag2,
        e_value=args.e,
        prior_gdd_cum=args.prior_gdd_cum,
        use_mock_weather=args.mock, verbose=verbose,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    import sys
    # 인자 없이 실행되면 전국 배치 모드 (GitHub Action 일일 실행용)
    # 인자가 있으면 기존 단일 sido CLI 모드
    if len(sys.argv) == 1:
        main()
    else:
        _cli()
