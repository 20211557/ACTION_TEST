"""
compute_baselines.py — 2014~2023년 6~9월 데이터에서 평년값(중앙값) 산출
─────────────────────────────────────────────────────────────────────
출력: BASELINES dict (predict.py 에 그대로 붙여넣기)
대상 변수: rh_mean_{3,7,15}, rain_sum_{3,7,15}, sunshine_sum_{3,7,15},
           soil_moisture_mean_{3,7,15}, soil_temp_mean_{3,7,15}

방법:
  1. 15개 학습 지역(KR_SIDO region!=None) 각각에 대해
     Open-Meteo Archive API 로 2014-05-18 ~ 2023-10-01 daily 데이터 fetch.
  2. 일별 → 3/7/15일 rolling window (open_meteo_3_7_15.py 와 동일 단위 변환).
  3. window 끝점이 6~9월인 표본만 모아 median 산출.
"""
import json
import ssl
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

# msys2 Python 에 CA 번들이 없어서 verify 실패 → 로컬 1회용 스크립트로 verify 우회
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"

# kr_regions.py 의 학습 지역(region != None) 만 추출
LEARN_REGIONS = {
    "부산": (35.1796, 129.0756),
    "대구": (35.8722, 128.6025),
    "인천": (37.4563, 126.7052),
    "광주": (35.1601, 126.8514),
    "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114),
    "세종": (36.4801, 127.2890),
    "경기": (37.2750, 127.0095),
    "강원": (37.8813, 127.7298),
    "충북": (36.6357, 127.4914),
    "충남": (36.6588, 126.6731),
    "전북": (35.8242, 127.1480),
    "전남": (34.8161, 126.4630),
    "경북": (36.5760, 128.5057),
    "경남": (35.2381, 128.6924),
}

DAILY_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "sunshine_duration",
    "soil_temperature_0_to_7cm_mean",
    # Archive API 에서는 0_to_10cm 이 없고 0_to_7cm 만 제공 (Open-Meteo 정책).
    # 운영 시점 forecast API 의 0_to_10cm 와 깊이 3cm 차이가 있으나
    # 6~9월 전국 median 기준선 용도로는 충분히 근사 가능.
    "soil_moisture_0_to_7cm_mean",
    "et0_fao_evapotranspiration",
]

START = "2014-05-18"
END   = "2023-10-01"

# open_meteo_3_7_15.py 의 단위 보정 상수
SUNSHINE_SCALE     = 1.0 / 360   # 초 → 0.1시간(deci-hour)
SOIL_MOISTURE_MULT = 65          # 학습 데이터 분포 중위 맞추기 휴리스틱

WINDOWS = (3, 7, 15)


import os
CACHE_DIR = "_baseline_cache"


def fetch_region(region, lat, lon):
    """단일 지역에 대해 2014-05-18 ~ 2023-10-01 daily 데이터 fetch + 디스크 캐시."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{region}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            daily = json.load(f)
        print(f"  ⚡ [{region}] cache hit ({len(daily['time'])} rows)")
        return daily

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START,
        "end_date": END,
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "wind_speed_unit": "ms",
    }
    url = ARCHIVE_API + "?" + urllib.parse.urlencode(params)
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(url, timeout=180, context=_SSL_CTX) as r:
                data = json.loads(r.read().decode("utf-8"))
            daily = data["daily"]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(daily, f)
            print(f"  ✓ [{region}] daily {len(daily['time'])} rows (cached)")
            return daily
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = 15 * attempt   # 429 에 더 길게
            print(f"  ⚠️ [{region}] fetch fail (attempt {attempt}/4): {e} — wait {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"[{region}] 4회 시도 실패")


def rolling_window_values(daily, var, W, transform=None):
    """
    daily[var] 의 W일 rolling window. transform 이 'sum' 이면 합, 그 외 'mean'.
    window 끝점 인덱스 i (≥ W-1) 의 (end_date, value) 리스트 반환.
    None / NaN 은 건너뜀 (W개 모두 유효해야 표본으로 인정).
    """
    times = daily["time"]
    vals  = daily[var]
    out = []
    for i in range(W - 1, len(vals)):
        window = vals[i - W + 1: i + 1]
        if any(v is None for v in window):
            continue
        if transform == "sum":
            value = sum(window)
        else:
            value = sum(window) / W
        end_date = date.fromisoformat(times[i])
        out.append((end_date, value))
    return out


def filter_summer(samples):
    """end_date 가 6~9월인 표본만."""
    return [v for (d, v) in samples if 6 <= d.month <= 9]


def main():
    # 지역별로 fetch → 변수·윈도우별 표본 누적
    bucket = {f"{var}_{W}": [] for var in ["rh_mean", "rain_sum", "sunshine_sum",
                                            "soil_moisture_mean", "soil_temp_mean",
                                            "evaporation_mean"]
                                 for W in WINDOWS}
    for region, (lat, lon) in LEARN_REGIONS.items():
        try:
            daily = fetch_region(region, lat, lon)
        except Exception as e:
            print(f"  ❌ [{region}] skip: {e}")
            continue

        for W in WINDOWS:
            # rh_mean
            samples = filter_summer(
                rolling_window_values(daily, "relative_humidity_2m_mean", W, "mean")
            )
            bucket[f"rh_mean_{W}"].extend(samples)

            # rain_sum
            samples = filter_summer(
                rolling_window_values(daily, "precipitation_sum", W, "sum")
            )
            bucket[f"rain_sum_{W}"].extend(samples)

            # sunshine_sum (단위 변환: 초 → 0.1시간)
            samples = filter_summer(
                rolling_window_values(daily, "sunshine_duration", W, "sum")
            )
            bucket[f"sunshine_sum_{W}"].extend([v * SUNSHINE_SCALE for v in samples])

            # soil_moisture_mean (× 65 단위 보정)
            samples = filter_summer(
                rolling_window_values(daily, "soil_moisture_0_to_7cm_mean", W, "mean")
            )
            bucket[f"soil_moisture_mean_{W}"].extend([v * SOIL_MOISTURE_MULT for v in samples])

            # soil_temp_mean
            samples = filter_summer(
                rolling_window_values(daily, "soil_temperature_0_to_7cm_mean", W, "mean")
            )
            bucket[f"soil_temp_mean_{W}"].extend(samples)

            # evaporation_mean (FAO ET0, mm/day, W일 평균)
            samples = filter_summer(
                rolling_window_values(daily, "et0_fao_evapotranspiration", W, "mean")
            )
            bucket[f"evaporation_mean_{W}"].extend(samples)

        # rate-limit 완화
        time.sleep(2.5)

    # 중앙값 산출
    print("\n" + "=" * 70)
    print("  BASELINES (2014~2023년 6~9월 중앙값, 전국 학습 지역 통합)")
    print("=" * 70)
    result = {}
    for key in sorted(bucket.keys()):
        vals = bucket[key]
        if not vals:
            print(f"  {key:30s}: (no samples)")
            continue
        med = statistics.median(vals)
        result[key] = med
        print(f"  {key:30s}: n={len(vals):>6d}  median={med:>10.4f}")

    print("\n# 복붙용 BASELINES dict")
    print("BASELINES = {")
    for key in sorted(result.keys()):
        v = result[key]
        if "rh_mean" in key:
            print(f"    '{key}': {round(v, 1)},")
        elif "rain_sum" in key or "sunshine_sum" in key:
            print(f"    '{key}': {round(v, 1)},")
        else:
            print(f"    '{key}': {round(v, 2)},")
    print("}")

    with open("baselines_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n💾 baselines_result.json 저장")


if __name__ == "__main__":
    main()
