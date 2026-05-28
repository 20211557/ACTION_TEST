"""
Open Meteo 16일 예보 → 3일 / 7일 / 15일 윈도우 변수 동시 추출

용도:
    EBM 모델 운영 추론 시점에서 input 피처를 만드는 스크립트.
    오늘을 기준으로 16일 예보를 받아오고, 그 안에서
        - 앞 3일치 → *_3
        - 앞 7일치 → *_7
        - 앞 15일치 → *_15
    의 윈도우 집계 변수를 한 번에 모두 계산합니다.

    학습 시 merged_df.csv 의 컬럼 컨벤션과 동일한 이름으로 출력하므로,
    그대로 EBM 모델의 input 으로 사용할 수 있습니다.
"""

import math
from datetime import datetime
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

from kr_regions import KR_SIDO, get_sido_info, is_learned, list_sido


# ────────────────────────────────────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────────────────────────────────────

WINDOWS = (3, 7, 15)

THRESHOLDS = {
    'hot_temp': 30,           # 폭염     : tmax  >= 30
    'cold_temp': 0,           # 한파     : tmin  <=  0
    'humid_threshold': 80,    # 습한날    : rh    >= 80
    'hot_humid_temp': 27,     # 습열      : tmax  >= 27 AND
    'hot_humid_rh':   60,     #             rh    >= 60
    'wet_cool_temp':  10,     # 습+한랭   : tmin  <= 10 AND precip > 0.1
    'rainy_threshold': 0.1,   # 강수일    : precip >= 0.1
    'heavy_rain':     20,     # 집중호우  : precip >= 20
}


# ────────────────────────────────────────────────────────────────────────────
# 핵심 클래스
# ────────────────────────────────────────────────────────────────────────────

class OpenMeteoMultiWindowExtractor:
    """Open Meteo 16일 예보 → {3,7,15}일 윈도우 변수 추출"""

    API_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(
        self,
        latitude: float,
        longitude: float,
        windows: Iterable[int] = WINDOWS,
        fetch_days: int = 16,
        region: Optional[str] = None,
        sido: Optional[str] = None,
    ):
        self.latitude   = latitude
        self.longitude  = longitude
        self.windows    = tuple(sorted(set(windows)))
        self.fetch_days = max(fetch_days, max(self.windows))
        self.fetch_days = min(self.fetch_days, 16)
        self.daily_df   = None

        # 지역 메타 (모델 input 의 '지역' 컬럼에 들어감)
        self.region = region    # 학습 데이터의 '지역' 컬럼 값 (예: '전북')
        self.sido   = sido      # 풀네임 (예: '전북특별자치도')

    # ─────────────── 광역단체 이름으로 인스턴스 생성 ───────────────

    @classmethod
    def from_region(
        cls,
        sido: str,
        windows: Iterable[int] = WINDOWS,
        fetch_days: int = 16,
        strict: bool = False,
    ) -> "OpenMeteoMultiWindowExtractor":
        """
        광역단체 이름으로 인스턴스 생성.
        - 좌표는 해당 시·도청 위치 (KR_SIDO 매핑에서 자동 조회)
        - 학습되지 않은 시·도(서울/제주) 면 경고. strict=True 면 예외.

        Examples
        --------
        >>> ex = OpenMeteoMultiWindowExtractor.from_region("전북특별자치도")
        >>> ex.fetch_data()
        """
        info = get_sido_info(sido)
        region_value = info["region"]

        if region_value is None:
            msg = (f"⚠️ '{sido}' 는 학습 데이터에 포함되지 않은 지역입니다. "
                   f"예측 정확도가 낮을 수 있습니다.")
            if strict:
                raise ValueError(msg)
            print(msg)

        print(f"📍 {sido} → 좌표 ({info['lat']:.4f}, {info['lon']:.4f}), "
              f"학습 지역코드: {region_value}")

        return cls(
            latitude=info["lat"], longitude=info["lon"],
            windows=windows, fetch_days=fetch_days,
            region=region_value, sido=sido,
        )

    # ─────────────── 1. API 호출 ───────────────

    def fetch_data(self, use_mock_data: bool = False) -> pd.DataFrame:
        if use_mock_data:
            return self._generate_mock_data()

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "relative_humidity_2m_mean",
                "dew_point_2m_mean",
                "wind_speed_10m_mean",
                "wind_speed_10m_max",
                "wind_direction_10m_dominant",
                "precipitation_sum",
                "sunshine_duration",
                "shortwave_radiation_sum",
                "soil_temperature_0_to_7cm_mean",
                "soil_moisture_0_to_10cm_mean",
                "et0_fao_evapotranspiration",
            ]),
            "forecast_days": self.fetch_days,
            "timezone": "auto",
            "wind_speed_unit": "ms",
        }

        print(f"🌐 Open Meteo 요청 ({self.latitude}, {self.longitude}, {self.fetch_days}일)")
        try:
            r = requests.get(self.API_URL, params=params, timeout=15)
            r.raise_for_status()
            d = r.json()["daily"]
            self.daily_df = pd.DataFrame({
                "date":              pd.to_datetime(d["time"]),
                "tmax":              d["temperature_2m_max"],
                "tmin":              d["temperature_2m_min"],
                "tmean":             d["temperature_2m_mean"],
                "rh_mean":           d["relative_humidity_2m_mean"],
                "dew_point":         d["dew_point_2m_mean"],
                "wind_speed_mean":   d["wind_speed_10m_mean"],
                "wind_speed_max":    d["wind_speed_10m_max"],
                "wind_dir":          d["wind_direction_10m_dominant"],
                "precipitation":     d["precipitation_sum"],
                "sunshine_duration": d["sunshine_duration"],
                "solar_radiation":   d["shortwave_radiation_sum"],
                "soil_temp":         d["soil_temperature_0_to_7cm_mean"],
                "soil_moisture":     d["soil_moisture_0_to_10cm_mean"],
                "evaporation":       d["et0_fao_evapotranspiration"],
            })
            print(f"✅ 일별 데이터 수신: {len(self.daily_df)}일")
            return self.daily_df
        except requests.RequestException as e:
            print(f"❌ API 실패: {e}\n💡 모의 데이터로 대체합니다.")
            return self._generate_mock_data()

    def _generate_mock_data(self) -> pd.DataFrame:
        """오프라인 / 테스트용 모의 일별 데이터"""
        n = self.fetch_days
        rng = np.random.default_rng(42)
        x = np.arange(n)
        dates = pd.date_range(datetime.now().date(), periods=n, freq="D")

        self.daily_df = pd.DataFrame({
            "date":              dates,
            "tmax":              22 + 6 * np.sin(x / 4) + rng.normal(0, 1.5, n),
            "tmin":              12 + 5 * np.sin(x / 4) + rng.normal(0, 1.2, n),
            "tmean":             17 + 5 * np.sin(x / 4) + rng.normal(0, 1.0, n),
            "rh_mean":           np.clip(65 + 12 * np.sin(x / 3) + rng.normal(0, 4, n), 20, 100),
            "dew_point":         10 + 4 * np.sin(x / 4) + rng.normal(0, 1, n),
            "wind_speed_mean":   np.clip(2.5 + 1.0 * np.abs(np.sin(x / 3)) + rng.normal(0, 0.4, n), 0, None),
            "wind_speed_max":    np.clip(4.0 + 1.5 * np.abs(np.sin(x / 3)) + rng.normal(0, 0.6, n), 0, None),
            "wind_dir":          rng.uniform(0, 360, n),
            "precipitation":     np.where(rng.random(n) > 0.65, rng.exponential(4, n), 0.0),
            "sunshine_duration": np.clip(28000 + 12000 * np.sin(x / 4) + rng.normal(0, 3000, n), 0, None),
            "solar_radiation":   np.clip(16 + 8 * np.sin(x / 4) + rng.normal(0, 1.5, n), 0, None),
            "soil_temp":         14 + 3 * np.sin(x / 4) + rng.normal(0, 0.4, n),
            "soil_moisture":     np.clip(0.26 + 0.04 * np.sin(x / 3) + rng.normal(0, 0.02, n), 0, 1),
            "evaporation":       np.clip(2.5 + 1.0 * np.sin(x / 4) + rng.normal(0, 0.3, n), 0, None),
        })
        print(f"📊 모의 일별 데이터 생성: {len(self.daily_df)}일")
        return self.daily_df

    # ─────────────── 2. 윈도우 변수 계산 ───────────────

    @staticmethod
    def _compute_window_block(d: pd.DataFrame, W: int) -> Dict[str, float]:
        """daily 데이터의 앞 W일에 대해 모든 집계 변수를 계산 → dict 반환"""

        s = d.head(W).copy()
        out: Dict[str, float] = {}

        # 일별 파생값
        s["dtr"] = s["tmax"] - s["tmin"]

        # 증기압 (hPa) - WMO 권장 Magnus 공식 (Sonntag 1990)
        #   e = 6.112 * exp((17.62 * Td) / (243.12 + Td))
        # Bolton(1980) 으로 바꾸려면 17.62 → 17.67, 243.12 → 243.5
        s["vapor"] = 6.112 * np.exp((17.62 * s["dew_point"]) / (243.12 + s["dew_point"]))

        # 기온
        out[f"temp_mean_{W}"] = s["tmean"].mean()
        out[f"tmax_max_{W}"]  = s["tmax"].max()
        out[f"tmin_min_{W}"]  = s["tmin"].min()
        out[f"dtr_mean_{W}"]  = s["dtr"].mean()
        out[f"hot_days_{W}"]  = int((s["tmax"] >= THRESHOLDS["hot_temp"]).sum())
        out[f"cold_days_{W}"] = int((s["tmin"] <= THRESHOLDS["cold_temp"]).sum())

        # 습도 / 이슬점 / 증기압
        out[f"rh_mean_{W}"]    = s["rh_mean"].mean()
        out[f"dew_mean_{W}"]   = s["dew_point"].mean()
        out[f"vapor_mean_{W}"] = s["vapor"].mean()
        out[f"humid_days_{W}"] = int((s["rh_mean"] >= THRESHOLDS["humid_threshold"]).sum())

        # 복합 일수
        out[f"hot_humid_days_{W}"] = int((
            (s["tmax"]    >= THRESHOLDS["hot_humid_temp"]) &
            (s["rh_mean"] >= THRESHOLDS["hot_humid_rh"])
        ).sum())
        out[f"wet_cool_days_{W}"] = int((
            (s["precipitation"] >  THRESHOLDS["rainy_threshold"]) &
            (s["tmin"]          <= THRESHOLDS["wet_cool_temp"])
        ).sum())

        # 풍속
        # ⚠️ wind_sum 학습 데이터 단위 매칭:
        #    학습 데이터의 wind_sum_W / (wind_mean_W × W) ≈ 610 으로 거의 일정.
        #    학습 원천이 10분 또는 시간별 풍속 누적값으로 추정됨.
        #    분포 중심 맞추기 위해 × 610 배율 적용 (휴리스틱).
        WIND_SUM_SCALE = 610
        out[f"wind_mean_{W}"]     = s["wind_speed_mean"].mean()
        out[f"wind_sum_{W}"]      = s["wind_speed_mean"].sum() * WIND_SUM_SCALE
        out[f"wind_mean_max_{W}"] = s["wind_speed_mean"].max()

        # 풍향 (벡터 평균)
        rad = np.radians(s["wind_dir"])
        out[f"wind_dir_sin_mean_{W}"] = np.sin(rad).mean()
        out[f"wind_dir_cos_mean_{W}"] = np.cos(rad).mean()

        # 강수
        out[f"rain_sum_{W}"]        = s["precipitation"].sum()
        out[f"rain_max_{W}"]        = s["precipitation"].max()
        out[f"rainy_days_{W}"]      = int((s["precipitation"] >= THRESHOLDS["rainy_threshold"]).sum())
        out[f"heavy_rain_days_{W}"] = int((s["precipitation"] >= THRESHOLDS["heavy_rain"]).sum())

        # 일조 / 일사
        # ⚠️ sunshine_duration 은 Open Meteo 에서 초(s) 단위로 반환됨.
        #    학습 데이터(merged_df.csv)는 기상청 ASOS 기반으로 0.1시간(deci-hour) 단위로 추정되므로
        #    ÷360 (= 3600/10) 으로 단위 통일.
        out[f"sunshine_sum_{W}"] = s["sunshine_duration"].sum() / 360
        out[f"solar_sum_{W}"]    = s["solar_radiation"].sum()
        out[f"solar_max_{W}"]    = s["solar_radiation"].max()

        # 증발산 (FAO ET0, mm/day) — W일 평균
        if "evaporation" in s.columns:
            out[f"evaporation_mean_{W}"] = s["evaporation"].mean()

        # 토양
        # ⚠️ soil_moisture 학습 데이터 단위 매칭:
        #    Open Meteo: m³/m³ (0~1 비율)
        #    학습 데이터: 분포 중위 ≈ 16, 단순 ×100 도 안 맞음 (측정 방식 차이 추정)
        #    학습 중위에 맞춰 × 65 배율 적용 (휴리스틱).
        SOIL_MOISTURE_SCALE = 65
        out[f"soil_moisture_mean_{W}"] = s["soil_moisture"].mean() * SOIL_MOISTURE_SCALE
        out[f"soil_temp_mean_{W}"]     = s["soil_temp"].mean()

        return out

    def compute_all_windows(self) -> Dict[str, float]:
        """모든 윈도우의 변수를 평탄한 dict 로 반환 (지역 메타 포함)"""
        if self.daily_df is None:
            self.fetch_data()
        result: Dict[str, float] = {}
        # 지역 메타 정보 (모델 input 으로 들어갈 수 있도록 앞에 배치)
        if self.region is not None:
            result["지역"] = self.region
        for W in self.windows:
            result.update(self._compute_window_block(self.daily_df, W))
        return result

    # ─────────────── 3. 출력 도우미 ───────────────

    def to_dataframe(self, results: Dict) -> pd.DataFrame:
        """단일 행 DataFrame (운영 시점 모델 input 형식)"""
        return pd.DataFrame([results])

    def to_long_dataframe(self, results: Dict) -> pd.DataFrame:
        """윈도우/변수별 longform DataFrame (사람이 보기 편한 형태)"""
        rows = []
        for k, v in results.items():
            # 변수명에서 마지막 '_{W}' 분리
            *base_parts, w = k.rsplit("_", 1)
            base = "_".join(base_parts)
            rows.append({"variable": base, "window": int(w), "value": v})
        return (pd.DataFrame(rows)
                  .pivot(index="variable", columns="window", values="value")
                  .reindex(columns=list(self.windows))
                  .round(3))

    def save(self, results: Dict, path: str = "weather_features_3_7_15.csv") -> None:
        self.to_dataframe(results).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 저장 완료: {path}")


# ────────────────────────────────────────────────────────────────────────────
# 사용 예제
# ────────────────────────────────────────────────────────────────────────────

def _prompt_sido() -> str:
    """
    데모용 콘솔 메뉴: 광역단체 번호로 선택.
    UI 없이 테스트할 때 사용 (UI가 붙으면 from_region 직접 호출).
    """
    sidos = list_sido()
    print("\n" + "=" * 50)
    print("  광역단체 선택  (✓=학습됨, ✗=학습 안 됨)")
    print("=" * 50)
    for i, s in enumerate(sidos, 1):
        flag = "✓" if is_learned(s) else "✗"
        print(f"  {i:2d}. [{flag}] {s}")
    print("=" * 50)

    while True:
        raw = input(f"번호 입력 (1-{len(sidos)}, Enter=기본 전북): ").strip()
        if raw == "":
            return "전북특별자치도"   # 기본값
        if raw.isdigit() and 1 <= int(raw) <= len(sidos):
            return sidos[int(raw) - 1]
        print("⚠️  올바른 번호를 입력하세요.")


if __name__ == "__main__":
    import sys

    # ─────────────────────────────────────────────────────────────
    # 지역 결정 우선순위:
    #   1) CLI 인자  : python open_meteo_3_7_15.py 전북특별자치도
    #   2) 인터랙티브 메뉴 (인자 없을 때 콘솔에서 골라쓰기)
    # ─────────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        sido = " ".join(sys.argv[1:])   # 공백 포함 이름 대비
    else:
        sido = _prompt_sido()

    # ─────────────────────────────────────────────────────────────
    # 광역단체 이름으로 인스턴스 생성 → 추출 → 저장
    # ─────────────────────────────────────────────────────────────
    ex = OpenMeteoMultiWindowExtractor.from_region(
        sido=sido,
        windows=(3, 7, 15),
    )
    ex.fetch_data(use_mock_data=False)

    feats = ex.compute_all_windows()
    print(f"\n✅ 추출 변수 개수: {len(feats)}  (지역 메타 + 26 × {len(ex.windows)} 윈도우)")

    print("\n📊 윈도우별 비교표")
    print("=" * 70)
    numeric_feats = {k: v for k, v in feats.items() if isinstance(v, (int, float, np.floating))}
    print(ex.to_long_dataframe(numeric_feats))

    # 파일명에 지역 포함 → 여러 지역 데모 시 결과 덮어쓰기 방지
    region_tag = ex.region or "unknown"
    ex.save(feats, f"weather_features_3_7_15_{region_tag}.csv")
