import os
import json
import requests
from datetime import datetime

# 주요 지역 위경도 매핑 (Open-Meteo 기준)
REGIONS = {
    "seoul": {"lat": 37.5665, "lon": 126.9780},
    "busan": {"lat": 35.1796, "lon": 129.0756},
    "jeju": {"lat": 33.4996, "lon": 126.5312},
    "daegu": {"lat": 35.8714, "lon": 128.6014}
}

def fetch_weather_for_region(lat, lon):
    # EBM 분석에 필수적인 기초 기상 데이터 요청 (기온, 상대습도, 강수량, 풍속)
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&"
        f"timezone=Asia%2FTokyo"
    )
    
    response = requests.get(url)
    response.raise_for_status() # HTTP 에러 발생 시 예외 처리
    return response.json()

def main():
    # Firebase Hosting이 인식할 public/regions 폴더 생성
    output_dir = os.path.join("public", "regions")
    os.makedirs(output_dir, exist_ok=True)
    
    for region_name, coords in REGIONS.items():
        print(f"[{region_name}] 날씨 데이터 가져오는 중...")
        
        try:
            weather_data = fetch_weather_for_region(coords["lat"], coords["lon"])
            current_weather = weather_data.get("current", {})
            
            # 클라이언트와 EBM이 사용하기 편하도록 데이터 정제
            region_output = {
                "region": region_name,
                "updated_at": datetime.now().isoformat(),
                "weather": {
                    "temperature": current_weather.get("temperature_2m"),
                    "humidity": current_weather.get("relative_humidity_2m"),
                    "precipitation": current_weather.get("precipitation"),
                    "wind_speed": current_weather.get("wind_speed_10m")
                },
                # 추후 EBM 인퍼런스 결과와 LLM 분석 결과를 병합할 자리
                "ebm_risk_level": "Pending", 
                "llm_insight": "분석 대기 중"
            }
            
            # 지역별 개별 JSON 파일로 저장 (예: public/regions/seoul.json)
            file_path = os.path.join(output_dir, f"{region_name}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(region_output, f, ensure_ascii=False, indent=4)
                
            print(f"저장 완료: {file_path}")
            
        except Exception as e:
            print(f"[{region_name}] 데이터 처리 중 에러 발생: {e}")

if __name__ == "__main__":
    main()