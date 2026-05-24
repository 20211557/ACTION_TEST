import os
import json
import requests
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------------------------------------------------
# 1. Firebase Admin SDK 초기화 설정
# ---------------------------------------------------------
# 로컬 테스트 시에는 다운로드 받은 서비스 계정 키 파일의 경로를 직접 입력합니다.
# 깃허브 액션 환경에서는 Secrets에 저장된 JSON을 파일로 임시 생성하여 읽게 됩니다.
FIREBASE_KEY_PATH = "firebase_service_account.json" 

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ---------------------------------------------------------
# 2. 지역 코드(region_id)와 실제 위경도(lat, lon) 매핑 테이블
# ---------------------------------------------------------
# 실제 캡스톤 데이터셋 환경이라면 하드코딩 대신 공공데이터 CSV나 
# 로컬 매핑 파일을 읽어오는 로직으로 대체하는 것이 좋습니다.
REGION_COORDS = {
    "seoul_mapo": {"name": "서울 마포구", "lat": 37.5663, "lon": 126.9016},
    "seoul_dongjak": {"name": "서울 동작구", "lat": 37.5124, "lon": 126.9393},
    "busan_haeundae": {"name": "부산 해운대구", "lat": 35.1631, "lon": 129.1636},
    "jeju_seogwipo": {"name": "제주 서귀포시", "lat": 33.2541, "lon": 126.5601}
}

def get_active_regions_from_db():
    """Firestore에서 유저들이 등록한 활성 지역 목록을 중복 없이 가져옵니다."""
    print("🔍 Firestore에서 유저 등록 지역 목록을 조회합니다...")
    docs = db.collection('user_profiles').stream()
    
    active_regions = set()
    for doc in docs:
        user_data = doc.to_dict()
        region_id = user_data.get('region_id')
        
        # DB에 있는 지역 ID가 우리가 좌표를 알고 있는 지역일 경우에만 추가
        if region_id and region_id in REGION_COORDS:
            active_regions.add(region_id)
            
    return list(active_regions)

def fetch_weather_for_region(lat, lon):
    """Open-Meteo API를 호출하여 해당 위경도의 기상 데이터를 가져옵니다."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&"
        f"timezone=Asia%2FTokyo"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status() # HTTP 200이 아니면 예외 발생
    return response.json()

def main():
    # 1. 수요 기반 타겟 지역 추출
    target_regions = get_active_regions_from_db()
    print(f"🎯 오늘 연산할 타겟 지역 수: {len(target_regions)}개")
    print(f"📋 타겟 지역 리스트: {target_regions}")
    
    if not target_regions:
        print("사용자가 등록한 지역이 없습니다. 작업을 종료합니다.")
        return

    # 2. 결과물을 저장할 폴더 생성 (Firebase Hosting 배포용)
    output_dir = os.path.join("public", "regions")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 타겟 지역별 API 호출 및 데이터 병합
    for region_id in target_regions:
        info = REGION_COORDS[region_id]
        print(f"\n🌤️ [{info['name']}] 날씨 데이터 분석 시작...")
        
        try:
            # 3-1. 기상 데이터 호출
            weather_data = fetch_weather_for_region(info["lat"], info["lon"])
            current_weather = weather_data.get("current", {})
            
            # ---------------------------------------------------------
            # [TODO] 여기에 ML 모델 인퍼런스 로직이 들어갈 자리입니다.
            # ebm_result = predict_plant_disease(current_weather)
            # llm_text = generate_insight_with_llm(ebm_result, info['name'])
            # ---------------------------------------------------------
            
            # 3-2. 최종 JSON 구조체 생성
            region_output = {
                "region_id": region_id,
                "region_name": info["name"],
                "updated_at": datetime.now().isoformat(),
                "weather": {
                    "temperature": current_weather.get("temperature_2m"),
                    "humidity": current_weather.get("relative_humidity_2m"),
                    "precipitation": current_weather.get("precipitation"),
                    "wind_speed": current_weather.get("wind_speed_10m")
                },
                "ebm_risk_level": "Pending",  # EBM 예측 결과 맵핑
                "llm_insight": "분석 대기 중" # LLM 분석 결과 맵핑
            }
            
            # 3-3. 로컬 파일로 저장 (예: public/regions/seoul_mapo.json)
            file_path = os.path.join(output_dir, f"{region_id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(region_output, f, ensure_ascii=False, indent=4)
                
            print(f"✅ JSON 저장 완료: {file_path}")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ [{info['name']}] 날씨 API 통신 에러: {e}")
        except Exception as e:
            print(f"❌ [{info['name']}] 데이터 처리 중 알 수 없는 에러: {e}")

    print("\n🚀 모든 배치 작업이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    main()