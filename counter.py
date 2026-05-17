import os
import json
import requests

# ⚠️ 본인의 Firebase 프로젝트 ID로 변경하세요.
PROJECT_ID = "test-b30a4" 
URL = f"https://{PROJECT_ID}.web.app/data.json"

def main():
    current_count = 0
    
    # 1. 기존에 Firebase Hosting에 배포된 JSON 파일이 있다면 읽어옵니다.
    try:
        response = requests.get(URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            current_count = data.get("count", 0)
            print(f"기존 카운트 확인 성공: {current_count}")
        else:
            print("기존 파일이 없거나 새로운 배포입니다. 0부터 시작합니다.")
    except Exception as e:
        print(f"기존 데이터를 읽어오지 못했습니다(첫 배포 시 발생 가능): {e}")
    
    # 2. 카운트 1 증가
    new_count = current_count + 1
    print(f"새로운 카운트 계산: {new_count}")
    
    # 3. Firebase Hosting이 배포할 public 폴더에 JSON 저장
    output_data = {
        "count": new_count
    }
    
    os.makedirs("public", exist_ok=True)
    with open("public/data.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("public/data.json 파일 생성 완료.")

if __name__ == "__main__":
    main()