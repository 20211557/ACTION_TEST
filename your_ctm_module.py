# your_ctm_module.py
import numpy as np

def compute_e(temp_mean_3: float, rh_mean_3: float) -> float:
    """
    회의록 10p 사양에 따른 CTM 환경 점수 E 계산 함수
    """
    # 1. 예외 처리 (NaN 또는 수치 변환 실패 시 0 처리)
    try:
        temp_mean_3 = float(temp_mean_3)
        rh_mean_3 = float(rh_mean_3)
    except (ValueError, TypeError):
        return 0.0
        
    if np.isnan(temp_mean_3) or np.isnan(rh_mean_3):
        return 0.0

    # 2. r(T): Yan-Hunt CTM 공식 적용 (Tmin=22, Topt=31, Tmax=35)
    t_min, t_opt, t_max = 22.0, 31.0, 35.0
    
    if t_min < temp_mean_3 < t_max:
        # 분수 지수 계산의 안전성을 위해 공식 그대로 구현
        alpha = (t_max - temp_mean_3) / (t_max - t_opt)
        beta = (temp_mean_3 - t_min) / (t_opt - t_min)
        exponent = (t_opt - t_min) / (t_max - t_opt)
        
        r_t = alpha * (beta ** exponent)
    else:
        r_t = 0.0

    # 3. f(RH): Sigmoid 공식 적용 (RH50=96)
    # exp 오버플로우 방지를 위해 수치적으로 안전하게 clip 후 계산
    minus_g = -0.5 * (rh_mean_3 - 96.0)
    minus_g = np.clip(minus_g, -50, 50) 
    f_rh = 1.0 / (1.0 + np.exp(minus_g))

    # 4. 최종 E 산출 및 [0, 1] 범위 보장
    e = r_t * f_rh
    return float(np.clip(e, 0.0, 1.0))

# ───────────────────────────────────────────────────────────────────────
# 단위 테스트 수행 (인계문서 12p 권장 사양)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 [단위 테스트 실행 중...]")
    
    # 테스트 케이스 1: 고위험 (Topt 근처 + 고습도) -> 1에 가까워야 함
    high_risk = compute_e(30.0, 95.0)
    print(f" - 고위험 케이스 (30°C, 95%): {high_risk:.4f} (1에 가깝고 0.25 임계값 이상인지 확인)")
    
    # 테스트 케이스 2: 저위험 (Tmin 미만 + 저습도) -> 0이어야 함
    low_risk = compute_e(15.0, 50.0)
    print(f" - 저위험 케이스 (15°C, 50%): {low_risk:.4f} (0.0000 확인)")
    
    # 테스트 케이스 3: 경계 조건 (Topt 정확히 일치)
    opt_risk = compute_e(31.0, 100.0)
    print(f" - 최적온도 케이스 (31°C, 100%): {opt_risk:.4f} (r(T)=1 이므로 습도값에 수렴)")