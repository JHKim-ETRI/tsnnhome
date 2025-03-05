"""
RA2 (Pacinian corpuscle) 수용체 클래스 구현
"""

from typing import Tuple, Dict
import numpy as np
from .base import BaseMechanoreceptor

class RA2Mechanoreceptor(BaseMechanoreceptor):
    """RA2 (Pacinian corpuscle) 수용체 클래스
    
    특성:
    - 고주파 진동 인코딩
    - 매우 빠른 적응
    - 주파수 응답: 0-300Hz
    - 문턱값: 0.5-1mN
    """
    
    def __init__(self,
                 location: Tuple[float, float],
                 parameters: Dict[str, float] = None):
        """
        매개변수
        ----------
        location : Tuple[float, float]
            수용체의 위치 (x, y) (mm)
        parameters : Dict[str, float], optional
            Izhikevich 뉴런 모델 매개변수
        """
        if parameters is None:
            # RA2의 기본 매개변수 (Fast Spiking)
            parameters = {
                'a': 0.02,      # 매우 빠른 회복
                'b': 0.25,      # 강한 공명
                'c': -60.0,     # 스파이크 후 리셋 전압
                'd': 4.0,       # 스파이크 후 회복 변수 증가
                'v_thresh': 30.0,
                'v_reset': -65.0
            }
            
        super().__init__(location, parameters)
        
        # RA2 특성 매개변수
        self.threshold = 0.75  # 문턱값 (mN/mm²)
        self.k = 40.0  # 전류 변환 계수 (pA)
        self.prev_derivative = 0.0  # 이전 변화율 저장
        self.prev_stimulus = 0.0    # 이전 자극값 저장
        
    def stimulus_to_current(self, stimulus_value: float) -> float:
        """
        압력 자극을 입력 전류로 변환
        RA2는 자극의 가속도(변화율의 변화율)에 민감

        매개변수
        ----------
        stimulus_value : float
            자극의 세기 (mN/mm²)

        반환값
        -------
        float
            변환된 입력 전류 (pA)
        """
        # 자극의 1차 변화율 계산
        dt = 1.0  # ms
        stimulus_derivative = (stimulus_value - self.prev_stimulus) / dt
        
        # 자극의 2차 변화율(가속도) 계산
        stimulus_acceleration = (stimulus_derivative - self.prev_derivative) / dt
        
        # 이전 값들 업데이트
        self.prev_stimulus = stimulus_value
        self.prev_derivative = stimulus_derivative
        
        # 문턱값 이하면 0
        if abs(stimulus_acceleration) < self.threshold:
            return 0.0
            
        # 가속도를 전류로 변환 (I = k·d²P/dt²)
        return self.k * abs(stimulus_acceleration) 