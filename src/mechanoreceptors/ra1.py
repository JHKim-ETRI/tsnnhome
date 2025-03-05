"""
RA1 (Meissner corpuscle) 수용체 클래스 구현
"""

from typing import Tuple, Dict
import numpy as np
from .base import BaseMechanoreceptor

class RA1Mechanoreceptor(BaseMechanoreceptor):
    """RA1 (Meissner corpuscle) 수용체 클래스
    
    특성:
    - 진동 인코딩
    - 빠른 적응
    - 주파수 응답: 0-250Hz
    - 문턱값: 2-5mN
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
            # RA1의 기본 매개변수 (Fast Spiking)
            parameters = {
                'a': 0.02,      # 매우 빠른 회복
                'b': 0.25,      # 강한 공명
                'c': -60.0,     # 스파이크 후 리셋 전압
                'd': 4.0,       # 스파이크 후 회복 변수 증가
                'v_thresh': 30.0,
                'v_reset': -65.0
            }
            
        super().__init__(location, parameters)
        
        # RA1 특성 매개변수
        self.threshold = 2.0  # 문턱값 (mN/mm²/ms)
        self.k = 100.0  # 전류 변환 계수 (pA/(mN/mm²/ms))
        self.prev_stimulus = 0.0  # 이전 자극값 저장
        
    def stimulus_to_current(self, stimulus_value: float) -> float:
        """
        압력 자극을 입력 전류로 변환
        RA1은 자극의 변화율(속도)에 민감

        매개변수
        ----------
        stimulus_value : float
            자극의 세기 (mN/mm²)

        반환값
        -------
        float
            변환된 입력 전류 (pA)
        """
        # 자극의 변화율 계산
        dt = 1.0  # ms
        stimulus_derivative = (stimulus_value - self.prev_stimulus) / dt
        
        # 이전 자극값 업데이트
        self.prev_stimulus = stimulus_value
        
        # 문턱값 이하면 0
        if abs(stimulus_derivative) < self.threshold:
            return 0.0
            
        # 변화율을 전류로 변환 (I = k·dP/dt)
        return self.k * abs(stimulus_derivative) 