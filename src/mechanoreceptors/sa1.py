"""
SA1 (Merkel cell) 수용체 클래스 구현
"""

from typing import Tuple, Dict
import numpy as np
from .base import BaseMechanoreceptor

class SA1Mechanoreceptor(BaseMechanoreceptor):
    """SA1 (Merkel cell) 수용체 클래스
    
    특성:
    - 압력 인코딩
    - 느린 적응
    - 주파수 응답: 0-150Hz
    - 문턱값: 7-10mN
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
            # SA1의 기본 매개변수 (Regular Spiking)
            parameters = {
                'a': 0.02,      # 느린 회복
                'b': 0.2,       # 약한 공명
                'c': -65.0,     # 스파이크 후 리셋 전압
                'd': 6.0,       # 스파이크 후 회복 변수 증가
                'v_thresh': 30.0,
                'v_reset': -65.0
            }
            
        super().__init__(location, parameters)
        
        # SA1 특성 매개변수
        self.threshold = 8.0  # 문턱값 (mN/mm²)
        self.k = 20.0  # 전류 변환 계수 (pA)
        
    def stimulus_to_current(self, stimulus_value: float) -> float:
        """
        압력 자극을 입력 전류로 변환 (로그 응답 특성)
        SA1은 정적 압력에 민감하며 천천히 적응

        매개변수
        ----------
        stimulus_value : float
            자극의 세기 (mN/mm²)

        반환값
        -------
        float
            변환된 입력 전류 (pA)
        """
        # 문턱값 이하면 0
        if stimulus_value < self.threshold:
            return 0.0
            
        # 로그 응답 특성 (I = k·log(P/P₀))
        return self.k * np.log(stimulus_value / self.threshold + 1.0) 