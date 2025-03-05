"""
기계수용체의 기본 클래스 정의
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, List, Optional
from ..stimuli.base import Stimulus

class BaseMechanoreceptor(ABC):
    """기계수용체의 기본 클래스"""
    
    def __init__(self,
                 location: Tuple[float, float],
                 parameters: Dict[str, float]):
        """
        매개변수
        ----------
        location : Tuple[float, float]
            수용체의 위치 (x, y) (mm)
        parameters : Dict[str, float]
            Izhikevich 뉴런 모델 매개변수
            - a: 회복 변수의 시간 스케일 (1/ms)
            - b: 회복 변수의 민감도
            - c: 스파이크 후 전압 리셋 값 (mV)
            - d: 스파이크 후 회복 변수 리셋
            - v_thresh: 발화 임계값 (mV)
            - v_reset: 휴지 전위 (mV)
        """
        self.location = location
        self.parameters = parameters
        
        # Izhikevich 뉴런 모델 상태 변수
        self.v = self.parameters.get('v_reset', -65.0)  # 막전위 (mV)
        self.u = self.parameters.get('b', 0.2) * self.v  # 회복 변수
        self.spike_times: List[float] = []  # 발화 시간 기록
        
    def reset(self):
        """상태 변수 초기화"""
        self.v = self.parameters.get('v_reset', -65.0)
        self.u = self.parameters.get('b', 0.2) * self.v
        self.spike_times = []
        
    @abstractmethod
    def stimulus_to_current(self, stimulus_value: float) -> float:
        """
        자극 값을 입력 전류로 변환

        매개변수
        ----------
        stimulus_value : float
            자극의 세기 (mN/mm²)

        반환값
        -------
        float
            변환된 입력 전류 (pA)
        """
        pass
    
    def update(self, stimulus: Stimulus, dt: float, t: float):
        """
        주어진 시간 단계에서 뉴런의 상태를 업데이트

        매개변수
        ----------
        stimulus : Stimulus
            입력 자극
        dt : float
            시간 간격 (ms)
        t : float
            현재 시간 (ms)
        """
        # 자극 값을 전류로 변환
        I = self.stimulus_to_current(
            stimulus.get_stimulus_value(t, self.location)
        )
        
        # Izhikevich 모델 매개변수
        a = self.parameters.get('a', 0.02)
        b = self.parameters.get('b', 0.2)
        c = self.parameters.get('c', -65.0)
        d = self.parameters.get('d', 6.0)
        v_thresh = self.parameters.get('v_thresh', 30.0)
        
        # 상태 변수 업데이트 (0.04는 스케일링 상수)
        dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
        du = (a * (b * self.v - self.u)) * dt
        
        self.v += dv
        self.u += du
        
        # 발화 체크 및 리셋
        if self.v >= v_thresh:
            self.spike_times.append(t)
            self.v = c
            self.u += d
    
    def get_spike_times(self, t_start: Optional[float] = None, t_end: Optional[float] = None) -> List[float]:
        """
        주어진 시간 구간의 발화 시간들을 반환

        매개변수
        ----------
        t_start : float, optional
            시작 시간 (ms)
        t_end : float, optional
            종료 시간 (ms)

        반환값
        -------
        List[float]
            발화 시간들의 리스트
        """
        if t_start is None and t_end is None:
            return self.spike_times
            
        return [t for t in self.spike_times if 
                (t_start is None or t >= t_start) and 
                (t_end is None or t < t_end)] 