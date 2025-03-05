"""
압력 자극 클래스 구현
"""

import numpy as np
from typing import Tuple, Optional
from .base import Stimulus

class PressureStimulus(Stimulus):
    """압력 자극 클래스"""
    
    def __init__(self,
                 amplitude: float,
                 location: Tuple[float, float],
                 duration: float,
                 radius: float = 1.0,
                 onset_time: float = 0.0,
                 offset_time: Optional[float] = None,
                 rise_time: float = 5.0,
                 fall_time: float = 5.0,
                 sampling_rate: float = 1000.0):
        """
        Parameters
        ----------
        amplitude : float
            자극의 최대 압력 (mN/mm²)
        location : Tuple[float, float]
            자극 중심 위치 (x, y) (mm)
        duration : float
            자극 총 지속 시간 (ms)
        radius : float
            자극의 공간적 반경 (mm)
        onset_time : float
            자극 시작 시간 (ms)
        offset_time : float, optional
            자극 종료 시간 (ms), None이면 duration - fall_time
        rise_time : float
            자극 상승 시간 (ms)
        fall_time : float
            자극 하강 시간 (ms)
        sampling_rate : float
            샘플링 레이트 (Hz)
        """
        super().__init__(duration, sampling_rate)
        
        self.amplitude = amplitude
        self.center_location = location
        self.radius = radius
        self.onset_time = onset_time
        self.offset_time = offset_time if offset_time is not None else duration - fall_time
        self.rise_time = rise_time
        self.fall_time = fall_time
        
    def get_stimulus_value(self, t: float, location: Tuple[float, float]) -> float:
        """
        주어진 시간과 위치에서의 압력 값을 반환

        Parameters
        ----------
        t : float
            시간 (ms)
        location : Tuple[float, float]
            위치 좌표 (x, y) (mm)

        Returns
        -------
        float
            압력 값 (mN/mm²)
        """
        # 시간에 따른 진폭 계산
        if t < self.onset_time:
            time_factor = 0.0
        elif t < self.onset_time + self.rise_time:
            time_factor = (t - self.onset_time) / self.rise_time
        elif t < self.offset_time:
            time_factor = 1.0
        elif t < self.offset_time + self.fall_time:
            time_factor = 1.0 - (t - self.offset_time) / self.fall_time
        else:
            time_factor = 0.0
            
        # 공간적 감쇠 계산 (가우시안)
        dx = location[0] - self.center_location[0]
        dy = location[1] - self.center_location[1]
        distance = np.sqrt(dx**2 + dy**2)
        space_factor = np.exp(-(distance**2) / (2 * self.radius**2))
        
        return self.amplitude * time_factor * space_factor 