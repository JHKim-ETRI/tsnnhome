"""
자극의 기본 클래스 정의
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class Stimulus(ABC):
    """자극의 기본 클래스"""
    
    def __init__(self, 
                 duration: float,
                 sampling_rate: float = 1000.0):
        """
        Parameters
        ----------
        duration : float
            자극 지속 시간 (ms)
        sampling_rate : float
            샘플링 레이트 (Hz), 기본값 1kHz
        """
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.dt = 1000.0 / sampling_rate  # ms 단위의 시간 간격
        self.time_points = np.arange(0, duration, self.dt)
        
    @abstractmethod
    def get_stimulus_value(self, t: float, location: Tuple[float, float]) -> float:
        """
        주어진 시간과 위치에서의 자극 값을 반환

        Parameters
        ----------
        t : float
            시간 (ms)
        location : Tuple[float, float]
            위치 좌표 (x, y) (mm)

        Returns
        -------
        float
            자극의 세기
        """
        pass
    
    def get_stimulus_array(self, 
                          location: Tuple[float, float],
                          t_start: Optional[float] = None,
                          t_end: Optional[float] = None) -> np.ndarray:
        """
        주어진 위치에서 시간 구간 동안의 자극 배열을 반환

        Parameters
        ----------
        location : Tuple[float, float]
            위치 좌표 (x, y) (mm)
        t_start : float, optional
            시작 시간 (ms)
        t_end : float, optional
            종료 시간 (ms)

        Returns
        -------
        np.ndarray
            자극 세기의 시계열 배열
        """
        if t_start is None:
            t_start = 0
        if t_end is None:
            t_end = self.duration
            
        t_indices = (self.time_points >= t_start) & (self.time_points < t_end)
        times = self.time_points[t_indices]
        
        return np.array([self.get_stimulus_value(t, location) for t in times]) 