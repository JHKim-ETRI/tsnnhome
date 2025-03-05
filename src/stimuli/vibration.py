"""
진동 자극 클래스 구현
"""

import numpy as np
from typing import Tuple, Optional, Union, List
from .base import Stimulus

class VibrationStimulus(Stimulus):
    """진동 자극 클래스"""
    
    def __init__(self,
                 amplitude: float,
                 frequency: Union[float, List[float]],
                 location: Tuple[float, float],
                 duration: float,
                 radius: float = 1.0,
                 onset_time: float = 0.0,
                 offset_time: Optional[float] = None,
                 rise_time: float = 5.0,
                 fall_time: float = 5.0,
                 phase: Union[float, List[float]] = 0.0,
                 sampling_rate: float = 1000.0):
        """
        Parameters
        ----------
        amplitude : float
            진동의 최대 진폭 (mN/mm²)
        frequency : float or List[float]
            진동 주파수 (Hz). 여러 주파수를 합성할 경우 리스트로 제공
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
        phase : float or List[float]
            진동의 초기 위상 (라디안). frequency가 리스트인 경우 이것도 리스트여야 함
        sampling_rate : float
            샘플링 레이트 (Hz)
        """
        super().__init__(duration, sampling_rate)
        
        self.amplitude = amplitude
        self.frequency = np.array([frequency] if isinstance(frequency, (int, float)) else frequency)
        self.center_location = location
        self.radius = radius
        self.onset_time = onset_time
        self.offset_time = offset_time if offset_time is not None else duration - fall_time
        self.rise_time = rise_time
        self.fall_time = fall_time
        
        # 위상 처리
        if isinstance(phase, (int, float)):
            self.phase = np.array([phase] * len(self.frequency))
        else:
            assert len(phase) == len(self.frequency), "주파수와 위상의 길이가 일치해야 합니다"
            self.phase = np.array(phase)
    
    def get_stimulus_value(self, t: float, location: Tuple[float, float]) -> float:
        """
        주어진 시간과 위치에서의 진동 값을 반환

        Parameters
        ----------
        t : float
            시간 (ms)
        location : Tuple[float, float]
            위치 좌표 (x, y) (mm)

        Returns
        -------
        float
            진동 값 (mN/mm²)
        """
        # 시간에 따른 진폭 변조 계산
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
        
        # 진동 합성
        t_sec = t / 1000.0  # ms를 초 단위로 변환
        vibration = np.sum([
            np.sin(2 * np.pi * f * t_sec + p)
            for f, p in zip(self.frequency, self.phase)
        ]) / len(self.frequency)  # 진폭 정규화
        
        return self.amplitude * time_factor * space_factor * vibration 