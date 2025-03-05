"""
압력 자극 클래스 테스트
"""

import pytest
import numpy as np
from src.stimuli.pressure import PressureStimulus

def test_pressure_stimulus_initialization():
    """기본 초기화 테스트"""
    stim = PressureStimulus(
        amplitude=1.0,
        location=(0.0, 0.0),
        duration=100.0
    )
    
    assert stim.amplitude == 1.0
    assert stim.center_location == (0.0, 0.0)
    assert stim.duration == 100.0
    assert stim.radius == 1.0  # 기본값
    assert stim.sampling_rate == 1000.0  # 기본값

def test_pressure_stimulus_timing():
    """시간에 따른 자극 값 테스트"""
    stim = PressureStimulus(
        amplitude=1.0,
        location=(0.0, 0.0),
        duration=100.0,
        onset_time=20.0,
        rise_time=10.0,
        fall_time=10.0
    )
    
    # 시작 전
    assert stim.get_stimulus_value(0.0, (0.0, 0.0)) == 0.0
    
    # 상승 구간
    assert 0.0 < stim.get_stimulus_value(25.0, (0.0, 0.0)) < 1.0
    
    # 최대 진폭 구간
    assert stim.get_stimulus_value(50.0, (0.0, 0.0)) == pytest.approx(1.0)
    
    # 하강 구간
    assert 0.0 < stim.get_stimulus_value(95.0, (0.0, 0.0)) < 1.0
    
    # 종료 후
    assert stim.get_stimulus_value(100.0, (0.0, 0.0)) == 0.0

def test_pressure_stimulus_spatial():
    """공간적 감쇠 테스트"""
    stim = PressureStimulus(
        amplitude=1.0,
        location=(0.0, 0.0),
        duration=100.0,
        radius=1.0
    )
    
    # 중심점
    center_value = stim.get_stimulus_value(50.0, (0.0, 0.0))
    assert pytest.approx(center_value) == 1.0
    
    # 반경 위치 (감쇠 ~37%)
    radius_value = stim.get_stimulus_value(50.0, (1.0, 0.0))
    assert pytest.approx(radius_value, abs=0.1) == np.exp(-0.5)
    
    # 먼 위치 (거의 0)
    far_value = stim.get_stimulus_value(50.0, (3.0, 0.0))
    assert far_value < 0.1

def test_pressure_stimulus_array():
    """시계열 배열 생성 테스트"""
    stim = PressureStimulus(
        amplitude=1.0,
        location=(0.0, 0.0),
        duration=100.0,
        sampling_rate=1000.0
    )
    
    # 전체 시계열
    full_array = stim.get_stimulus_array((0.0, 0.0))
    assert len(full_array) == 100  # 100ms * 1kHz = 100 samples
    
    # 부분 시계열
    partial_array = stim.get_stimulus_array((0.0, 0.0), t_start=25.0, t_end=75.0)
    assert len(partial_array) == 50  # 50ms * 1kHz = 50 samples 