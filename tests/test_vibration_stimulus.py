"""
진동 자극 클래스 테스트
"""

import pytest
import numpy as np
from src.stimuli.vibration import VibrationStimulus

def test_vibration_stimulus_initialization():
    """기본 초기화 테스트"""
    # 단일 주파수
    stim = VibrationStimulus(
        amplitude=1.0,
        frequency=50.0,  # 50Hz
        location=(0.0, 0.0),
        duration=100.0
    )
    
    assert stim.amplitude == 1.0
    assert len(stim.frequency) == 1
    assert stim.frequency[0] == 50.0
    assert stim.center_location == (0.0, 0.0)
    assert stim.duration == 100.0
    
    # 다중 주파수
    stim_multi = VibrationStimulus(
        amplitude=1.0,
        frequency=[30.0, 60.0],  # 30Hz + 60Hz
        location=(0.0, 0.0),
        duration=100.0,
        phase=[0.0, np.pi/2]  # 위상차 있는 진동
    )
    
    assert len(stim_multi.frequency) == 2
    assert len(stim_multi.phase) == 2
    assert stim_multi.phase[1] == np.pi/2

def test_vibration_stimulus_timing():
    """시간에 따른 자극 값 테스트"""
    stim = VibrationStimulus(
        amplitude=1.0,
        frequency=50.0,
        location=(0.0, 0.0),
        duration=100.0,
        onset_time=20.0,
        rise_time=10.0,
        fall_time=10.0
    )
    
    # 시작 전
    assert stim.get_stimulus_value(0.0, (0.0, 0.0)) == 0.0
    
    # 상승 구간
    values_rise = [stim.get_stimulus_value(t, (0.0, 0.0)) for t in np.linspace(20.0, 30.0, 10)]
    assert all(-1.0 <= v <= 1.0 for v in values_rise)  # 진폭 범위 확인
    
    # 최대 진폭 구간
    values_max = [stim.get_stimulus_value(t, (0.0, 0.0)) for t in np.linspace(40.0, 50.0, 10)]
    assert any(abs(v) > 0.9 for v in values_max)  # 최대 진폭에 도달
    
    # 종료 후
    assert stim.get_stimulus_value(100.0, (0.0, 0.0)) == 0.0

def test_vibration_stimulus_spatial():
    """공간적 감쇠 테스트"""
    stim = VibrationStimulus(
        amplitude=1.0,
        frequency=50.0,
        location=(0.0, 0.0),
        duration=100.0,
        radius=1.0
    )
    
    # 시간 고정 (진동의 최대값에서)
    t = 50.0  # 중간 시점
    
    # 중심점에서의 최대 진폭
    center_values = [stim.get_stimulus_value(t + i*1.0, (0.0, 0.0)) for i in range(20)]
    max_amplitude = max(abs(v) for v in center_values)
    assert pytest.approx(max_amplitude, abs=0.1) == 1.0
    
    # 반경 위치에서의 감쇠
    radius_values = [stim.get_stimulus_value(t + i*1.0, (1.0, 0.0)) for i in range(20)]
    max_radius_amplitude = max(abs(v) for v in radius_values)
    assert pytest.approx(max_radius_amplitude, abs=0.1) == np.exp(-0.5)
    
    # 먼 위치에서의 감쇠
    far_values = [stim.get_stimulus_value(t + i*1.0, (3.0, 0.0)) for i in range(20)]
    max_far_amplitude = max(abs(v) for v in far_values)
    assert max_far_amplitude < 0.1

def test_vibration_stimulus_frequency():
    """주파수 특성 테스트"""
    # 50Hz 진동
    stim_50hz = VibrationStimulus(
        amplitude=1.0,
        frequency=50.0,
        location=(0.0, 0.0),
        duration=1000.0  # 1초
    )
    
    # 1초 동안의 데이터 샘플링
    t = np.linspace(0, 1000, 1000)  # 1kHz 샘플링
    values = [stim_50hz.get_stimulus_value(t_i, (0.0, 0.0)) for t_i in t]
    
    # FFT로 주파수 분석
    fft = np.fft.fft(values)
    freqs = np.fft.fftfreq(len(t), 1/1000)  # 주파수 축
    
    # 주파수 피크 찾기
    peak_freq = freqs[np.argmax(np.abs(fft[1:len(fft)//2])) + 1]
    assert pytest.approx(abs(peak_freq), abs=1.0) == 50.0  # 50Hz 근처에 피크가 있어야 함 