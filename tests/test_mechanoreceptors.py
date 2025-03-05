"""
기계수용체 클래스 테스트
"""

import pytest
import numpy as np
from src.stimuli.pressure import PressureStimulus
from src.stimuli.vibration import VibrationStimulus
from src.mechanoreceptors import SA1Mechanoreceptor, RA1Mechanoreceptor, RA2Mechanoreceptor

def test_mechanoreceptor_initialization():
    """기본 초기화 테스트"""
    location = (0.0, 0.0)
    
    # SA1 초기화
    sa1 = SA1Mechanoreceptor(location)
    assert sa1.location == location
    assert sa1.v == -65.0  # 기본 휴지 전위
    assert len(sa1.spike_times) == 0
    
    # RA1 초기화
    ra1 = RA1Mechanoreceptor(location)
    assert ra1.location == location
    assert ra1.prev_stimulus == 0.0
    
    # RA2 초기화
    ra2 = RA2Mechanoreceptor(location)
    assert ra2.location == location
    assert ra2.prev_stimulus == 0.0
    assert ra2.prev_derivative == 0.0

def test_sa1_response():
    """SA1 수용체의 정적 압력 반응 테스트"""
    sa1 = SA1Mechanoreceptor((0.0, 0.0))
    
    # 정적 압력 자극 생성
    stim = PressureStimulus(
        amplitude=1.0,
        location=(0.0, 0.0),
        duration=1000.0,
        onset_time=100.0,
        rise_time=50.0,
        fall_time=50.0
    )
    
    # 시뮬레이션
    dt = 1.0  # 1ms 간격
    for t in np.arange(0, 1000, dt):
        sa1.update(stim, dt, t)
    
    # 발화 시간 분석
    spikes = sa1.get_spike_times()
    assert len(spikes) > 0  # 발화가 있어야 함
    
    # 자극 구간에서 발화율이 높아야 함
    spikes_during_stim = sa1.get_spike_times(150.0, 900.0)  # 자극이 안정된 구간
    assert len(spikes_during_stim) > len(spikes) / 2  # 대부분의 발화가 자극 중에 발생

def test_ra1_response():
    """RA1 수용체의 중간 주파수 진동 반응 테스트"""
    ra1 = RA1Mechanoreceptor((0.0, 0.0))
    
    # 중간 주파수 진동 자극 생성 (30Hz)
    stim = VibrationStimulus(
        amplitude=1.0,
        frequency=30.0,  # RA1이 잘 반응하는 주파수
        location=(0.0, 0.0),
        duration=1000.0,
        onset_time=100.0,
        rise_time=20.0,
        fall_time=20.0
    )
    
    # 시뮬레이션
    dt = 0.1  # 더 작은 시간 간격 (진동 자극을 위해)
    for t in np.arange(0, 1000, dt):
        ra1.update(stim, dt, t)
    
    # 발화 시간 분석
    spikes = ra1.get_spike_times()
    assert len(spikes) > 0
    
    # 진동 구간에서의 발화율 계산
    spikes_during_stim = ra1.get_spike_times(120.0, 980.0)  # 안정된 진동 구간
    firing_rate = len(spikes_during_stim) / 0.86  # Hz (860ms 동안의 발화율)
    
    # 발화율이 자극 주파수와 비슷해야 함 (1:1 응답)
    assert 20 < firing_rate < 40  # 30Hz 근처의 발화율

def test_ra2_response():
    """RA2 수용체의 고주파수 진동 반응 테스트"""
    ra2 = RA2Mechanoreceptor((0.0, 0.0))
    
    # 고주파수 진동 자극 생성 (200Hz)
    stim = VibrationStimulus(
        amplitude=1.0,
        frequency=200.0,  # RA2가 잘 반응하는 주파수
        location=(0.0, 0.0),
        duration=1000.0,
        onset_time=100.0,
        rise_time=20.0,
        fall_time=20.0
    )
    
    # 시뮬레이션
    dt = 0.1
    for t in np.arange(0, 1000, dt):
        ra2.update(stim, dt, t)
    
    # 발화 시간 분석
    spikes = ra2.get_spike_times()
    assert len(spikes) > 0
    
    # 진동 구간에서의 발화 분석
    spikes_during_stim = ra2.get_spike_times(120.0, 980.0)
    firing_rate = len(spikes_during_stim) / 0.86  # Hz
    
    # 발화율이 자극 주파수보다 낮아야 함 (부분 응답)
    assert 50 < firing_rate < 200  # 자극 주파수의 1/4에서 1 사이
    
    # 저주파수 자극에 대한 반응 테스트
    low_freq_stim = VibrationStimulus(
        amplitude=1.0,
        frequency=30.0,  # 낮은 주파수
        location=(0.0, 0.0),
        duration=1000.0,
        onset_time=100.0,
        rise_time=20.0,
        fall_time=20.0
    )
    
    # 초기화 후 저주파 시뮬레이션
    ra2.reset()
    for t in np.arange(0, 1000, dt):
        ra2.update(low_freq_stim, dt, t)
    
    # 저주파 자극에서는 발화가 거의 없어야 함
    low_freq_spikes = ra2.get_spike_times(120.0, 980.0)
    assert len(low_freq_spikes) < len(spikes_during_stim) / 4

def test_reset():
    """상태 초기화 테스트"""
    # 각 수용체 타입 테스트
    receptors = [
        SA1Mechanoreceptor((0.0, 0.0)),
        RA1Mechanoreceptor((0.0, 0.0)),
        RA2Mechanoreceptor((0.0, 0.0))
    ]
    
    # 각 수용체에 적합한 자극 생성
    stims = [
        PressureStimulus(  # SA1용 압력 자극
            amplitude=2.0,
            location=(0.0, 0.0),
            duration=100.0,
            rise_time=5.0,
            fall_time=5.0
        ),
        VibrationStimulus(  # RA1용 중간 주파수 자극
            amplitude=1.0,
            frequency=30.0,
            location=(0.0, 0.0),
            duration=100.0,
            rise_time=5.0,
            fall_time=5.0
        ),
        VibrationStimulus(  # RA2용 고주파수 자극
            amplitude=1.0,
            frequency=200.0,
            location=(0.0, 0.0),
            duration=100.0,
            rise_time=5.0,
            fall_time=5.0
        )
    ]
    
    for receptor, stim in zip(receptors, stims):
        # 시뮬레이션 실행
        dt = 0.1
        for t in np.arange(0, 100, dt):
            receptor.update(stim, dt, t)
        
        # 발화가 있었는지 확인
        assert len(receptor.spike_times) > 0
        
        # 초기화
        receptor.reset()
        
        # 상태 변수 확인
        assert receptor.v == -65.0
        assert receptor.u == 0.0
        assert len(receptor.spike_times) == 0 