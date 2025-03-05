"""
Example code demonstrating the response characteristics of different mechanoreceptors
"""

import numpy as np
import matplotlib.pyplot as plt
from src.stimuli.pressure import PressureStimulus
from src.stimuli.vibration import VibrationStimulus
from src.mechanoreceptors import SA1Mechanoreceptor, RA1Mechanoreceptor, RA2Mechanoreceptor

def simulate_response(receptor, stimulus, duration=1000.0, dt=1.0):
    """시뮬레이션을 실행하고 자극, 입력 전류, 스파이크 시간, 멤브레인 전위를 반환"""
    time_points = np.arange(0, duration, dt)
    stimulus_values = []
    current_values = []
    membrane_potentials = []
    receptor.reset()
    
    # 시뮬레이션 실행
    for t in time_points:
        stim_value = stimulus.get_stimulus_value(t, (0.0, 0.0))
        stimulus_values.append(stim_value)
        current = receptor.stimulus_to_current(stim_value)
        current_values.append(current)
        membrane_potentials.append(receptor.v)  # 현재 멤브레인 전위 저장
        receptor.update(stimulus, dt, t)
    
    return time_points, stimulus_values, current_values, receptor.spike_times, membrane_potentials

def plot_responses(receptors, stimulus, title, duration=1000.0, dt=1.0):
    """모든 수용체의 반응을 하나의 figure에 플롯"""
    fig, axes = plt.subplots(len(receptors), 4, figsize=(20, 12))
    fig.suptitle(title, fontsize=14)
    
    for i, (name, receptor) in enumerate(receptors.items()):
        times, stim, curr, spikes, v_m = simulate_response(receptor, stimulus, duration, dt)
        
        # 자극
        axes[i, 0].plot(times, stim, 'b-')
        axes[i, 0].set_ylabel(f'{name}\nStimulus\n(mN/mm²)')
        axes[i, 0].set_xlim(0, duration)
        axes[i, 0].grid(True)
        
        # 입력 전류
        axes[i, 1].plot(times, curr, 'g-')
        axes[i, 1].set_ylabel('Current (pA)')
        axes[i, 1].set_xlim(0, duration)
        axes[i, 1].grid(True)
        
        # 멤브레인 전위
        axes[i, 2].plot(times, v_m, 'r-')
        axes[i, 2].set_ylabel('Membrane\nPotential (mV)')
        axes[i, 2].set_xlim(0, duration)
        axes[i, 2].grid(True)
        
        # 스파이크
        if len(spikes) > 0:
            axes[i, 3].eventplot(spikes, lineoffsets=0.5, linelengths=0.5, color='k')
        axes[i, 3].set_ylabel('Spikes')
        axes[i, 3].set_xlim(0, duration)
        axes[i, 3].set_ylim(0, 1)
        axes[i, 3].grid(True)
        
        # x축 레이블은 마지막 행에만
        if i == len(receptors) - 1:
            axes[i, 0].set_xlabel('Time (ms)')
            axes[i, 1].set_xlabel('Time (ms)')
            axes[i, 2].set_xlabel('Time (ms)')
            axes[i, 3].set_xlabel('Time (ms)')
    
    plt.tight_layout()
    return fig

def calculate_firing_rate(spike_times, start_time=200, end_time=800):
    """주어진 시간 구간에서의 발화율 계산"""
    spikes_in_window = [t for t in spike_times if start_time <= t <= end_time]
    return len(spikes_in_window) * 1000 / (end_time - start_time)  # Hz

def main():
    # 수용체 생성
    receptors = {
        'SA1': SA1Mechanoreceptor((0.0, 0.0)),
        'RA1': RA1Mechanoreceptor((0.0, 0.0)),
        'RA2': RA2Mechanoreceptor((0.0, 0.0))
    }
    
    # 시뮬레이션 파라미터
    duration = 1000.0  # ms
    dt = 1.0  # ms
    
    # 1. 다양한 압력 자극에 대한 반응
    pressures = [5, 10, 20, 50, 100, 200]  # mN/mm²
    pressure_rates = {name: [] for name in receptors.keys()}
    
    for pressure in pressures:
        pressure_stim = PressureStimulus(
            amplitude=pressure,
            location=(0.0, 0.0),
            duration=duration,
            onset_time=200.0,
            offset_time=800.0,
            rise_time=50.0,
            fall_time=50.0
        )
        plot_responses(receptors, pressure_stim, f'Responses to {pressure} mN/mm² Pressure', duration, dt)
        
        # 발화율 계산
        for name, receptor in receptors.items():
            _, _, _, spikes, _ = simulate_response(receptor, pressure_stim)
            pressure_rates[name].append(calculate_firing_rate(spikes))
    
    # 2. 다양한 주파수의 진동 자극에 대한 반응
    frequencies = [30, 100, 200, 1000]  # Hz
    frequency_rates = {name: [] for name in receptors.keys()}
    
    for freq in frequencies:
        vibration_stim = VibrationStimulus(
            amplitude=1.0,
            frequency=freq,
            location=(0.0, 0.0),
            duration=duration,
            onset_time=200.0,
            offset_time=800.0,
            rise_time=50.0,
            fall_time=50.0
        )
        plot_responses(receptors, vibration_stim, f'Responses to {freq}Hz Vibration', duration, dt)
        
        # 발화율 계산
        for name, receptor in receptors.items():
            _, _, _, spikes, _ = simulate_response(receptor, vibration_stim)
            frequency_rates[name].append(calculate_firing_rate(spikes))
    
    # 3. 발화율 요약 그래프
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Summary of Firing Rates', fontsize=14)
    
    # 압력-발화율 그래프
    for name in receptors.keys():
        ax1.plot(pressures, pressure_rates[name], 'o-', label=name)
    ax1.set_xscale('log')
    ax1.set_xlabel('Pressure (mN/mm²)')
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_title('Response vs Pressure')
    ax1.grid(True)
    ax1.legend()
    
    # 주파수-발화율 그래프
    for name in receptors.keys():
        ax2.plot(frequencies, frequency_rates[name], 'o-', label=name)
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Firing Rate (Hz)')
    ax2.set_title('Response vs Vibration Frequency')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 