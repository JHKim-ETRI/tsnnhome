# 촉각 스파이킹 신경망 (Tactile Spiking Neural Network)

이 프로젝트는 생물학적으로 영감을 받은 촉각 정보 처리를 위한 스파이킹 신경망을 구현합니다.

## 주요 기능

- Izhikevich 뉴런 모델 기반 SNN 구현
- 다양한 기계수용기(SA1, RA1, RA2) 모델링
- 쿠네이트핵과 체성감각피질의 계층적 구조 구현
- 압력과 진동 자극에 대한 처리
- STDP(Spike-Timing-Dependent Plasticity) 학습 구현

## 설치 방법

```bash
git clone https://github.com/yourusername/TactileSpikingNeuralNetwork.git
cd TactileSpikingNeuralNetwork
pip install -r requirements.txt
```

## 사용 예시

```python
from src.models.network import TactileSNN
from src.stimuli.pressure import PressureStimulus

# 네트워크 생성
network = TactileSNN(
    input_size=(10, 10),
    n_mechanoreceptors=100,
    n_cuneate=50,
    n_cortical=200
)

# 자극 생성
stimulus = PressureStimulus(
    amplitude=1.0,
    location=(5, 5),
    duration=1000
)

# 시뮬레이션 실행
results = network.simulate(stimulus)

# 결과 시각화
network.visualize_results(results)
```

## 구조

```
src/
├── models/      # 핵심 모델 구현
├── layers/      # 신경망 층 구현
├── utils/       # 유틸리티 함수
└── stimuli/     # 자극 생성 모듈
```

## 참고 문헌

1. Izhikevich, E. M. (2003). Simple model of spiking neurons.
2. Johansson, R. S., & Flanagan, J. R. (2009). Coding and use of tactile signals from the fingertips in object manipulation tasks.
3. Saal, H. P., & Bensmaia, S. J. (2014). Touch is a team effort: interplay of submodalities in cutaneous sensibility.

## 라이선스

MIT License 