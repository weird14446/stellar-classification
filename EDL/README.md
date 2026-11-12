# Evolutionary Deep Learning (EDL): Topology & Weight Evolving Neural Network (TWEANN)

SDSS-17 천체 데이터셋에 대해 **신경망의 가중치(Weights)뿐만 아니라 신경망 구조(Topology: 은닉층 깊이, 층별 뉴런 수, 활성화 함수, 입력 스킵 연결, 시냅스 희소성) 자체를 함께 진화시키는 TWEANN(Topology & Weight Evolving Artificial Neural Network)** 파이프라인입니다.

---

## 1. 진화 신경망(EDL / TWEANN)의 핵심 아키텍처

1. **구조 유전체 (`TopologyGenome`) + 가중치 유전체 동시 진화**:
   - 사람이 고정된 `(64, 32)` 구조를 지정하는 대신, 8개의 독립된 섬(Island)이 서로 다른 초기 씨앗 구조에서 출발하여 **은닉층 깊이(Depth), 층별 뉴런 폭(Width), 활성화 함수(`relu`, `gelu`, `silu`, `elu`, `tanh`), 입력 직결 스킵 연결(`input_skips`), 시냅스 가지치기(Synapse Pruning)**를 스스로 진화시킵니다.
2. **함수 보존형 Net2Net 구조 변이 연산자 (Function-Preserving Structural Mutations)**:
   - 무작위로 뉴런이나 층을 추가하면 기존 학습된 가중치 표현이 파괴되는 문제를 해결하기 위해, **Net2Wider**(기존 뉴런 가중치 분할 복제) 및 **Net2Deeper**(항등 잔차 행렬 $W \approx I$ 초기화) 연산을 적용하여 구조가 변이된 직후에도 부모 신경망의 예측 함수를 보존합니다.
3. **종 분화 보호(Speciation Protection) & 상동 서브행렬 교차(Homologous Sub-matrix Crossover)**:
   - 구조 변이 직후 20 미니배치의 짧은 국소 적응(Speciation Protection)을 부여해 새로운 위상이 도태되지 않고 평가받게 하며, 서로 다른 층수·뉴런 폭을 가진 섬 개체 간에도 **공통 부분 행렬($\min(r_A, r_B) \times \min(c_A, c_B)$)을 통한 상동 교차**가 가능합니다.
4. **미분 불가능한 Macro-F1 결정 경계 유전자 직접 진화 (Boundary ES)**:
   - 각 개체의 출력단에 내장된 6개의 결정 경계 유전자(`class_scale` $\mathbf{s} \in \mathbb{R}^3$, `class_bias` $\mathbf{b} \in \mathbb{R}^3$)를 무미분 진화 전략(ES)으로 직접 진화시켜 비미분 지표인 **Macro-F1**을 극대화합니다.

---

## 2. 6가지 신경망 구조 진화 연산자 (`EvolvableMLP`)

| 구조 변이 연산자 | 메서드 | 수학적 / 구조적 메커니즘 |
|---|---|---|
| **1) 뉴런 폭 확장/축소** | `mutate_width(layer, ±Δ)` | **Net2Wider**: 뉴런 복제 시 다음 층 출력 가중치를 $1/2$씩 분할하여 출력 보존 / **Pruning**: $\ell_1$-노름 하위 뉴런 제거 |
| **2) 은닉층 깊이 추가** | `mutate_add_layer()` | **Net2Deeper**: 마지막 은닉층 뒤에 항등 매핑($W = I + \mathcal{N}(0, 0.01^2)$)에 가까운 신규 은닉층 삽입 (최대 4층) |
| **3) 은닉층 깊이 제거** | `mutate_remove_layer()` | 마지막 은닉층이 이전 층과 동일 폭일 때 불필요한 층을 제거하여 모델 경량화 (최소 2층) |
| **4) 활성화 함수 진화** | `mutate_activation(layer)` | 층별 비선형 함수를 `['relu', 'gelu', 'silu', 'elu', 'tanh']` 중에서 진화적으로 교체 |
| **5) 입력 스킵 연결 토글** | `mutate_skip_connection(layer)` | 원본 10차원 천체 특성 $\mathbf{x} \in \mathbb{R}^{10}$을 깊은 은닉층에 직접 주입($[\mathbf{h}_{l-1}; \mathbf{x}]$)하는 Dense/Skip 경로 생성·해제 |
| **6) 시냅스 가지치기** | `mutate_synapse_pruning()` | 미세 가중치 연결($|w_{ij}|$ 하위 4%)을 0으로 소거하여 희소 연결 구조(Sparse Connectivity) 형성 |

---

## 3. 45세대 진화 후 발견된 8개 섬(Island)의 이종 신경망 구조

| 섬 (Island) | 초기 씨앗 구조 | **최종 진화된 챔피언 신경망 구조 (TWEANN)** | 파라미터 수 | Val Macro-F1 | Test Accuracy |
|---|---|---|---|---|---|
| **Island 0** | `10 -> [48:relu] -> [24:relu] -> 3` | `10 -> [64:relu] -> [24:relu] -> 3` | 2,345 | 0.9691 | 97.20% |
| **Island 1** | `10 -> [64:relu] -> [32:relu] -> 3` | `10 -> [64:relu] -> [32:elu] -> [32:gelu] -> 3` | 3,945 | 0.9693 | 97.36% |
| **Island 2** | `10 -> [64:gelu] -> [32+skip:silu] -> 3` | `10 -> [64:relu] -> [40+skip:silu] -> 3` | 3,833 | 0.9698 | 97.28% |
| **Island 3** | `10 -> [64:silu] -> [32+skip:relu] -> 3` | `10 -> [80:silu] -> [32:gelu] -> [32:gelu] -> [32:relu] -> 3` (4층!) | 5,689 | 0.9699 | 97.35% |
| **Island 4 (챔피언)** | `10 -> [80:gelu] -> [40:gelu] -> 3` | **`10 -> [104:gelu] -> [48:gelu] -> [48+skip:silu] -> 3`** | **9,169** | **0.9713** | **97.43%** |
| **Island 5** | `10 -> [64:relu] -> [48+skip:gelu] -> [24:relu] -> 3` | `10 -> [80:relu] -> [48+skip:gelu] -> [24:elu] -> 3` | 6,505 | 0.9707 | 97.41% |
| **Island 6** | `10 -> [72:elu] -> [36+skip:relu] -> 3` | `10 -> [72:elu] -> [44+skip:gelu] -> 3` | 4,585 | 0.9699 | 97.32% |
| **Island 7** | `10 -> [56:silu] -> [28:silu] -> 3` | `10 -> [64:silu] -> [60:silu] -> 3` | 4,793 | 0.9696 | 97.23% |

---

## 4. 실행 방법

```bash
# Conda 환경 활성화
conda activate dl_env

# 1. 신경망 구조 + 가중치 동시 진화 학습 (기본값: memetic TWEANN, 45세대, 8개 섬)
python EDL/train_edl.py --algo memetic --generations 45 --pop-size 8

# 2. 순수 무미분 자연 진화 전략(Orthogonal ES) 모드로 훈련
python EDL/train_edl.py --algo es --generations 200 --pop-size 80 --outdir EDL/models_es

# 3. 유전 알고리즘(GA) 모드로 훈련
python EDL/train_edl.py --algo ga --generations 100 --pop-size 60 --outdir EDL/models_ga

# 4. 테스트 세트(20,000건) 정밀 평가
python EDL/evaluate_edl.py

# 5. 진화 곡선, 파라미터 복잡도 변화, 혼동 행렬 시각화
python EDL/plot_evolution.py

# 6. 추론 실행 (Self-Check 및 CSV 추론)
python EDL/predict_edl.py
python EDL/predict_edl.py sample_input.csv -o edl_predictions.csv
```

---

## 5. 실험 결과 및 벤치마크 비교 (Held-out Test Split 20,000건)

| 평가 지표 | 초기 EDL (Pure ES, 고정 32-16) | DL 단일 MLP (Backprop, 고정 64-32) | **진화된 단일 챔피언 EDL (TWEANN)** | **이종 위상 명예의 전당 앙상블 EDL (TWEANN)** |
|---|---|---|---|---|
| **신경망 구조** | 고정 `10 -> 32 -> 16 -> 3` | 고정 `10 -> 64 -> 32 -> 3` | **진화됨: `10 -> [104:gelu] -> [48:gelu] -> [48+skip:silu] -> 3`** | **상위 4개 진화 섬 이종 위상 결합** |
| **Test Accuracy** | 96.48% | 97.37% | **97.43%** (+0.06%p vs DL MLP) | **97.45%** (+0.97%p vs 초기 EDL) |
| **Test Macro-F1** | 0.9599 | 0.9700 | **0.9706** (+0.0006 vs DL MLP) | **0.9709** (+0.0110 vs 초기 EDL) |
| **GALAXY F1** | 0.9709 | 0.9780 | 0.9784 | **0.9786** (Recall 98.0%) |
| **STAR F1** | 0.9728 | 0.9850 | 0.9851 | **0.9853** (**Recall 99.8%**) |
| **QSO F1** | 0.9360 | 0.9470 | 0.9483 | **0.9487** (Precision 96.7%) |
| **학습 시간 (Wall-Clock)** | **13.55 s** (190세대) | **24.16 s ± 5.31 s** (1-Seed) / **72.49 s** (3-Seed) | **11.78 s** (섬 1개당 45세대) | **94.22 s** (8개 섬 전체 45세대) |
| **추론 시간 (20,000건 배치)** | **1.42 ms** | **2.83 ms** (단일) / **121.86 ms** (+GBDT) | **7.08 ms** (단일 챔피언) | **27.26 ms** (상위 4개 섬 결합) |
