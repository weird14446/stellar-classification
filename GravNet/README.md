# GravNet: 시공간 중력장 궤도 신경망 (Astrophysical Potential Dynamical Network)

SDSS-17 천체 분류 데이터셋에 대해 **오차 역전파(Backpropagation)나 시간 역전파(BPTT)를 단 한 번도 사용하지 않고**, 실제 천체역학의 **다중 헤일로 플러머 중력 퍼텐셜(Multi-Halo Plummer Gravitational Potential)**과 **심플렉틱 감쇠 리프프로그(Symplectic Dissipative Leapfrog)** 궤도 적분만으로 천체(`GALAXY`, `QSO`, `STAR`)를 분류하는 물리 동역학 기반 신경망 파이프라인입니다.

---

## 1. 핵심 물리·수학 아키텍처

1. **다중 해상도 천체 분광 렌즈 (`SpectralAstrophysicalLens`)**:
   - 10차원 관측 별빛($u, g, r, i, z$, 색지수 4종, 적색편이 `redshift`)에 대해 기하 급수 주파수 격자($\omega \in [0.25, 120.0]$)의 1차원 도플러 조화 파동, 2차 분광 곡률 상호작용($x_i x_j$), 그리고 비등방성 다중 스케일 10차원 Bochner 랜덤 푸리에 기저($M = 3,826$)를 결합하여 $z \approx 0$ 부근의 미세 항성 경계와 고적색편이 퀘이사 스펙트럼을 동시에 분해합니다.
2. **우주 원점 발사 조건 & 섬유 다발 위상 공간 ($\mathbb{R}^6 \times \mathbb{R}^6$)**:
   - 모든 천체 입자는 초기 시각 $t=0$에 2차원 초은하단 궤도 평면의 **미분류 원점 $\mathbf{q}_{\text{orb}}(0) = (0, 0)$**에서 출발하며($T=0$ 정확도 `59.44%`), 4차원 분광 섬유 좌표 $\mathbf{q}_{\text{spec}}(0)$와 도플러 초기 속도 충격 $\mathbf{p}(0) = \Phi(\mathbf{x})\mathbf{W}_p$를 부여받습니다.
3. **다중 헤일로 플러머 중력장과 리아푸노프 안정성 ($\frac{dH}{dt} \le 0$)**:
   - 정삼각형 $120^\circ$ 대칭(`90°`, `210°`, `330°`)으로 배치된 3개 초은하단 주위에 클래스당 $K=6$개(총 18개)의 플러머 중력 핵 $\{\boldsymbol{\mu}_{c,m}, G_{c,m}\}$이 응축되어 닫힌 수식 중력 $\mathbf{F}_{\text{grav}}(\mathbf{q}) = -\nabla_{\mathbf{q}} U(\mathbf{q})$를 행사합니다.
   - 허블 마찰 계수 $\gamma > 0$를 포함한 대칭 스트랭 분할 리프프로그 적분기는 총 해밀토니안 에너지의 단조 감소($\frac{dH}{dt} = -\gamma \|\mathbf{p}\|_2^2 \le 0$)를 보장하며, $T=10$ 스텝 만에 입자를 중력 우물 바닥으로 포획합니다.
4. **무-BPTT 시공간 곡률 조각 & 라그랑주 $L_1$ 안장점 불확실성 탐지**:
   - 궤도를 거꾸로 미분하지 않고 종착 위치 $\mathbf{q}(T)$에서의 볼츠만 포획 잔차와 **종착 조석력 벡터(Terminal Tidal Force Vector)** $\mathbf{f}_{\text{tidal}}$만으로 플러머 질량 $G_{c,m}$, 핵 위치 $\boldsymbol{\mu}_{c,m}$, 발사 렌즈 $(\mathbf{W}_q, \mathbf{W}_p)$를 직접 조각합니다.
   - 두 초은하단의 중력이 비기는 **라그랑주 $L_1$ 능선 장벽 $\Delta E_{L_1}$**을 측정하여, 별도의 불확실성 모듈 없이도 모은하 빛이 섞인 저적색편이 퀘이사 경계 천체를 물리적으로 식별합니다.

---

## 2. 디렉토리 구성

```
GravNet/
├── preprocess.py           # DL/EDL과 100% 동일한 결정론적 63,999 / 16,000 / 20,000 분할
├── physics_model.py        # 플러머 중력장, 심플렉틱 리프프로그, 무-BPTT 조석 학습, SymPy 검증
├── train_gravnet.py        # 단일 우주 및 3-우주 다중우주(Multiverse) 학습 & 속도 측정 CLI
├── evaluate_gravnet.py     # 20,000건 테스트 정밀 평가 및 라그랑주 L1 안장점 불확실성 분석
├── plot_gravnet.py         # 중력장 등퍼텐셜·궤도, 리아푸노프 소산, 라그랑주 분석 시각화 (4종)
├── predict_gravnet.py      # CSV 추론(포획 확률 + 라그랑주 장벽 출력) 및 Self-Check CLI
├── models/                 # 학습된 체크포인트 (.joblib) 및 평가 지표 (.json)
└── figures/                # 300 DPI 출판 품질 시각화 차트 (.png)
```

---

## 3. 실행 방법 (Quickstart)

```bash
# Conda 가상환경 활성화
conda activate dl_env

# 1. 기호(SymPy) 및 수치 물리 방정식 자가 검증
python GravNet/physics_model.py

# 2. GravNet 단일 우주 및 3-우주 다중우주 학습
python GravNet/train_gravnet.py --epochs 18 --orbital-steps 10

# 3. 20,000건 테스트 세트 평가 및 라그랑주 L1 불확실성 분석
python GravNet/evaluate_gravnet.py

# 4. 과학 시각화 그래프 4종 생성 (GravNet/figures/)
python GravNet/plot_gravnet.py

# 5. 테스트 세트 자가 검증 (Self-Check) 및 외부 CSV 추론
python GravNet/predict_gravnet.py
python GravNet/predict_gravnet.py --input star_classification.csv --output gravnet_preds.csv
```

---

## 4. 벤치마크 성능 요약 (20,000건 Held-out Test Split)

| 모델 패러다임 | 역전파 사용 | Test Accuracy | Test Macro-F1 | STAR Recall | 학습 시간 |
|---|---|---|---|---|---|
| `EDL` 순수 진화 전략 (`32-16`) | 미사용 | 96.48% | 0.9599 | 99.63% | 13.55 s |
| `DL` 단일 MLP (`64-32`, Adam) | 전역 역전파 | 97.37% | 0.9700 | 99.72% | 24.16 s |
| `DL` 3-Seed MLP 앙상블 | 전역 역전파 | 97.42% | 0.9705 | 99.77% | 72.49 s |
| `EDL` TWEANN 단일 챔피언 | 비미분 진화 결합 | 97.43% | 0.9706 | 99.79% | 94.22 s (섬당 11.78s) |
| `EDL` TWEANN 상위 4섬 앙상블 | 비미분 진화 결합 | 97.45% | 0.9709 | 99.84% | 94.22 s |
| **`GravNet` 단일 우주 (18 Halos)** | **완전 무역전파** | **97.55%** | **0.9718** | **99.93%** | **40.69 s** (1단계 7.45s) |
| **`GravNet` 3-우주 다중우주** | **완전 무역전파** | **97.59%** | **0.9724** | **100.00% (4,319/4,319)** | **119.83 s** |

- **라그랑주 $L_1$ 안장점 물리 불확실성 분리 효과**:
  - 깊은 중력 우물 포획군 (상위 95%, 19,000개 천체): 테스트 정확도 **`98.71%`** (`Macro-F1 0.9848`)
  - 라그랑주 $L_1$ 능선 경계군 (하위 5%, 1,000개 천체): 테스트 정확도 **`75.50%`** (퀘이사 중앙 적색편이 $z = 0.977$ vs 깊은 우물 $z = 1.727$)
