# 대안적 및 신개념 딥러닝 학습 패러다임 총람 (Catalog of Alternative & Novel Deep Learning Paradigms)

본 문서는 SDSS-17 천체 분류 프로젝트에서 기존에 구축한 **① 전역 오차 역전파 딥러닝 (`DL/`, Backpropagation)** 및 **② 생물학적 위상·가중치 진화 신경망 (`EDL/`, Neuroevolution / TWEANN)**을 넘어, **신경망이 정보를 표현하고 학습하는 원리 자체를 새롭게 정의하는 차세대 학습 모델 후보군**을 체계적으로 기록한 연구 설계 문서입니다.

문서는 크게 두 파트로 구성됩니다:
- **Part I. 비역전파·해석적 차세대 학술 패러다임 (5종 - 향후 후보 연구 과제)**: 현대 머신러닝 학계에서 오차 역전파의 생물학적 비현실성·블랙박스 한계를 극복하기 위해 제안된 이론적 설계 모델 (`FFDL`, `ELM`, `KAN`, `EP/Hebb`, `BNN`)
- **Part II. 물리·파동·복잡계 원리 기반 독창적 신개념 모델 (4종)**: 본 천체 분류 도메인('우주, 중력, 빛의 파동, 공명, 창발')으로부터 제1원리(First Principles)로 새롭게 고안한 패러다임 전환형 모델:
  - **`GravNet` (Part II-1)**: **저장소 내 완전 구현 및 검증 완료** (단일 `97.55%`, 3-우주 다중우주 `97.59%`, 항성 전수 포획 `100.00%`).
  - **`QWIN`, `NCA-Jury`, `ResonanceNet` (Part II-2~4)**: 후속 확장을 위한 독창적 물리·복잡계 이론 설계 모델.

---

## 한눈에 보는 전체 후보 모델 비교 매트릭스

| 구분 | 모델명 (코드명) | 핵심 추론 메커니즘 | 핵심 학습 메커니즘 | 역전파(BP) 사용 | 주요 강점 및 과학적 의의 |
|---|---|---|---|---|---|
| **기준선 1** | **정통 다층 신경망 (`DL/`)** | 층별 행렬곱 + ReLU | 전역 연쇄법칙 오차 역전파 (Adam) | 전역 BP 사용 | 빠르고 안정적인 표준 베이스라인 (`97.37%`) |
| **기준선 2** | **위상·가중치 진화망 (`EDL/`)** | 자율 진화된 이종 스킵 위상 | Net2Net 구조 변이 + 라마르크 + 경계 ES | 비미분 진화 결합 | 뇌 구조 자체를 스스로 발명 (`97.45%`) |
| **Part I-1** | **전방-전방 신경망 (`FFDL`)** | 층별 활성 에너지(Goodness) 계산 | 긍정/부정 샘플 2회 순전파 층별 국소 학습 | **무역전파** | 뇌 시냅스의 국소 가소성에 부합 (Hinton, 2022) |
| **Part I-2** | **극한 학습 기계 (`ELM`)** | 고차원 무작위 푸리에/비선형 투영 | 티호노프 정규화 닫힌 해(Closed-Form) 직결 | **무역전파 (0 Epoch)** | 0.3초 이내 단 한 번의 행렬 방정식으로 전역 최적해 도출 |
| **Part I-3** | **콜모고로프-아놀드망 (`KAN`)** | 시냅스 간선 위의 스플라인 함수 합산 | B-스플라인/체비쇼프 기저 계수 최적화 | 기저 계수 미분 | 10차원 천체 물리량 간 비선형 수식 시각화·기호 회귀 가능 |
| **Part I-4** | **평형 전파 / 헤비안망 (`EP/Hebb`)** | 에너지 함수 동역학 평형점 수렴 | 자유 위상 vs 고정 위상 국소 에너지 차이 | **무역전파** | 물리적 아날로그 회로·뉴로모픽 칩 구현 가능 |
| **Part I-5** | **베이지안 변분 신경망 (`BNN`)** | 가중치 분포 몬테카를로 샘플링 | 변분 자유 에너지(ELBO) 최대화 | 재매개변수 미분 | 모호한 저적색편이 퀘이사에 대해 정량적 불확실성 제공 |
| **Part II-1** | **시공간 중력장 궤도망 (`GravNet`)** | **플러머 중력장 내 입자 궤도 시뮬레이션** | **무-BPTT 시공간 곡률 조각 + 조석 헤비안** | **무-BPTT 물리 학습** | **구현 완료: 단일 `97.55%` / 다중우주 `97.59%` (`STAR` 재현율 100%)** |
| **Part II-2** | **복소 파동 간섭망 (`QWIN`)** | 복소 물질파 유니타리 광학 간섭계 | 위상 공액 홀로그래피 간섭 기록 | 복소 위상 동기화 | 별빛을 실제 파동으로 취급, 보강·상쇄 간섭으로 분류 |
| **Part II-3** | **세포 군집 합의망 (`NCA-Jury`)** | $8\times 8$ 격자 세포의 국소 확산·토론 | 공유 미소 유전자 규칙 최적화 | 국소/진화 규칙 | 중앙 통제 없는 집단지성 합의 및 세포 손상 시 자가 치유 |
| **Part II-4** | **스펙트럼 공명 역추론망 (`ResonanceNet`)** | 클래스별 독립 다양체 생성기 공명 측정 | 단일 종족 데이터 기반 독립 다양체 학습 | 클래스간 경계 미사용 | 미지의 특이 천체(이상치) 자동 검출 및 무한 클래스 확장성 |

---

## Part I. 비역전파 및 해석적 차세대 학술 패러다임 상세

### 1. 전방-전방 알고리즘 (Forward-Forward Algorithm, `FFDL`)
- **제안 배경**: 생물학적 대뇌 피질에는 순전파 경로와 정확히 대칭인 가중치 전치 행렬($\mathbf{W}^\top$)을 따라 오차를 거꾸로 흘려보내는 역전파 회로(Weight Transport Problem)가 존재하지 않습니다.
- **수학적 정식화**:
  - 입력 특성 $\mathbf{x} \in \mathbb{R}^{10}$와 원-핫 클래스 벡터 $\mathbf{y} \in \{0, 1\}^3$를 결합한 입력 $(\mathbf{x}, \mathbf{y})$를 구성합니다.
  - 긍정 데이터 $(\mathbf{x}, \mathbf{y}_{\text{true}})$에 대해서는 각 은닉층 $l$의 활성 벡터 $\mathbf{h}_l \in \mathbb{R}^{d_l}$의 제곱 노름(Goodness) $G_l(\mathbf{h}_l) = \|\mathbf{h}_l\|_2^2$가 임계값 $\theta_{\text{th}} > 0$을 넘도록 하고, 부정 데이터 $(\mathbf{x}, \mathbf{y}_{\text{false}})$에 대해서는 $\theta_{\text{th}}$ 아래로 억제하는 층별 국소 손실함수 $\mathcal{L}_l$을 독립적으로 최소화합니다:
    $$\mathcal{L}_l = \ln\!\Big(1 + \exp\big(-(G_l(\mathbf{h}_l^{\text{pos}}) - \theta_{\text{th}})\big)\Big) + \ln\!\Big(1 + \exp\big(G_l(\mathbf{h}_l^{\text{neg}}) - \theta_{\text{th}}\big)\Big)$$
  - 다음 층 $l+1$로 전달할 때는 크기 정보를 제거한 단위 방향 벡터 $\hat{\mathbf{h}}_l = \frac{\mathbf{h}_l}{\|\mathbf{h}_l\|_2 + \varepsilon}$ ($\varepsilon > 0$)만 넘겨주어, 상위 층이 하위 층의 크기 값에 무임승차하지 않고 고차 특징을 새롭게 추출하도록 유도합니다.

### 2. 닫힌 해 극한 학습 기계 및 랜덤 푸리에 커널 신경망 (`ELM` / `RFN`)
- **제안 배경**: 다층 신경망의 반복적 경사하강법은 비볼록(Non-convex) 손실 지형에서 수십~수백 에폭을 소모합니다. Bochner 정리와 무작위 특징 이론(Rahimi & Recht, 2007)을 활용하면 비선형 커널 공간을 유한 차원 신경망으로 근사한 뒤 선형대수 닫힌 해로 즉시 풀 수 있습니다.
- **수학적 정식화**:
  - 다중 스케일 주파수 행렬 $\mathbf{\Omega} \sim \mathcal{N}(\mathbf{0}, \gamma_{\text{RBF}}^2 \mathbf{I})$ 및 위상 $\mathbf{b} \sim \mathcal{U}(0, 2\pi)$를 고정하고 비선형 은닉 특징 행렬 $\mathbf{H} \in \mathbb{R}^{N \times M}$을 구성합니다:
    $$\mathbf{H} = \sqrt{\frac{2}{M}} \cos(\mathbf{X}\mathbf{\Omega} + \mathbf{1}\mathbf{b}^\top)$$
  - 정규화 계수 $\lambda > 0$에 대하여 그람 행렬 $\mathbf{G}_\lambda = \mathbf{H}^\top \mathbf{H} + \lambda \mathbf{I}_M$은 대칭 양의 정부호(Symmetric Positive Definite, $\det(\mathbf{G}_\lambda) > 0$)이므로, 숄레스키 분해(Cholesky Decomposition)를 통해 출력 가중치 $\mathbf{B}^* \in \mathbb{R}^{M \times 3}$의 전역 유일 최적해를 **단 1회의 행렬 연산(0 Epoch)**으로 계산합니다:
    $$\mathbf{B}^* = (\mathbf{H}^\top \mathbf{H} + \lambda \mathbf{I}_M)^{-1} \mathbf{H}^\top \mathbf{Y}$$

### 3. 콜모고로프-아놀드 네트워크 (`KAN`: Kolmogorov-Arnold Networks)
- **제안 배경**: 콜모고로프-아놀드 표현 정리(Kolmogorov-Arnold Representation Theorem)는 임의의 다변수 연속 함수가 1변수 연속 함수들의 유한 합성과 덧셈으로 정확히 표현됨을 보장합니다.
- **수학적 정식화**:
  - 선형 가중치 행렬 대신 각 시냅스 간선 $(l, i, j)$가 기저 함수 $b(x) = \text{SiLU}(x)$와 $G$-구간 B-스플라인 기저 $\{B_k(x)\}$의 선형 결합을 학습합니다:
    $$\phi_{l,j,i}(x) = w_b \, b(x) + w_s \sum_{k=1}^{G+p} c_{l,j,i,k} B_k(x)$$
  - 천체물리 변수($\text{redshift}$, $u-g$ 등)가 어떤 비선형 함수 곡선으로 변환되는지 직접 플롯하고 기호 수식(Symbolic Formula)으로 추출할 수 있습니다.

### 4. 에너지 평형 전파 및 소프트 헤비안 신경망 (`Equilibrium Propagation` & `SoftHebb`)
- **수학적 정식화**:
  - 대칭 연결 가중치 $\mathbf{W}$를 갖는 연속 상태 네트워크의 리아푸노프 에너지 함수 $E(\mathbf{s}, \mathbf{x}; \mathbf{W})$를 정의합니다.
  - 외력이 없는 자유 평형 상태 $\mathbf{s}^0$ ($\nabla_{\mathbf{s}} E(\mathbf{s}^0) = \mathbf{0}$)와 출력단을 정답 $\mathbf{y}$ 쪽으로 미세 계수 $\beta > 0$만큼 끌어당긴 약한 고정 평형 상태 $\mathbf{s}^\beta$ 두 지점에서의 국소 뉴런 상관관계 차이만으로 역전파와 수학적으로 동치인 기울기를 얻습니다(Scellier & Bengio, 2017):
    $$\Delta w_{ij} \propto \frac{1}{\beta} \left( \rho(s_i^\beta)\rho(s_j^\beta) - \rho(s_i^0)\rho(s_j^0) \right)$$

### 5. 베이지안 변분 추론 신경망 (`Bayesian Variational Neural Network`)
- **수학적 정식화**:
  - 가중치 사전 분포 $p(\boldsymbol{\theta}) = \mathcal{N}(\mathbf{0}, \sigma_0^2 \mathbf{I})$ 하에서 변분 사후 분포 $q_\phi(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2))$ ($\boldsymbol{\sigma} = \ln(1 + \exp(\boldsymbol{\rho})) > \mathbf{0}$)를 학습합니다.
  - 추론 시 $S$회 가중치 표본 추출을 통해 예측 확률의 기대값 $\bar{\mathbf{p}}(\mathbf{x})$뿐만 아니라 **상호정보량 기반 인식론적 불확실성(Epistemic Uncertainty)** $\mathcal{I}(y; \boldsymbol{\theta} \mid \mathbf{x}) = \mathcal{H}[\bar{\mathbf{p}}(\mathbf{x})] - \frac{1}{S}\sum_{s=1}^S \mathcal{H}[\mathbf{p}(\mathbf{x}; \boldsymbol{\theta}^{(s)})]$를 정량화하여, 모은하 빛이 섞인 저적색편이 퀘이사를 자동으로 선별합니다.

---

## Part II. 물리·파동·복잡계 원리 기반 독창적 신개념 모델 (4종) 상세

### 1. 시공간 중력장 궤도 신경망 (`GravNet`: Astrophysical Potential Dynamical Network) ★공식 구현 완료 모델★

- **구현 코드 바로가기**: [`GravNet/physics_model.py`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/GravNet/physics_model.py), [`GravNet/train_gravnet.py`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/GravNet/train_gravnet.py), [`GravNet/evaluate_gravnet.py`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/GravNet/evaluate_gravnet.py), [`GravNet/predict_gravnet.py`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/GravNet/predict_gravnet.py)
- **정식 학술 논문 및 보고서**: [IEEE 저널 논문 (docs/gravnet_ieee.pdf)](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/gravnet_ieee.pdf), [기술 보고서 (docs/gravnet_research_report.pdf)](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/gravnet_research_report.pdf)

#### (1) 핵심 발상
천체를 분류하는 인공지능 내부에 아예 **실제 천체역학 시뮬레이션(Plummer 다체 중력장 + 우주 마찰 감쇠 동역학)**을 구축합니다. 신경망의 '은닉층'을 없애고, 관측된 천체 $\mathbf{x} \in \mathbb{R}^{10}$를 위상 공간 $(\mathbf{q}(0), \mathbf{p}(0)) \in \mathbb{R}^d \times \mathbb{R}^d$ ($d=6$)의 **천체 입자(Particle)**로 발사하여 중력장을 따라 스스로 궤도 운동하며 자신의 클래스 은하단(중력 우물)으로 포획되게 만듭니다.

#### (2) 물리적·수학적 구조 및 3단계 학습 정식화

1. **다중 해상도 분광 렌즈 ($\Phi(\mathbf{x}) \in \mathbb{R}^{3826}$) 및 우주 원점 발사 조건**:
   - 극단치 완충 $\tilde{\mathbf{x}} = 15\tanh(\mathbf{x}/15)$에 대해, $z \approx 0$ 근방의 미세 도플러 경계를 분해하는 1차원 조화 기저 $\boldsymbol{\omega}_{\text{1D}} \in \mathbb{R}^{44}$ ($\omega \in [0.25, 120.0]$ 로그 스케일), SED 2차 곡률 $\frac{1}{4}(\tilde{x}_i \tilde{x}_j)_{i \le j}$, 다중 대역폭 가우시안 Bochner 랜덤 푸리에 모드 $\boldsymbol{\Omega} \in \mathbb{R}^{10 \times 2880}$를 결합한 $M=3,826$차원 분광 광학 렌즈 $\Phi(\mathbf{x})$를 구성합니다:
     $$\Phi(\mathbf{x}) = \Big[ 1 \;\Big\Vert\; \tilde{\mathbf{x}}^\top \;\Big\Vert\; \tfrac{1}{4}(\tilde{x}_i \tilde{x}_j)_{i \le j} \;\Big\Vert\; \tfrac{1}{\sqrt{F}}\sin(\tilde{\mathbf{x}}\otimes\boldsymbol{\omega}) \;\Big\Vert\; \tfrac{1}{\sqrt{F}}\cos(\tilde{\mathbf{x}}\otimes\boldsymbol{\omega}) \;\Big\Vert\; \sqrt{\tfrac{2}{M_{\text{RFF}}}}\cos(\tilde{\mathbf{x}}^\top\boldsymbol{\Omega} + \boldsymbol{\varphi}^\top) \Big]^\top$$
   - **우주 원점 발사 구속**: 발사 행렬 $[\mathbf{W}_q]_{:, 1:2} = \mathbf{0}$으로 구속되어 모든 입자는 초기 시각 $t=0$에서 초은하단 궤도 평면의 미분류 원점 $\mathbf{q}_{\text{orb}}(0) = (0, 0)$에서 출발하며, 4차원 분광 섬유 좌표 $\mathbf{q}_{\text{spec}}(0) = \Phi(\mathbf{x})^\top [\mathbf{W}_q]_{:, 3:6}$ 및 초기 모멘텀 $\mathbf{p}(0) = \Phi(\mathbf{x})^\top \mathbf{W}_p$를 부여받습니다.

2. **다중 헤일로 플러머 중력 퍼텐셜 ($K=6$ 서브헤일로, 총 18개 중력 핵)**:
   구상성단과 은하 헤일로를 모델링하는 실제 천체물리학의 **플러머 퍼텐셜(Plummer, 1911)**을 도입합니다. 각 천체 클래스 $c \in \{0, 1, 2\}$마다 $K = 6$개의 플러머 중력 핵 위치 $\{\boldsymbol{\mu}_{c,m} \in \mathbb{R}^6\}_{m=1}^6$과 양의 중력 질량 $G_{c,m} > 0$ (총 $3 \times 6 = 18$개 중력원)이 존재하며, 연화 길이 $\epsilon = 0.35$, 우주 조화 가둠 계수 $\kappa = 0.02$ 하에서 총 퍼텐셜 $U(\mathbf{q})$는 다음과 같습니다:
   $$U(\mathbf{q}) = -\sum_{c=0}^{2} \sum_{m=1}^{K} \frac{G_{c,m}}{\sqrt{\|\mathbf{q} - \boldsymbol{\mu}_{c,m}\|_2^2 + \epsilon^2}} + \frac{1}{2}\kappa \|\mathbf{q}\|_2^2$$
   위치 $\mathbf{q}$에서 입자가 받는 해석적 중력 $\mathbf{F}_{\text{grav}}(\mathbf{q}) = -\nabla_{\mathbf{q}} U(\mathbf{q})$는 자동미분 없이 닫힌 수식으로 즉시 계산됩니다:
   $$\mathbf{F}_{\text{grav}}(\mathbf{q}) = -\sum_{c=0}^{2} \sum_{m=1}^{K} G_{c,m} \frac{\mathbf{q} - \boldsymbol{\mu}_{c,m}}{\left(\|\mathbf{q} - \boldsymbol{\mu}_{c,m}\|_2^2 + \epsilon^2\right)^{3/2}} - \kappa \mathbf{q}$$

3. **심플렉틱 감쇠 리프프로그 적분기 및 리아푸노프 단조 소산 정리**:
   입자는 허블 마찰 감쇠 계수 $\gamma > 0$ 하에서 해밀턴 운동 방정식 $\dot{\mathbf{q}} = \mathbf{p}$, $\dot{\mathbf{p}} = \mathbf{F}_{\text{grav}}(\mathbf{q}) - \gamma \mathbf{p}$를 따릅니다. 총 역학적 에너지 $H(\mathbf{q}, \mathbf{p}) = \frac{1}{2}\|\mathbf{p}\|_2^2 + U(\mathbf{q})$의 시간 도함수는
   $$\frac{dH}{dt} = \langle \nabla_{\mathbf{q}} U(\mathbf{q}), \mathbf{p} \rangle + \langle \mathbf{p}, -\nabla_{\mathbf{q}} U(\mathbf{q}) - \gamma \mathbf{p} \rangle = -\gamma \|\mathbf{p}\|_2^2 \le 0$$
   이므로, 라살의 불변성 원리(LaSalle's Invariance Principle)에 의해 모든 천체 입자는 유한 시간 내에 운동 에너지를 잃고 중력 우물 바닥(안정 평형점)으로 수렴합니다. 대칭 스트랭 분할(Strang-Split Leapfrog, $\Delta t = 0.11$, 감쇠 인자 $\eta_d = e^{-\gamma \Delta t / 2} \in (0, 1)$)을 적용합니다:
   $$\begin{aligned}
   \mathbf{p}_{t + 1/2} &= \eta_d \, \mathbf{p}_t + \frac{\Delta t}{2} \mathbf{F}_{\text{grav}}(\mathbf{q}_t), \\
   \mathbf{q}_{t + 1} &= \mathbf{q}_t + \Delta t \, \mathbf{p}_{t + 1/2}, \\
   \mathbf{p}_{t + 1} &= \eta_d \, \mathbf{p}_{t + 1/2} + \frac{\Delta t}{2} \mathbf{F}_{\text{grav}}(\mathbf{q}_{t+1}).
   \end{aligned}$$
   외력이 없을 때 단위 속도당 $T$ 스텝 누적 탄도 도달 거리 $R_{\text{reach}}(T) = \Delta t \, e^{-\gamma\Delta t/2}\frac{1 - e^{-\gamma T\Delta t}}{1 - e^{-\gamma\Delta t}}$를 닫힌 해로 유도하여 초기 탄도 발사에 반영합니다.

4. **3단계 순수 물리 학습 법칙 (100% Backpropagation-Free)**:
   - **1단계 (Cholesky 탄도 발사 보정)**: 렌즈 특징 그람 행렬 $\mathbf{G}_\lambda = \boldsymbol{\Phi}^\top \boldsymbol{\Phi} + \lambda \mathbf{I}_M$을 숄레스키 분해($\mathbf{G}_\lambda = \mathbf{L}\mathbf{L}^\top$)하여 초기 탄도 발사 행렬 $\mathbf{W}_p, \mathbf{W}_q$를 단 7.45초 만에 닫힌 해로 최적화합니다.
   - **2단계 (무-BPTT 시공간 곡률 조각 및 종착 조석력 가소성)**: $T$스텝 후 종착 위치 $\mathbf{q}_T$에서 대정준 볼츠만 포획 확률 $P(c \mid \mathbf{q}_T)$의 잔차 $r_{c} = \mathbb{I}[c=y] - P(c \mid \mathbf{q}_T)$에 기반하여, 플러머 질량 증감($\Delta G_{c,m}$), 은하단 중심 완화($\Delta \boldsymbol{\mu}_{c,m}$), 그리고 종착 조석력 벡터 $\mathbf{f}_{\text{tidal}}(\mathbf{q}_T) \triangleq -\nabla_{\mathbf{q}_T}\mathcal{L}$에 의한 발사 렌즈 헤비안 갱신($\Delta \mathbf{W}_p = \frac{1}{B}\boldsymbol{\Phi}_B^\top \mathbf{f}_{\text{tidal}} - \lambda_w \mathbf{W}_p$)을 국소적으로 수행합니다.
   - **3단계 (대정준 화학 퍼텐셜 보정 및 라그랑주 $L_1$ 안장점 분석)**: 결합 퍼텐셜 우물 깊이 비대칭성을 해소하기 위해 검증 세트에서 Macro-$F_1$을 직접 극대화하는 무미분 화학 퍼텐셜 $\boldsymbol{\nu} = (\nu_0, \nu_1, \nu_2)^\top$ ($\sum \nu_c = 0$) 탐색을 수행하고, 라그랑주 $L_1$ 안장점 퍼텐셜 장벽 $\Delta E_{L_1}(\mathbf{q}_T) \triangleq E_{(2)}(\mathbf{q}_T) - E_{(1)}(\mathbf{q}_T) \ge 0$을 통해 모호한 천체를 물리적으로 분리합니다.

5. **실제 구현 및 벤치마크 검증 성과**:
   - **기호 및 수치 엄밀성 검증**: `SymPy` 기호 연산으로 중력장 잔차 $\mathbf{0}$ 및 해밀토니안 소산 항등식 $\dot{H} + \gamma\|\mathbf{p}\|_2^2 = 0$ 증명 완료, 유한차분 오차 $1.84 \times 10^{-10}$ 이내 확인.
   - **벤치마크 성능 (20,000건 테스트 세트)**:
     - **단일 우주 `GravNet` (18개 플러머 헤일로)**: 정확도 **`97.55%`** / Macro-$F_1$ **`0.9718`** (학습 시간 `40.69s`, 1단계 단독 `7.45s`) — 역전파 `DL` 단일/3-Seed MLP(`97.37%`/`97.42%`) 및 `EDL/TWEANN` 앙상블(`97.45%`) 상회.
     - **3-우주 다중우주 `GravNet` (`Multiverse`)**: 정확도 **`97.59%`** / Macro-$F_1$ **`0.9724`**, 항성(`STAR`) 재현율 **`100.00%` (4,319/4,319 전수 포획)**.
     - **라그랑주 $L_1$ 분리 효과**: 상위 95% 깊은 우물 포획군 정확도 **`98.71%`** vs 하위 5% $L_1$ 능선 경계군 **`75.50%`**.

---

### 2. 복소 파동 간섭 홀로그래피 신경망 (`QWIN`: Quantum Wave-Interference Network)

#### (1) 핵심 발상
우리가 관측하는 $u, g, r, i, z$ 등급과 적색편이는 본질적으로 우주 공간을 전파해 온 **전자기파(빛의 파동)**입니다. 실수 스칼라 가중치 대신 신경망 전체를 **진폭(Amplitude)과 위상(Phase)을 가진 복소 물질파 광학 간섭계**로 구성합니다.

#### (2) 수학적 정식화 및 학습 메커니즘
1. **복소 파동함수 인코딩**:
   - 입력 천체 $\mathbf{x} \in \mathbb{R}^{10}$로부터 진폭 포락선 $a_k(\mathbf{x}) > 0$과 스펙트럼 위상각 $\phi_k(\mathbf{x}) \in [0, 2\pi)$를 생성하여 정규화된 복소 상태 벡터 $|\psi_0(\mathbf{x})\rangle \in \mathbb{C}^H$ ($\|\psi_0(\mathbf{x})\|_2 = 1$)를 구성합니다:
     $$\psi_{0,k}(\mathbf{x}) = \frac{a_k(\mathbf{x}) \exp\!\big(i\,\phi_k(\mathbf{x})\big)}{\sqrt{\sum_{j=1}^{H} a_j(\mathbf{x})^2}}, \qquad i = \sqrt{-1}$$
2. **케일리 변환(Cayley Transform) 기반 에너지 보존 유니타리 광학계**:
   - 에르미트 행렬 $\mathbf{A}_l = \mathbf{B}_l + \mathbf{B}_l^\dagger$ ($\mathbf{A}_l^\dagger = \mathbf{A}_l$, 고유값 $\lambda \in \mathbb{R}$이므로 $\det(\mathbf{I}_H + i\mathbf{A}_l) \neq 0$ 항상 보장)로부터 엄밀한 유니타리 산란 행렬 $\mathbf{U}_l \in \mathrm{U}(H)$를 구성합니다:
     $$\mathbf{U}_l = (\mathbf{I}_H - i\mathbf{A}_l)(\mathbf{I}_H + i\mathbf{A}_l)^{-1}, \qquad \mathbf{U}_l^\dagger \mathbf{U}_l = \mathbf{I}_H$$
   - 파동 크기의 폭발이나 소실($\|\mathbf{U}_l \psi\|_2 = \|\psi\|_2 = 1$)이 원천적으로 불가능합니다.
3. **보른 규칙(Born Rule) 슬릿 간섭 검출 및 홀로그래피 위상 공액 학습**:
   - 최종 파동 $|\psi_L(\mathbf{x})\rangle$가 3개의 클래스 공명 슬릿 $\{|\chi_c\rangle\}_{c=0}^2$에 도달할 때의 복소 내적 제곱으로 검출 확률을 판정합니다:
     $$P(c \mid \mathbf{x}) = \big|\langle \chi_c \mid \psi_L(\mathbf{x})\rangle\big|^2 = \left|\sum_{k=1}^{H} \chi_{c,k}^* \, \psi_{L,k}(\mathbf{x})\right|^2$$
   - 레이저 홀로그램 기록 원리인 참조파-물체파 간섭 외적($|\chi_y\rangle\langle\psi(\mathbf{x})|$)을 통해 정답 슬릿에서는 위상차 $\Delta\phi \to 0$(보강 간섭), 오답 슬릿에서는 $\Delta\phi \to \pi$(상쇄 간섭 소멸)를 유도합니다.

---

### 3. 자가 치유 세포 군집 합의 모델 (`NCA-Jury`: Neural Cellular Automata Swarm)

#### (1) 핵심 발상
단일 중앙 판단자(출력층)를 없애고, $8 \times 8 = 64$개의 미세 세포(Cell)들이 2차원 격자에 모여 상하좌우 이웃 세포와만 정보를 교환하며 스스로 패턴을 형성(Morphogenesis)하여 만장일치 합의를 도출하는 분산 복잡계 모델입니다.

#### (2) 수학적 정식화 및 학습 메커니즘
1. **격자 상태 공간과 공간 미분 지각 필터**:
   - 시각 $t$에서 각 격자점 $(u, v) \in \{1,\dots,8\}^2$의 세포는 상태 벡터 $\mathbf{s}_{u,v}^{(t)} \in \mathbb{R}^{16}$을 보유합니다(채널 $0\text{--}2$: `GALAXY/QSO/STAR` 투표 농도, 채널 $3\text{--}12$: 관측 파장 수용체, 채널 $13\text{--}15$: 확산 신호 화학물질).
   - 각 세포는 $3 \times 3$ 소벨 기울기 커널($\mathbf{K}_x, \mathbf{K}_y$)과 이산 라플라시안 확산 커널($\mathbf{K}_{\Delta}$)을 통해서만 국소 지각 벡터 $\mathbf{z}_{u,v}^{(t)} = [\mathbf{s}_{u,v}^{(t)} \,\Vert\, (\mathbf{K}_x * \mathbf{S}^{(t)})_{u,v} \,\Vert\, (\mathbf{K}_y * \mathbf{S}^{(t)})_{u,v} \,\Vert\, (\mathbf{K}_{\Delta} * \mathbf{S}^{(t)})_{u,v}]$를 읽습니다.
2. **단일 공유 유전자 규칙(Shared Micro-DNA Rule) 및 자가 치유(Self-Healing)**:
   - 64개 세포 전체가 약 800개 파라미터 크기의 **단일 공유 업데이트 함수 $g_\theta$**와 확률적 비동기 마스크 $m_{u,v}^{(t)} \sim \operatorname{Bernoulli}(p_{\text{fire}})$만으로 상태를 갱신합니다:
     $$\mathbf{s}_{u,v}^{(t+1)} = \mathbf{s}_{u,v}^{(t)} + m_{u,v}^{(t)} \, g_\theta\!\left(\mathbf{z}_{u,v}^{(t)}\right)$$
   - 추론 도중 격자의 $30\%\text{--}50\%$ 세포를 강제로 0으로 지워버리는 손상(Lesion)을 가해도, 인접 세포들의 확산 신호가 결손 영역을 재분극시켜 합의 색상을 복구합니다.

---

### 4. 스펙트럼 공명 역추론 모델 (`ResonanceNet`: Analysis-by-Synthesis Spectral Resonance)

#### (1) 핵심 발상
기존 분류기들은 클래스 사이의 경계선(Decision Boundary)만 외우기 때문에 한 번도 본 적 없는 기이한 천체(초신성, 중력렌즈 다중상, 기기 노이즈)가 들어와도 99% 확신으로 기존 셋 중 하나로 오판합니다. `ResonanceNet`은 경계선을 전혀 학습하지 않고 **각 천체 종족의 고유 물리 법칙 다양체(Manifold)를 독립된 공명기로 학습**합니다.

#### (2) 수학적 정식화 및 학습 메커니즘
1. **완전 격리된 종족별 다양체 생성·복원기 학습**:
   - `GALAXY` 데이터만으로 $\mathcal{R}_{\theta_0}$, `QSO` 데이터만으로 $\mathcal{R}_{\theta_1}$, `STAR` 데이터만으로 $\mathcal{R}_{\theta_2}$라는 3개의 저차원 병목 다양체 사상($\mathbb{R}^{10} \to \mathbb{R}^{k} \to \mathbb{R}^{10}$, $k < 10$)과 잔차 공분산 행렬 $\boldsymbol{\Sigma}_c \succ \mathbf{0}$ ($\det(\boldsymbol{\Sigma}_c) > 0$)을 서로의 존재를 모른 채 독립적으로 학습합니다.
2. **마할라노비스 스펙트럼 공명 에너지 및 미지 천체(OOD) 경보**:
   - 새로운 천체 $\mathbf{x}$를 세 공명기에 통과시켜 고유 스펙트럼 복원 오차(공명 비조화 에너지)를 계산합니다:
     $$E_c(\mathbf{x}) = \big(\mathbf{x} - \mathcal{R}_{\theta_c}(\mathbf{x})\big)^\top \boldsymbol{\Sigma}_c^{-1} \big(\mathbf{x} - \mathcal{R}_{\theta_c}(\mathbf{x})\big) + \ln\det(\boldsymbol{\Sigma}_c)$$
   - 가장 낮은 비조화 에너지(최대 공명)를 보이는 클래스 $\hat{c} = \arg\min_{c \in \{0,1,2\}} E_c(\mathbf{x})$로 분류하되, 만약 $\min_c E_c(\mathbf{x}) > \tau_{\text{OOD}}$ ($\chi^2_{10, 1-\alpha}$ 임계값)이면 **"기존 3종족에 속하지 않는 미지의 특이 천체(Unknown Celestial Anomaly)"**로 즉시 분리합니다. 또한 향후 제4의 천체 종족(예: 백색왜성 쌍성계)이 추가될 때 기존 3개 모델을 재학습할 필요 없이 4번째 공명기 $\mathcal{R}_{\theta_3}$만 꽂으면 되는 완벽한 모듈형 확장성(Continual Class-Incremental Learning)을 갖춥니다.
