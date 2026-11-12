# Stellar Classification: DL, Evolutionary TWEANN & Astrophysical GravNet

SDSS-17 (Sloan Digital Sky Survey) 관측 데이터를 활용하여 천체(은하 **GALAXY**, 별 **STAR**, 퀘이사 **QSO**)를 고정밀로 자동 분류하는 멀티 패러다임 머신러닝/딥러닝 연구 저장소입니다.

본 저장소는 서로 완전히 다른 세 가지 학습 원리를 독립적으로 구현하고 동일 데이터 분할 위에서 엄밀하게 벤치마크합니다:
1. **`DL/` (전역 오차 역전파 딥러닝)**: 표준 다층 퍼셉트론(MLP) 및 이종 GBDT(`XGBoost`, `HistGB`) 앙상블
2. **`EDL/` (위상·가중치 진화 신경망, TWEANN)**: Net2Net 함수 보존 변이와 결정 경계 진화 전략을 통한 뇌 구조 자율 발명
3. **`GravNet/` (시공간 중력장 궤도 신경망)**: **100% 무역전파(Zero-Backprop)** 다중 헤일로 플러머 중력장 + 심플렉틱 감쇠 리프프로그 궤도 포획 모델

---

## 1. 프로젝트 아키텍처 개요

```
stellar-classification/
├── DL/                             # [Paradigm 1] 전역 오차 역전파 & 이종 앙상블 파이프라인
│   ├── preprocess.py               # 측광 스케일링 & 색지수 엔지니어링
│   ├── model.py                    # PyTorch MLP (고정 64, 32)
│   ├── train_final.py              # MLP + XGBoost + HistGB 소프트 보팅 앙상블 학습
│   ├── predict.py                  # DL 추론 및 self-check CLI
│   ├── explore_dataset.py          # SDSS 데이터 분포 및 통계 탐색
│   ├── compare_class_distributions.py # 천체 클래스별 측광 분포 비교
│   ├── experiment_depth.py         # 은닉층 깊이/너비 절제 실험
│   ├── experiment_torch.py         # PyTorch 단일/앙상블 실험
│   ├── experiment_ensemble.py      # 소프트/하드 보팅 앙상블 실험
│   ├── experiment_gbdt.py          # XGBoost / HistGB 단독 실험
│   ├── experiment_error_analysis.py # 오분류 샘플 에러 분석
│   ├── plot_style.py               # 과학 시각화 스타일 설정
│   ├── figures/                    # DL 시각화 플롯 (혼동행렬, 특성 중요도 등)
│   └── models/                     # 학습된 PyTorch 및 GBDT 체크포인트
│
├── EDL/                            # [Paradigm 2] 구조+가중치 진화 신경망 (TWEANN)
│   ├── preprocess.py               # DL/GravNet과 동일한 층화 분할 전처리
│   ├── models.py                   # TopologyGenome 및 6종 구조 변이를 갖춘 EvolvableMLP
│   ├── evolution.py                # 다중 섬 구조+가중치 진화(Memetic TWEANN), NES, GA
│   ├── train_edl.py                # 구조 및 가중치 진화 학습 CLI
│   ├── evaluate_edl.py             # 20,000건 테스트 셋 평가
│   ├── plot_evolution.py           # 진화 세대별 적응도·위상 복잡도 시각화
│   ├── predict_edl.py              # EDL 추론 및 self-check CLI
│   ├── figures/                    # 진화 과정 및 앙상블 시각화 플롯
│   └── models/                     # 진화된 최고 엘리트 게놈 및 체크포인트
│
├── GravNet/                        # [Paradigm 3] 무역전파 시공간 중력장 궤도 신경망
│   ├── preprocess.py               # DL/EDL과 동일한 63,999 / 16,000 / 20,000 층화 분할
│   ├── physics_model.py            # 플러머 중력장, 심플렉틱 리프프로그, 조석 가소성, SymPy 검증
│   ├── train_gravnet.py            # 단일 우주 및 3-우주 다중우주(Multiverse) 무역전파 학습
│   ├── evaluate_gravnet.py         # 테스트 정밀 평가 및 라그랑주 L1 안장점 불확실성 분석
│   ├── plot_gravnet.py             # 중력장 등퍼텐셜·궤도, 리아푸노프 소산, 라그랑주 시각화
│   ├── predict_gravnet.py          # 포획 확률 + 라그랑주 장벽 출력 추론 CLI
│   ├── figures/                    # 논문 수록 4대 고해상도 물리 시각화 플롯
│   └── models/                     # 단일 우주 및 3-우주 다중우주 학습 파라미터 (.pt, .npz)
│
├── docs/                           # 연구 보고서 (LaTeX/PDF) 및 차세대 신개념 모델 설계 총람
│   ├── README.md                   # 수록 문서 상세 안내 및 XeLaTeX 컴파일 가이드
│   ├── gravnet_ieee.pdf            # IEEE Transactions 저널 형식 GravNet 정식 논문 (6쪽 PDF)
│   ├── gravnet_ieee.tex            # IEEEtran XeLaTeX 원본 소스 코드
│   ├── gravnet_research_report.pdf # GravNet 개발 과정·수식 유도·벤치마크 상세 보고서 (12쪽 PDF)
│   ├── gravnet_research_report.tex # 기술 보고서 LaTeX 원본 소스 코드
│   ├── proposed_novel_models.md    # 대안적 및 독창적 신개념 딥러닝 모델 9종 상세 설계서
│   ├── research_blog_stellar_classification.pdf # DL vs EDL 대중 리서치 블로그 (13쪽 PDF)
│   ├── research_blog_stellar_classification.tex # 리서치 블로그 LaTeX 원본 소스 코드
│   └── research_blog_stellar_classification.md  # 리서치 블로그 마크다운 버전
│
├── star_classification.csv         # SDSS-17 천체 관측 원본 데이터셋 (100,000행)
├── memory/                         # 날짜별 실험 및 아키텍처 연구 기록
└── README.md                       # 본 문서
```

---

## 2. 3대 학습 패러다임 핵심 비교 (20,000건 Held-out Test Split)

| 구분 | `DL/` (역전파 딥러닝) | `EDL/` (위상·가중치 진화 TWEANN) | `GravNet/` (시공간 중력장 궤도망) |
|---|---|---|---|
| **핵심 추론 원리** | 고정 은닉층 행렬곱 `(64, 32)` | 진화된 스킵 위상 `104-48-48+skip` | **18-헤일로 플러머 중력장 궤도 적분 ($T=10$)** |
| **핵심 학습 원리** | 전역 연쇄법칙 오차 역전파 (Adam) | Net2Net 변이 + 라마르크 + 경계 ES | **100% 무역전파 시공간 조각 + 종착 조석력** |
| **단일 모델 테스트 성능** | Acc **97.37%** / F1 **0.9700** | Acc **97.43%** / F1 **0.9706** | **Acc 97.55% / F1 0.9718** |
| **동종 앙상블 테스트 성능** | Acc **97.42%** / F1 **0.9705** (3-Seed) | Acc **97.45%** / F1 **0.9709** (Top-4 섬) | **Acc 97.59% / F1 0.9724** (3-우주 다중우주) |
| **항성(`STAR`) 재현율** | 99.72% (단일) / 99.77% (3-Seed) | 99.79% (단일) / 99.84% (Top-4 섬) | **99.93% (단일) / 100.00% (4,319/4,319 전수 포획)** |
| **물리적 불확실성 진단** | 별도 온도 보정 필요 | 앙상블 분산 활용 | **라그랑주 $L_1$ 장벽 자동 분리 (깊은 우물 98.71%)** |
| **학습 소요 시간** | 24.16 s (단일) / 72.49 s (3-Seed) | 11.78 s (섬당) / 94.22 s (8개 섬) | **40.69 s (단일, 1단계 7.45s) / 119.83 s (3-우주)** |
| *(참고) 이종 트리 결합* | Acc **97.88%** (`MLP×3 + XGB + HistGB`) | — | — |

---

## 3. 빠른 시작 (Quickstart)

Conda 가상환경 `dl_env`를 활성화하여 실행합니다.

```bash
conda activate dl_env

# 1. 시공간 중력장 궤도 신경망 (GravNet) 물리 검증, 학습, 평가, 시각화 및 추론
python GravNet/physics_model.py
python GravNet/train_gravnet.py --epochs 18 --orbital-steps 10
python GravNet/evaluate_gravnet.py
python GravNet/plot_gravnet.py
python GravNet/predict_gravnet.py

# 2. 위상·가중치 진화 신경망 (EDL / TWEANN) 학습, 평가, 시각화 및 추론
python EDL/train_edl.py --islands 8 --generations 30
python EDL/evaluate_edl.py
python EDL/plot_evolution.py
python EDL/predict_edl.py

# 3. 역전파 딥러닝 (DL) 학습, 평가 및 추론
python DL/train_final.py
python DL/predict.py
```
