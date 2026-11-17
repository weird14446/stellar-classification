# Deep Learning (DL) Pipeline: Stellar Classification

SDSS-17 천체 데이터셋(`star_classification.csv`)에 대한 기존 역전파(Backpropagation) 기반 딥러닝 및 경사 부스팅(GBDT) 앙상블 파이프라인입니다.

---

## 1. 개요 및 파이프라인 구성

- **데이터 전처리 (`preprocess.py`)**:
  - 오류 센티널 행(u/g/z = -9999) 제거
  - 측광 특성(`u, g, r, i, z`) 및 색지수 파생 특성(`u-g, g-r, r-i, i-z`)에 `RobustScaler` 적용
  - `redshift`에 `log1p` + `StandardScaler` 적용
  - 재현 가능한 64k/16k/20k (train/val/test) 계층화 분할 (Stratified Split)
- **신경망 아키텍처 (`model.py`)**:
  - 은닉층 (64, 32) 구조의 PyTorch MLP
- **앙상블 및 최종 모델 (`train_final.py`)**:
  - PyTorch MLP (3개 시드 앙상블) + XGBoost + HistGradientBoosting 소프트 보팅 앙상블
- **추론 (`predict.py`)**:
  - 저장된 모델 체크포인트를 사용한 단일 파이프라인 배치 추론 및 자가 검증

---

## 2. 주요 파일 목록

| 파일 | 설명 |
|---|---|
| `preprocess.py` | 데이터 정제 및 스케일링 파이프라인 |
| `model.py` | PyTorch 기반 MLP 아키텍처 정의 |
| `train_final.py` | MLP + XGBoost + HistGB 최종 앙상블 훈련 및 저장 |
| `predict.py` | 저장된 모델을 통한 추론 CLI 및 self-check |
| `experiment_torch.py` | PyTorch MLP ablation 실험 (색지수, 클래스 가중치) |
| `experiment_depth.py` | 신경망 은닉층 깊이/폭 비교 실험 |
| `experiment_gbdt.py` | LightGBM, XGBoost, CatBoost 비교 실험 |
| `experiment_ensemble.py` | 모델 앙상블 가중치 최적화 실험 |
| `explore_dataset.py` | SDSS 데이터셋 탐색적 데이터 분석(EDA) |
| `models/` | 학습된 모델 가중치, 전처리기, 메타데이터 |
| `figures/` | 실험 결과 시각화 차트 |

---

## 3. 실행 방법

```bash
# Conda 환경 활성화
conda activate dl_env

# 1. 전처리 데모
python DL/preprocess.py

# 2. 앙상블 모델 훈련
python DL/train_final.py

# 3. 테스트 세트 자가 검증 (Self-Check)
python DL/predict.py

# 4. 새 데이터 추론
python DL/predict.py new_data.csv -o predictions.csv
```

---

## 4. 성능 (Test Split 20,000건)

- **Test Accuracy**: **97.88%**
- **Test Macro-F1**: **0.9755**

