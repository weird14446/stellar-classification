# Drift Remediation Report: Design Documents vs. Codebase Implementation

* **최초 점검 일시**: 2026-11-12 20:11:59 (KST)
* **조치 및 수정 완료 일시**: 2026-11-12 21:05:00 (KST)
* **상태**: **모든 불일치(Drift) 해소 완료 (100% Reconciled)**

---

## 1. 종합 요약 (Executive Summary)

본 리포트는 저장소 내 설계 문서(Documentation)와 실제 코드베이스 구현체(Codebase Implementation) 간의 정합성을 교차 검증하고, 식별된 모든 불일치 및 누락 사항을 완벽하게 수정·반영한 내역을 기록합니다.

| 검토 대상 문서 | 최초 상태 | 조치 내용 | 최종 상태 |
|---|---|---|---|
| [`README.md`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/README.md) | 불완전한 트리 & 비대칭 Quickstart | `DL/`, `EDL/`, `docs/` 전체 파일 인덱싱 및 3대 패러다임 대칭 실행 명령어 반영 | **[RESOLVED] 일치** |
| [`docs/proposed_novel_models.md`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/proposed_novel_models.md) | 구현 스펙 누락 & 링크 부재 | 3,826차원 렌즈, $K=6$, 3단계 물리 학습, $L_1$ 안장점 스펙 추가 및 소스코드 링크 연결 | **[RESOLVED] 일치** |
| [`docs/research_blog_stellar_classification.md`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/research_blog_stellar_classification.md) | 3패러다임 누락 & 모델 링크 부재 | 서두 제3패러다임 연계 안내문 추가, `EDL/models/` 파일 링크 추가, Quickstart에 GravNet 반영 | **[RESOLVED] 일치** |
| [`docs/research_blog_stellar_classification.tex`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/research_blog_stellar_classification.tex) | 3패러다임 누락 & 모델 링크 부재 | `insightbox` 안내문 추가, `EDL/models/` 파일 명시, `codebox` GravNet 추가 및 PDF 재컴파일 완료 | **[RESOLVED] 일치** |
| [`docs/README.md`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/README.md) | 최신 정합 유지 | 신규 IEEE 저널 논문(`gravnet_ieee.pdf`, `.tex`) 및 전체 수록 문서 인덱싱 완비 | **[NO DRIFT] 일치** |
| [`docs/gravnet_research_report.tex`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/gravnet_research_report.tex) | 최신 정합 유지 | 코드베이스 수식, SymPy 기호 검증, 벤치마크 수치와 100% 일치 | **[NO DRIFT] 일치** |
| [`docs/gravnet_ieee.tex`](file:///Users/yiinsu/Library/CloudStorage/GoogleDrive-ii3289857@gmail.com/My%20Drive/Github/stellar-classification/docs/gravnet_ieee.tex) | 최신 정합 유지 | IEEE 저널 규격 6쪽 조판 완비, LaSalle 불변성 증명 및 코드베이스와 100% 일치 | **[NO DRIFT] 일치** |

---

## 2. 문서별 상세 불일치 점검 및 수정 조치 내역

### 1) `README.md`
* **Incomplete Documentation Tree Index**:
  * *기존 문제*: `docs/` 트리에 PDF 4종만 표시되고 `README.md`, LaTeX 소스(`.tex`), 마크다운(`.md`)이 누락됨.
  * *수정 조치*: `docs/` 하위의 모든 산출물(`README.md`, `gravnet_ieee.pdf`, `gravnet_ieee.tex`, `gravnet_research_report.pdf`, `gravnet_research_report.tex`, `proposed_novel_models.md`, `research_blog_stellar_classification.pdf`, `.tex`, `.md`)을 명시하여 완벽히 일치시킴.
* **Incomplete `DL/` Pipeline File Tree**:
  * *기존 문제*: `explore_dataset.py`, `compare_class_distributions.py`, `experiment_depth.py`, `experiment_torch.py`, `experiment_ensemble.py`, `experiment_gbdt.py`, `experiment_error_analysis.py`, `plot_style.py` 등 탐색·실험 스크립트 및 `figures/`, `models/`가 트리에 누락됨.
  * *수정 조치*: `DL/` 파이프라인의 12개 파일 및 하위 디렉토리를 트리에 전수 반영함.
* **Incomplete `EDL/` Pipeline File Tree**:
  * *기존 문제*: `EDL/plot_evolution.py`, `figures/`, `models/`가 트리에 누락됨.
  * *수정 조치*: `EDL/` 파이프라인의 진화 시각화 스크립트 및 모델 아티팩트 경로를 전수 반영함.
* **Asymmetric Quickstart Workflow**:
  * *기존 문제*: `GravNet/`만 전체 워크플로우(검증·학습·평가·플롯·추론)가 기재되고, `DL/`과 `EDL/`은 단일 추론 스크립트만 기재되어 비대칭적이었음.
  * *수정 조치*: `DL/train_final.py`, `EDL/train_edl.py`, `EDL/evaluate_edl.py`, `EDL/plot_evolution.py`를 추가하여 3대 패러다임 모두 동일한 학습·평가·추론 완결 워크플로우 명령어를 제공하도록 수정함.

---

### 2) `docs/proposed_novel_models.md`
* **Unimplemented Candidate Architectures (성격 규정)**:
  * *기존 문제*: 8개 후보 아키텍처(Part I 5종, Part II 3종)와 1개 구현체(`GravNet`)의 구현 상태 구분이 불명확했음.
  * *수정 조치*: Part I(5종) 및 Part II-2~4(3종)는 '향후 연구를 위한 차세대 이론적 후보 설계'로 명확히 규정하고, Part II-1 `GravNet`은 '저장소 내 완전 구현 및 검증 완료 모델'로 상태 구분을 체계화함.
* **Missing Architectural Specification (`GravNet`)**:
  * *기존 문제*: 코드베이스(`GravNet/physics_model.py`)의 3,826차원 다중 해상도 광학 렌즈(1D 도플러 파동 $\omega \in [0.25, 120.0]$, SED 2차 곡률, Bochner RFF 2,880개) 및 우주 원점 발사 구속($[\mathbf{W}_q]_{:, 1:2} = \mathbf{0}$)이 추상적 $(\mathbf{q}(0), \mathbf{p}(0))$로만 기재됨.
  * *수정 조치*: $\Phi(\mathbf{x}) \in \mathbb{R}^{3826}$의 엄밀한 수학적 정의와 초기 발사 조건을 수식으로 추가 명시함.
* **Missing Stage 3 Training Specification (`GravNet`)**:
  * *기존 문제*: 1단계(Cholesky), 2단계(조석 헤비안)만 기술되고 3단계 대정준 화학 퍼텐셜 보정 및 라그랑주 $L_1$ 안장점 분석이 누락됨.
  * *수정 조치*: 검증 세트 Macro-$F_1$을 극대화하는 무미분 화학 퍼텐셜 $\boldsymbol{\nu}$ 보정과 라그랑주 $L_1$ 퍼텐셜 장벽 $\Delta E_{L_1}$ 수식을 정식화하여 추가함.
* **Unspecified Sub-halo Count ($K$)**:
  * *기존 문제*: 서브헤일로 개수가 추상 변수 $K$로만 표기됨.
  * *수정 조치*: 클래스당 $K=6$개(총 $3 \times 6 = 18$개 플러머 중력 핵)의 실제 구현 파라미터를 명시함.
* **Missing Direct Source Code Links**:
  * *수정 조치*: `GravNet/physics_model.py`, `GravNet/train_gravnet.py`, `GravNet/evaluate_gravnet.py`, `GravNet/predict_gravnet.py`, `docs/gravnet_ieee.pdf`, `docs/gravnet_research_report.pdf`로 이어지는 상대/절대 파일 링크를 전면 추가함.

---

### 3) `docs/research_blog_stellar_classification.md` & `.tex`
* **Scope Drift (Third Paradigm Omission)**:
  * *기존 문제*: 블로그 작성 시점(1단계) 특성상 2개 패러다임(`DL` vs `EDL`)만 다루고 있어, 저장소의 제3 패러다임(`GravNet`)이 배제된 것처럼 읽힐 수 있음.
  * *수정 조치*: 블로그 서두 콜아웃 영역(`insightbox` / alert)에 본 글이 1단계 연구 성과(DL vs EDL)를 다룸을 안내하고, 제3 패러다임인 `GravNet`의 공식 IEEE 논문(`docs/gravnet_ieee.pdf`)과 기술 보고서(`docs/gravnet_research_report.pdf`)로 연결되는 안내 배너를 추가함.
* **Missing Direct Links to Evolved Topologies**:
  * *기존 문제*: Section 6.3에 8개 진화 섬 구조는 열거되었으나 실제 저장된 가중치 및 체크포인트 파일 경로가 누락됨.
  * *수정 조치*: `EDL/models/` 내의 `edl_model.pt`, `edl_hof_elites.pt`, `edl_weights.npy`, `metadata.json`, `history.json`에 대한 직접 링크를 추가함.
* **Reproducibility Guide Omission**:
  * *기존 문제*: 재현 CLI 가이드에 `GravNet` 실행 명령어가 누락됨.
  * *수정 조치*: Section 8 재현 가이드에 `GravNet/physics_model.py`, `train_gravnet.py`, `evaluate_gravnet.py`, `predict_gravnet.py` 명령어를 추가함.
  * *컴파일 검증*: `docs/research_blog_stellar_classification.tex`를 XeLaTeX으로 재컴파일하여 `research_blog_stellar_classification.pdf`(13쪽)를 오류 없이 최신화함.

---

## 3. 결론

코드베이스의 실제 구현 파일, 파라미터($K=6, M=3826$), 3단계 물리 학습 메커니즘, 체크포인트 산출물, 재현 CLI 워크플로우가 설계 문서(`README.md`, `proposed_novel_models.md`, `research_blog_stellar_classification.md/.tex`, `gravnet_ieee.tex`, `gravnet_research_report.tex`)와 **완벽하게 1:1로 일치**함을 확인하였습니다.
