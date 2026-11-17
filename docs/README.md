# Documentation & Research Reports (`docs/`)

본 디렉토리는 **SDSS-17 천체 분류 프로젝트**의 세 가지 독립적 학습 패러다임(**① 역전파 딥러닝 `DL/`**, **② 위상·가중치 진화 신경망 `EDL/TWEANN`**, **③ 무역전파 시공간 중력장 궤도 신경망 `GravNet/`**)과 차세대 대안 모델 설계 총람을 담은 연구 문서 저장소입니다.

---

## 수록 문서 목록

### A. 시공간 중력장 궤도 신경망 (`GravNet`) 연구 보고서 및 학술 논문

1. **[`gravnet_ieee.pdf`](./gravnet_ieee.pdf)** *(IEEE Transactions 저널 형식 최종 조판 논문 — 6쪽)*
   - **문서 클래스**: `\documentclass[journal]{IEEEtran}` (2단 컬럼 학술 논문 규격)
   - **수록 내용**: 영문 초록(Abstract) 및 색인어(Index Terms), I~VII 로마자 섹션 체계, 4단계 물리 추론 및 무-BPTT 조석 순환 TikZ 다이어그램(`figure*`), 다중 헤일로 플러머 퍼텐셜, 심플렉틱 리프프로그 적분기, 리아푸노프 역학적 에너지 단조 소산($\frac{dH}{dt} \le 0$) 및 LaSalle 불변성 정리 증명($\blacksquare$), SymPy 기호/수치 자가 검증 결과표(Table I), 궤도 평면 시뮬레이션·에너지 소산 곡선·라그랑주 $L_1$ 물리적 불확실성 해부(`figure*`), 3대 패러다임 종합 벤치마크 비교표(Table III) 및 10편의 공식 IEEE 참고문헌.
   - **XeLaTeX 컴파일**:
     ```bash
     cd docs && /Library/TeX/texbin/latexmk -xelatex gravnet_ieee.tex
     ```

2. **[`gravnet_ieee.tex`](./gravnet_ieee.tex)** *(IEEEtran XeLaTeX 원본 소스 코드)*

3. **[`gravnet_research_report.pdf`](./gravnet_research_report.pdf)** *(기술 보고서 형식 상세 PDF — 12쪽)*
   - 오차 역전파(Backpropagation/BPTT) 없이 다중 헤일로 플러머 중력장과 심플렉틱 리프프로그 적분만으로 천체를 분류하는 `GravNet`의 개발 전 과정, 시행착오(도플러 주파수 돌파), 상세 구현 코드 분석 및 3대 패러다임 벤치마크(`97.59%`, 항성 재현율 `100.00%`)를 수록한 기술 보고서입니다.

4. **[`gravnet_research_report.tex`](./gravnet_research_report.tex)** *(기술 보고서 LaTeX 원본 소스 — XeLaTeX + `kotex`)*

5. **[`proposed_novel_models.md`](./proposed_novel_models.md)** *(차세대 대안 및 독창적 신개념 딥러닝 패러다임 9종 상세 설계서)*
   - **Part I (학술 비역전파·해석적 모델 5종)**: 전방-전방(`FFDL`), 닫힌 해 극한 학습 기계(`ELM`), 콜모고로프-아놀드망(`KAN`), 에너지 평형 전파/소프트 헤비안(`EP/Hebb`), 베이지안 변분 신경망(`BNN`)
   - **Part II (물리·파동·복잡계 기반 독창적 모델 4종)**: 시공간 중력장 궤도망(`GravNet`, 구현 완료), 복소 파동 간섭 홀로그래피망(`QWIN`), 자가 치유 세포 군집 합의망(`NCA-Jury`), 스펙트럼 공명 역추론망(`ResonanceNet`)

---

### B. 딥러닝(`DL`) vs. 진화 신경망(`EDL/TWEANN`) 대중 리서치 블로그
1. **[`research_blog_stellar_classification.pdf`](./research_blog_stellar_classification.pdf)** *(리서치 블로그 PDF — 13쪽)*
2. **[`research_blog_stellar_classification.tex`](./research_blog_stellar_classification.tex)** *(LaTeX 원본 소스)*
3. **[`research_blog_stellar_classification.md`](./research_blog_stellar_classification.md)** *(웹/GitHub 게재용 Markdown 버전)*
