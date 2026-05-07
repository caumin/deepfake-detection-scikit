# Deepfake Detection Scikit

## Project Summary

- AI 생성 이미지와 실제 이미지를 구분하는 image authenticity detection 실험
- 딥러닝 모델을 직접 학습하지 않고, 이미지 포렌식 특징을 추출한 뒤 전통 ML 모델로 분류
- 생성 이미지 탐지에서 통계, 주파수, 색상, 노이즈, 텍스처 특징이 주는 신호 확인
- Production detector가 아니라 feature-based detection 실험 프로젝트

## Dataset

- AI-Generated Images vs Real Images
- Source: https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images

## What I Built

- real/fake 이미지 폴더에서 feature CSV를 생성하는 추출 파이프라인
- FFT, color statistics, residual noise, GLCM/Haralick, MSCN, DCT, wavelet 등 포렌식 특징 추출 모듈
- Linear SVM, RBF SVM approximation, RandomForest, XGBoost, Logistic Regression 학습 코드
- accuracy, AUROC, classification report, confusion matrix 기반 평가 코드
- 단일 이미지 또는 폴더 예측용 스크립트
- feature group 성능 비교 및 결과 시각화 유틸리티

## Main Files

- `features.py`
  - 이미지 포렌식 특징 추출 함수 모음
  - spectrum, color, residual, texture, wavelet, phase, NLF feature 포함
- `extract_features.py`
  - `real/`, `fake/` 이미지 폴더에서 feature CSV 생성
  - resize, JPEG re-encoding, 병렬 처리 지원
- `train.py`
  - feature CSV 기반 모델 학습
  - `linsvm`, `rbfsvm`, `rf`, `xgb`, `logreg` 지원
- `eval.py`
  - test CSV 평가
  - accuracy, AUROC, classification report, confusion matrix 저장
  - feature importance, decision boundary, SHAP/permutation importance 시각화 지원
- `predict.py`
  - 저장된 `.joblib` 모델로 이미지 또는 폴더 예측
- `run_pipeline.py`
  - feature extraction, training, evaluation end-to-end 실행
- `probe_features.py`
  - feature group별 성능 비교
- `viz_forensic.py`
  - FFT spectrum, residual, saturation 등 포렌식 관점 시각화
- `analyze_correlation.py`
  - feature correlation heatmap 생성
- `visualize_reports.py`
  - 여러 evaluation report 비교 시각화

## Data Layout

```text
dataset/
  train/
    real/
    fake/
  test/
    real/
    fake/
```

- `train/test` 구조가 있으면 사전 분리 데이터로 처리
- `train/test` 구조가 없으면 전체 `real/`, `fake/` 폴더에서 feature 추출 후 split 수행

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py --data_dir /path/to/dataset --model rf
```

## Run Step By Step

```bash
python extract_features.py --real_dir data/real --fake_dir data/fake --out_csv out/features.csv
python train.py --csv out/features.csv --model rf --out_model out/model.joblib
python eval.py --csv out/test_features.csv --model out/model.joblib --report_dir out/report
python predict.py --model out/model.joblib --input /path/to/image_or_dir
```

## Notes

- `predict.py`의 feature extraction 옵션은 학습 때 사용한 옵션과 맞아야 함
- 성능은 데이터셋, 생성 모델, 압축/리사이즈 방식에 크게 영향받음
- 실험 목적: AI-generated image detection에서 수작업 포렌식 특징의 유효성 확인
