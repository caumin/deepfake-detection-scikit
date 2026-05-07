# Deepfake Detection Scikit

AI 생성 이미지와 실제 이미지를 구분하기 위해 딥러닝 모델을 직접 학습하기보다, 이미지 포렌식 특징을 추출한 뒤 전통적인 머신러닝 모델로 분류하는 실험 프로젝트입니다.

## Overview

이 프로젝트는 이미지의 통계, 주파수, 색상, 노이즈, 텍스처 단서를 feature vector로 만들고, 해당 feature를 기반으로 real/fake 이진 분류 모델을 학습합니다. 생성 이미지 탐지 문제에서 어떤 수작업 특징이 신호를 줄 수 있는지 빠르게 실험하는 데 초점을 둡니다.

## Main Files

- `features.py`: FFT spectrum, color statistics, residual noise, GLCM/Haralick, MSCN, DCT, wavelet, phase, NLF 등 포렌식 특징 추출
- `extract_features.py`: `real/` 및 `fake/` 이미지 폴더에서 feature CSV 생성
- `train.py`: Linear SVM, RBF SVM approximation, RandomForest, XGBoost, Logistic Regression 학습
- `eval.py`: accuracy, AUROC, classification report, confusion matrix, feature importance 시각화 생성
- `predict.py`: 저장된 `.joblib` 모델로 단일 이미지 또는 폴더 예측
- `run_pipeline.py`: feature extraction, training, evaluation을 end-to-end로 실행
- `probe_features.py`: feature group별 성능 비교
- `viz_forensic.py`, `analyze_correlation.py`, `visualize_reports.py`: 포렌식 특징 및 결과 분석용 시각화 도구

## Data Layout

사전 분리된 데이터셋은 아래 구조를 권장합니다.

```text
dataset/
  train/
    real/
    fake/
  test/
    real/
    fake/
```

`train/test`가 없는 경우 `real/`, `fake/` 폴더 전체에서 feature를 추출한 뒤 train/test split을 수행합니다.

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py --data_dir /path/to/dataset --model rf
```

개별 단계로 실행할 수도 있습니다.

```bash
python extract_features.py --real_dir data/real --fake_dir data/fake --out_csv out/features.csv
python train.py --csv out/features.csv --model rf --out_model out/model.joblib
python eval.py --csv out/test_features.csv --model out/model.joblib --report_dir out/report
python predict.py --model out/model.joblib --input /path/to/image_or_dir
```

## Notes

- `predict.py`의 feature extraction 옵션은 학습 때 사용한 옵션과 맞아야 합니다.
- 성능은 데이터셋, 생성 모델, 압축/리사이즈 방식에 크게 영향을 받습니다.
- 이 프로젝트는 production detector가 아니라 AI-generated image detection을 이해하기 위한 실험 코드입니다.
