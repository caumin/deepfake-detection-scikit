# AI-Generated Image Detector

This project provides a lightweight machine learning-based detector to distinguish between real and AI-generated images. It avoids deep learning models and instead relies on traditional ML classifiers trained on statistical and frequency-domain features.

This approach is based on the idea that AI-generated images often contain subtle artifacts from the generation process (e.g., upsampling), which can be captured by analyzing features in the frequency domain, color space, and noise patterns.

---

## (KOR)

## AI 생성 이미지 탐지기

본 프로젝트는 AI 생성 이미지와 실제 이미지를 구분하는 경량 머신러닝 탐지기입니다. 딥러닝 모델을 사용하지 않고, 통계 및 주파수 기반 특징을 추출하여 전통적인 머신러닝 분류기를 학습시키는 방식을 사용합니다.

이 접근법은 AI 생성 이미지가 생성 과정(예: 업샘플링)에서 발생하는 미묘한 아티팩트를 포함하는 경우가 많으며, 이러한 아티팩트는 주파수 도메인, 색상 공간, 노이즈 패턴 분석을 통해 포착될 수 있다는 아이디어에 기반합니다.

### 핵심 특징

본 탐지기는 5가지 주요 통계적 특징 그룹을 추출하여 이미지를 분석합니다.

1.  **1D 파워 스펙트럼 (1D Power Spectrum)**: 이미지의 2D 푸리에 변환 후 방위각 적분(azimuthal integration)을 통해 얻은 1D 파워 스펙트럼은 생성 모델 특유의 주기적인 패턴을 탐지하는 데 효과적입니다.
2.  **스펙트럼 왜곡 (Spectral Distortions)**: 업샘플링 과정에서 발생하는 스펙트럼의 왜곡을 포착하기 위해 특정 주파수 대역의 에너지 비율 등을 계산합니다.
3.  **색상 단서 (Color Cues)**: 실제 카메라의 이미지 처리 파이프라인(ISP)과 생성 모델의 색상 처리 방식의 차이를 활용합니다. HSV 색 공간에서의 채도(Saturation) 분포, RGB 채널 간 상관 계수 등을 특징으로 사용합니다.
4.  **노이즈 잔여물 (Noise Residuals)**: 이미지에서 노이즈 제거 필터를 적용한 후 남는 잔여물(residual)을 분석합니다. 실제 이미지의 센서 노이즈와 생성된 노이즈 패턴의 통계적 차이를 활용합니다.
5.  **GLCM (Gray-Level Co-occurrence Matrix)**: 이미지의 텍스처(질감)를 분석하기 위해 GLCM을 계산하고, 이로부터 대조(Contrast), 동질성(Homogeneity), 에너지(Energy) 등 Haralick 특징을 추출합니다.

### 프로젝트 구조

```
deepdect/
├── data/                               # 데이터셋 루트 폴더
│   └── <dataset_name>/
│       ├── train/
│       │   ├── real/
│       │   └── fake/
│       └── test/
│           ├── real/
│           └── fake/
├── <dataset_name>_output/              # 파이프라인 실행 시 생성되는 결과물 폴더 (예: CIFAKE_output/)
│   ├── <dataset_name>_train_features.csv
│   ├── <dataset_name>_test_features.csv
│   ├── <model_name>.joblib             # 학습된 모델 (예: CIFAKE_rf.joblib)
│   └── report_<model_name>/            # 모델 평가 리포트 폴더
│       ├── report.json
│       └── confusion_matrix.png
├── out/                                # 기타 출력 폴더 (예: 상관관계 분석 결과)
├── viz/                                # 시각화 스크립트에서 생성되는 이미지 샘플
├── aigen-vs-real_output/               # 'AI-Generated Images vs Real Images' 데이터셋 관련 출력
├── CIFAKE_output/                      # 'CIFAKE' 데이터셋 관련 출력
├── gemini_output/                      # 'Gemini' 데이터셋 관련 출력
├── analyze_correlation.py              # 특징 간 상관관계 분석 및 시각화
├── eval.py                             # 학습된 모델을 평가
├── extract_features.py                 # 이미지에서 특징을 추출하고 CSV로 저장
├── features.py                         # 특징 추출 함수 모음
├── gemini.md                           # 프로젝트 초기 설계 및 가이드 문서
├── predict.py                          # 단일 이미지 또는 폴더에 대해 예측 수행
├── probe_features.py                   # 특정 특징 그룹의 성능을 탐색
├── README.md                           # 이 문서
├── requirements.txt                    # Python 의존성 목록
├── run_pipeline.py                     # 전체 파이프라인(특징 추출, 학습, 평가)을 실행
├── skeleton.py                         # (참고용) 기본 스크립트 구조 또는 유틸리티
├── train.py                            # 특징 CSV 파일을 이용해 모델을 학습
├── visualize_reports.py                # 학습 및 평가 리포트를 시각화
├── viz_forensic.py                     # 이미지 포렌식 관점의 시각화 도구 (FFT, Residual 등)
└── ...
```

### 설치 및 준비

1.  **저장소 복제:**
    ```bash
    git clone <repository-url>
    cd deepdect
    ```

2.  **가상 환경 생성 및 활성화:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **의존성 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **데이터 준비:**
    `data` 폴더 아래에 데이터셋을 준비합니다. `run_pipeline.py`는 아래와 같은 구조를 예상합니다.
    - `data/<dataset_name>/train/real/`
    - `data/<dataset_name>/train/fake/`
    - `data/<dataset_name>/test/real/`
    - `data/<dataset_name>/test/fake/`

### 데이터셋

이 프로젝트는 다음 데이터셋을 사용하여 테스트되었습니다. `data` 폴더에 다운로드하여 준비할 수 있습니다.

- **CIFAKE: Real and AI-Generated Synthetic Images**: [Kaggle Link](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **AI-Generated Images vs Real Images**: [Kaggle Link](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images)

### 사용법

#### 전체 파이프라인 실행

`run_pipeline.py` 스크립트는 지정된 데이터셋에 대해 **특징 추출, 여러 모델 학습, 평가**까지의 전체 과정을 자동화합니다.

```bash
python run_pipeline.py --data_dir data/CIFAKE
```

- 이 명령은 `data/CIFAKE` 데이터셋을 사용하여, `CIFAKE_output` 폴더에 특징, 학습된 모델, 평가 리포트를 저장합니다. 기본적으로, 이미 추출된 특징이나 학습된 모델이 있으면 해당 단계를 건너뛰어 시간을 절약합니다.

##### 파이프라인 옵션

- `--feature-importance`: 지원되는 모델(`rf`, `xgb`)에 대해 각 특징의 중요도를 분석한 그래프(`feature_importance.png`)를 리포트 폴더에 함께 저장합니다.
- `--force-extract`: 특징 파일(`_features.csv`)이 이미 존재하더라도 강제로 특징을 다시 추출합니다.
- `--force-train`: 모델 파일(`.joblib`)이 이미 존재하더라도 강제로 모델을 다시 학습합니다.
- `--residual_mode`: 노이즈 잔차 계산 방식을 선택합니다 ('denoise': 정교하고 느림, 'highpass': 빠름).

**예시 (빠른 특징 추출 모드 사용):**
```bash
python run_pipeline.py --data_dir data/CIFAKE --residual_mode highpass
```

#### 개별 스크립트 실행

**1. 특징 추출 (`extract_features.py`)**

- 이 스크립트는 멀티코어 병렬 처리를 사용하여 빠르게 특징을 추출합니다.
- 주요 옵션:
    - `--n_jobs`: 사용할 CPU 코어 수를 지정합니다 (-1은 전체 코어 사용).
    - `--residual_mode`: 노이즈 잔차 계산 방식을 선택합니다 ('denoise', 'highpass').

```bash
python extract_features.py --real_dir data/CIFAKE/train/real --fake_dir data/CIFAKE/train/fake --out_csv CIFAKE_output/train_features.csv --residual_mode highpass
```

**2. 모델 학습 (`train.py`)**

```bash
python train.py --csv CIFAKE_output/train_features.csv --model rf --out_model CIFAKE_output/model_rf.joblib
```

**3. 모델 평가 (`eval.py`)**

```bash
python eval.py --csv CIFAKE_output/test_features.csv --model CIFAKE_output/model_rf.joblib --report_dir CIFAKE_output/report_rf
```

**4. 예측 (`predict.py`)**

```bash
python predict.py --model CIFAKE_output/model_rf.joblib --input /path/to/your/image.jpg
```

#### 유틸리티 스크립트

프로젝트에는 개발 및 분석을 돕기 위한 여러 유틸리티 스크립트가 포함되어 있습니다.

-   **`analyze_correlation.py`**: 추출된 특징들 간의 상관관계를 분석하고 히트맵 등으로 시각화하여 특징의 독립성 및 중요도를 파악하는 데 도움을 줍니다.
-   **`probe_features.py`**: 특정 특징 그룹(예: 스펙트럼 특징, 색상 특징 등)만을 사용하여 모델을 학습하고 평가함으로써, 각 특징 그룹의 탐지 성능 기여도를 개별적으로 분석할 수 있습니다.
-   **`visualize_reports.py`**: `eval.py`에서 생성된 JSON 형식의 리포트 파일들을 읽어와 그래프(예: ROC 곡선 비교, 여러 모델의 성능 지표 비교)로 시각화합니다.
-   **`viz_forensic.py`**: 이미지 자체의 특성(예: 푸리에 스펙트럼, 노이즈 잔여물, GLCM 텍스처 등)을 시각적으로 탐색하고 비교하기 위한 도구입니다. 리얼 및 생성 이미지 간의 시각적 차이를 이해하는 데 유용합니다.
