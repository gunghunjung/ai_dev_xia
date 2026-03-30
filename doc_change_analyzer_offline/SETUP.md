# 오프라인 문서 변경 분석 AI — 설치 및 실행 가이드

> 아나콘다(Anaconda) 환경 기준 / 인터넷 없는 폐쇄망 환경 완전 지원

---

## 목차

1. [폴더 구조](#1-폴더-구조)
2. [환경 구성 (온라인 PC에서 먼저 준비)](#2-환경-구성-온라인-pc에서-먼저-준비)
3. [오프라인 PC로 이전](#3-오프라인-pc로-이전)
4. [로컬 모델 다운로드](#4-로컬-모델-다운로드)
5. [실행 방법 (CLI)](#5-실행-방법-cli)
6. [엑셀 데이터 형식](#6-엑셀-데이터-형식)
7. [컬럼 매핑 예시](#7-컬럼-매핑-예시)
8. [오류 해결](#8-오류-해결)
9. [성능 팁](#9-성능-팁)

---

## 1. 폴더 구조

```
doc_change_analyzer_offline/
│
├── main.py           ← CLI 진입점 (여기서 실행)
├── train.py          ← 학습 로직
├── predict.py        ← 예측 로직
├── model.py          ← 멀티태스크 모델 (T5 / KoBART)
├── model_loader.py   ← 저장/로드 관리
├── utils.py          ← 데이터 전처리 유틸
│
├── requirements.txt  ← 패키지 목록
├── SETUP.md          ← 이 파일
│
├── data/             ← 학습/예측용 엑셀 파일 여기에 넣기
├── local_model/      ← 사전 학습 모델 (HuggingFace 로컬 파일)
│   ├── kobart/       ← KoBART 모델 파일
│   └── t5/           ← T5 모델 파일
├── saved_model/      ← 학습 완료 모델 저장됨
└── predictions/      ← 예측 결과 xlsx 저장됨
```

---

## 2. 환경 구성 (온라인 PC에서 먼저 준비)

### 2-1. 아나콘다 가상환경 생성

```bash
conda create -n doc_ai python=3.10 -y
conda activate doc_ai
```

### 2-2. PyTorch 설치

**CPU 전용 (GPU 없음 — 권장):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 11.8):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### 2-3. 나머지 패키지 설치

```bash
pip install transformers>=4.35.0 tokenizers sentencepiece protobuf
pip install pandas numpy openpyxl scikit-learn tqdm
```

또는 한번에:
```bash
pip install -r requirements.txt
```

### 2-4. 오프라인 이전용 패키지 whl 파일 저장

```bash
# 오프라인 환경 이전을 위해 whl 파일 다운로드
pip download -r requirements.txt -d ./wheels/
# torch는 별도 (용량이 큼)
pip download torch --index-url https://download.pytorch.org/whl/cpu -d ./wheels/
```

---

## 3. 오프라인 PC로 이전

### 3-1. 복사할 항목

| 복사 대상 | 설명 |
|-----------|------|
| `doc_change_analyzer_offline/` 폴더 전체 | 소스 코드 |
| `wheels/` 폴더 | pip 설치용 whl 파일 |
| `local_model/kobart/` 또는 `local_model/t5/` | 사전학습 모델 |
| 아나콘다 설치 파일 (Anaconda3-*.exe) | 미설치 시 |

### 3-2. 오프라인 PC에서 환경 구성

```bash
# 아나콘다 설치 후
conda create -n doc_ai python=3.10 -y
conda activate doc_ai

# 오프라인 pip 설치 (인터넷 없이)
pip install --no-index --find-links=./wheels/ -r requirements.txt
```

---

## 4. 로컬 모델 다운로드

### 방법 A: Python 스크립트로 다운로드 (온라인 PC에서)

```python
# download_model.py — 온라인 PC에서 실행 후 폴더째 복사
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# KoBART (한국어 BART)
model_name = "gogamza/kobart-base-v2"
save_path  = "./local_model/kobart"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"저장 완료: {save_path}")
```

```python
# T5 (한국어 T5)
model_name = "paust/pko-t5-base"
save_path  = "./local_model/t5"

from transformers import AutoTokenizer, T5ForConditionalGeneration
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"저장 완료: {save_path}")
```

### 방법 B: git lfs로 직접 다운로드

```bash
# git lfs 필요
git lfs install
git clone https://huggingface.co/gogamza/kobart-base-v2 ./local_model/kobart
```

### 방법 C: HuggingFace 웹사이트에서 수동 다운로드

1. https://huggingface.co/gogamza/kobart-base-v2 접속
2. "Files and versions" 탭 → 모든 파일 다운로드
3. `./local_model/kobart/` 폴더에 저장

**필수 파일 목록 (kobart 예시):**
```
local_model/kobart/
  config.json
  tokenizer_config.json
  tokenizer.json
  vocab.txt  (또는 sentencepiece.bpe.model)
  pytorch_model.bin  (또는 model.safetensors)
```

---

## 5. 실행 방법 (CLI)

> 모든 명령은 `doc_change_analyzer_offline/` 폴더 안에서 실행

```bash
cd doc_change_analyzer_offline
conda activate doc_ai
```

### 5-1. 신규 학습 (단일 파일)

```bash
python main.py train \
  --input_file ./data/sample.xlsx \
  --before_col 변경전 \
  --after_col 변경후 \
  --summary_col 요약 \
  --reason_col 사유 \
  --code_col 코드 \
  --model_path ./local_model/kobart \
  --save_path ./saved_model/v1 \
  --epochs 10 \
  --batch_size 4
```

### 5-2. 신규 학습 (다중 파일 — 디렉터리 자동 통합)

```bash
python main.py train \
  --input_dir ./data \
  --before_col 변경전 \
  --after_col 변경후 \
  --summary_col 요약 \
  --reason_col 사유 \
  --code_col 코드 \
  --model_path ./local_model/kobart \
  --save_path ./saved_model/v1 \
  --save_merged
```

### 5-3. 증분 학습 (기존 모델에 새 데이터 추가)

```bash
python main.py train \
  --input_file ./data/new_data.xlsx \
  --before_col 변경전 --after_col 변경후 \
  --summary_col 요약 --reason_col 사유 --code_col 코드 \
  --model_path ./saved_model/v1 \
  --save_path ./saved_model/v1 \
  --incremental \
  --epochs 5 \
  --lr 1e-5
```

### 5-4. 예측 (단일 파일)

```bash
python main.py predict \
  --model_path ./saved_model/v1 \
  --input_file ./data/new.xlsx \
  --output_file ./predictions/result.xlsx
```

### 5-5. 예측 (다중 파일 — 파일별 저장)

```bash
python main.py predict \
  --model_path ./saved_model/v1 \
  --input_dir ./data/predict \
  --output_mode separate \
  --output_dir ./predictions
```

### 5-6. 예측 (다중 파일 — 통합 저장)

```bash
python main.py predict \
  --model_path ./saved_model/v1 \
  --input_dir ./data/predict \
  --output_mode combined \
  --output_file ./predictions/combined_result.xlsx
```

### 5-7. 저장된 모델 목록 조회

```bash
python main.py list --models_root ./saved_model
```

### 5-8. 모델 상세 정보 출력

```bash
python main.py info --model_path ./saved_model/v1
```

---

## 6. 엑셀 데이터 형식

### 학습용 엑셀 (필수 컬럼)

| 컬럼명 | 설명 | 필수 여부 |
|--------|------|-----------|
| 변경전 | 변경 전 원문 텍스트 | **필수** |
| 변경후 | 변경 후 수정 텍스트 | **필수** |
| 코드   | 변경 유형 코드 (예: A01, B02) | **필수** |
| 요약   | 변경 내용 요약 | 선택 (없으면 빈 값) |
| 사유   | 변경 사유/이유 | 선택 (없으면 빈 값) |

### 예시

| 변경전 | 변경후 | 요약 | 사유 | 코드 |
|--------|--------|------|------|------|
| 제1조 계약기간은 1년으로 한다 | 제1조 계약기간은 2년으로 한다 | 계약기간 연장 | 고객 요청 | A01 |
| 위약금은 계약금의 10%로 한다 | 위약금은 계약금의 20%로 한다 | 위약금 인상 | 법규 개정 | B02 |

### 예측용 엑셀 (최소 컬럼)

예측 시에는 **변경전 + 변경후** 컬럼만 있으면 됩니다.
(config.json의 컬럼 매핑을 자동으로 읽어 사용)

---

## 7. 컬럼 매핑 예시

파일마다 컬럼명이 다른 경우 `--before_col`, `--after_col` 등으로 직접 지정:

```bash
# 영문 컬럼명 파일
python main.py train \
  --input_file ./data/english_cols.xlsx \
  --before_col before_text \
  --after_col after_text \
  --summary_col summary \
  --reason_col reason \
  --code_col change_code \
  --model_path ./local_model/t5 \
  --save_path ./saved_model/v2
```

---

## 8. 오류 해결

### ❌ `[오프라인 모델 없음]` 오류

```
OSError: [오프라인 모델 없음] './local_model/kobart'
```

**해결:** `local_model/` 폴더에 모델 파일이 없습니다. [4. 로컬 모델 다운로드](#4-로컬-모델-다운로드)를 참고하세요.

---

### ❌ `컬럼 없음` 오류

```
ValueError: 컬럼 없음: ['변경전']
```

**해결:** 엑셀 파일의 실제 컬럼명을 확인하고 `--before_col`, `--after_col` 인자를 맞게 수정하세요.

```bash
# 컬럼 확인 (Python)
python -c "import pandas as pd; print(pd.read_excel('./data/sample.xlsx').columns.tolist())"
```

---

### ❌ `config.json 없음` 오류

```
FileNotFoundError: config.json 없음: ./saved_model/v1/config.json
```

**해결:** 학습이 완료되지 않은 경로입니다. `python main.py train ...`을 먼저 실행하세요.

---

### ❌ 메모리 부족 (OOM)

**해결:**
```bash
# batch_size 줄이기
python main.py train ... --batch_size 2

# 입력 토큰 줄이기
python main.py train ... --max_input_len 128 --max_target_len 64
```

---

### ❌ `sentencepiece` 오류 (T5 토크나이저)

```bash
pip install sentencepiece protobuf
```

---

### ❌ `openpyxl` 오류 (엑셀 읽기)

```bash
pip install openpyxl
```

---

## 9. 성능 팁

### CPU 환경 최적화

| 설정 | 권장값 | 이유 |
|------|--------|------|
| `--batch_size` | 2 ~ 4 | CPU 메모리 절약 |
| `--max_input_len` | 128 ~ 256 | 속도 향상 |
| `--max_target_len` | 64 ~ 128 | 생성 속도 향상 |
| `--num_beams` | 1 ~ 2 | 예측 속도 향상 (품질↓) |
| `--epochs` | 3 ~ 5 | 과적합 방지 + 속도 |

### 학습 시간 예측 (CPU 기준)

| 데이터 크기 | 예상 시간/에폭 |
|-------------|----------------|
| 100건 | 약 2~5분 |
| 500건 | 약 10~30분 |
| 1,000건 | 약 20~60분 |
| 5,000건+ | GPU 권장 |

### 학습 결과물 구조

```
saved_model/v1/
  config.json          ← 컬럼 매핑 + 학습 메타 (예측 시 자동 참조)
  meta.json            ← 모델 타입 + 클래스 수
  label_encoder.pkl    ← 코드 클래스 인코더
  classifier.pt        ← 분류 헤드 가중치
  config.json          ← HuggingFace 모델 config
  pytorch_model.bin    ← 모델 가중치 (또는 model.safetensors)
  tokenizer_config.json
  tokenizer.json
  vocab.txt / spiece.model
```

---

## 빠른 시작 (Quick Start)

```bash
# 1. 환경 활성화
conda activate doc_ai

# 2. 프로젝트 폴더로 이동
cd doc_change_analyzer_offline

# 3. 학습 (data/ 폴더에 엑셀 파일 준비 후)
python main.py train \
  --input_file ./data/my_data.xlsx \
  --before_col 변경전 --after_col 변경후 \
  --summary_col 요약 --reason_col 사유 --code_col 코드 \
  --model_path ./local_model/kobart \
  --save_path ./saved_model/v1

# 4. 예측
python main.py predict \
  --model_path ./saved_model/v1 \
  --input_file ./data/new.xlsx \
  --output_file ./predictions/result.xlsx

# 5. 결과 확인
# predictions/result.xlsx 파일 열기
# pred_summary, pred_reason, pred_code, confidence_score 컬럼 확인
```

---

*최종 수정: 2026-03-27*
