"""
model_loader.py ─ 아티팩트 중앙 관리 모듈

역할:
  · 모델 + 토크나이저 + 레이블인코더 + config 저장/로드
  · 저장된 모델 목록 조회 및 버전 관리
  · config.json에 컬럼 매핑, 학습 파라미터 등 모든 메타데이터 보관

config.json 구조:
  {
    "version"        : "1.0",
    "created_at"     : "2024-01-01T12:00:00",
    "model_type"     : "t5" | "bart",
    "base_model_path": "./local_model/t5",
    "num_classes"    : 5,
    "label_classes"  : ["A01", "B02", ...],
    "column_mapping" : {"before":"변경전", "after":"변경후", ...},
    "train_params"   : {"epochs":10, "lr":3e-5, ...},
    "data_stats"     : {"total_rows":1000, "train_rows":800, ...}
  }
"""

import os
import json
import pickle
import logging
import datetime
import torch
from typing import Optional, Tuple, Dict, List

from model import MultiTaskT5Model, MultiTaskBartModel, build_model

logger = logging.getLogger(__name__)

# config 파일명 (고정)
CONFIG_FILE = "config.json"
LABEL_ENC_FILE = "label_encoder.pkl"
META_FILE = "meta.json"


# ══════════════════════════════════════════════════════════════
# config 저장 / 로드
# ══════════════════════════════════════════════════════════════

def save_config(config: Dict, save_dir: str):
    """config dict를 save_dir/config.json에 저장"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, CONFIG_FILE)
    # datetime 직렬화 처리
    config_copy = config.copy()
    if "created_at" not in config_copy:
        config_copy["created_at"] = datetime.datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_copy, f, ensure_ascii=False, indent=2)
    logger.info(f"config 저장: {path}")


def load_config(model_dir: str) -> Dict:
    """save_dir/config.json 로드"""
    path = os.path.join(model_dir, CONFIG_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config.json 없음: {path}\n"
            "  → 'python main.py train ...' 을 먼저 실행하세요."
        )
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    logger.info(f"config 로드: {path}")
    return config


# ══════════════════════════════════════════════════════════════
# 아티팩트 저장
# ══════════════════════════════════════════════════════════════

def save_artifacts(
    model,
    tokenizer,
    label_encoder,
    config: Dict,
    save_dir: str,
):
    """
    모델, 토크나이저, 레이블인코더, config를 save_dir에 일괄 저장.
    model.save() 내부에서 meta.json도 생성됨.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 모델 + meta.json
    model.save(save_dir)

    # 2. 토크나이저
    tokenizer.save_pretrained(save_dir)

    # 3. 레이블 인코더
    le_path = os.path.join(save_dir, LABEL_ENC_FILE)
    with open(le_path, "wb") as f:
        pickle.dump(label_encoder, f)

    # 4. config (컬럼 매핑 + 학습 메타 포함)
    config["label_classes"] = list(label_encoder.classes_)
    config["num_classes"] = len(label_encoder.classes_)
    save_config(config, save_dir)

    logger.info(f"아티팩트 저장 완료 → {save_dir}")
    print(f"  ✓ 모델 저장: {save_dir}")
    print(f"  ✓ 클래스: {list(label_encoder.classes_)}")


# ══════════════════════════════════════════════════════════════
# 아티팩트 로드
# ══════════════════════════════════════════════════════════════

def load_artifacts(model_dir: str) -> Tuple:
    """
    model_dir에서 모델, 토크나이저, 레이블인코더, config를 로드.

    Returns:
        (model, tokenizer, label_encoder, config)
    """
    # config 로드
    config = load_config(model_dir)
    model_type = config.get("model_type", "t5")
    num_classes = config["num_classes"]

    logger.info(f"아티팩트 로드: {model_dir} (type={model_type}, classes={num_classes})")

    # 모델 로드
    if model_type == "bart":
        model = MultiTaskBartModel.load(model_dir)
    else:
        model = MultiTaskT5Model.load(model_dir)

    # 토크나이저 로드 (로컬 경로, 오프라인)
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True
        )
    except OSError as e:
        raise OSError(
            f"토크나이저 로드 실패: {model_dir}\n"
            f"  → 모델 저장이 완료된 경로를 확인하세요.\n"
            f"  원본 오류: {e}"
        ) from e

    # 레이블 인코더 로드
    le_path = os.path.join(model_dir, LABEL_ENC_FILE)
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"레이블 인코더 없음: {le_path}")
    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    logger.info(f"아티팩트 로드 완료: {len(label_encoder.classes_)}개 클래스")
    return model, tokenizer, label_encoder, config


# ══════════════════════════════════════════════════════════════
# 모델 목록 조회 (버전 관리)
# ══════════════════════════════════════════════════════════════

def list_models(models_root: str = "./saved_model") -> List[Dict]:
    """
    models_root 아래에서 config.json이 있는 모든 모델 디렉터리 검색.

    Returns:
        [{"path": str, "version": str, "created_at": str,
          "model_type": str, "num_classes": int, "label_classes": list}, ...]
    """
    if not os.path.isdir(models_root):
        return []

    results = []
    for name in sorted(os.listdir(models_root)):
        candidate = os.path.join(models_root, name)
        config_path = os.path.join(candidate, CONFIG_FILE)
        if os.path.isdir(candidate) and os.path.exists(config_path):
            try:
                with open(config_path, encoding="utf-8") as f:
                    cfg = json.load(f)
                results.append({
                    "path": candidate,
                    "name": name,
                    "version": cfg.get("version", "?"),
                    "created_at": cfg.get("created_at", "?"),
                    "model_type": cfg.get("model_type", "?"),
                    "num_classes": cfg.get("num_classes", 0),
                    "label_classes": cfg.get("label_classes", []),
                    "column_mapping": cfg.get("column_mapping", {}),
                })
            except Exception as e:
                logger.warning(f"config 읽기 실패: {candidate}: {e}")
    return results


def print_model_list(models_root: str = "./saved_model"):
    """저장된 모델 목록을 콘솔에 출력"""
    models = list_models(models_root)
    if not models:
        print(f"저장된 모델 없음: {models_root}")
        return
    print(f"\n{'═'*60}")
    print(f"  저장된 모델 목록 ({models_root})")
    print(f"{'═'*60}")
    for m in models:
        print(f"  [{m['name']}]")
        print(f"    경로      : {m['path']}")
        print(f"    생성일    : {m['created_at']}")
        print(f"    모델 타입 : {m['model_type']}")
        print(f"    클래스 수 : {m['num_classes']}  → {m['label_classes']}")
        cm = m.get("column_mapping", {})
        if cm:
            print(f"    컬럼 매핑 : {cm}")
        print()


# ══════════════════════════════════════════════════════════════
# 모델 정보 출력
# ══════════════════════════════════════════════════════════════

def print_model_info(model_dir: str):
    """단일 모델 상세 정보 출력"""
    try:
        config = load_config(model_dir)
    except FileNotFoundError as e:
        print(str(e))
        return

    print(f"\n{'═'*60}")
    print(f"  모델 정보: {model_dir}")
    print(f"{'═'*60}")

    for key, val in config.items():
        if key == "train_params":
            print(f"  학습 파라미터:")
            for k, v in val.items():
                print(f"    {k:20s}: {v}")
        elif key == "data_stats":
            print(f"  데이터 통계:")
            for k, v in val.items():
                print(f"    {k:20s}: {v}")
        elif key == "column_mapping":
            print(f"  컬럼 매핑:")
            for k, v in val.items():
                print(f"    {k:20s}: {v}")
        else:
            print(f"  {key:20s}: {val}")
    print()
