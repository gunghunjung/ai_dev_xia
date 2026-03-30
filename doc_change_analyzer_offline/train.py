"""
train.py ─ 멀티태스크 모델 학습 (오프라인 CLI)

포함:
  · train()            : 처음부터 학습
  · incremental_train(): 기존 모델 로드 후 추가 데이터 fine-tuning
  · _run_epoch()       : 단일 에폭 실행 (공유)
  · _run_loop()        : 학습/검증 루프 (공유)
  · _make_loaders()    : DataLoader 생성 (공유)
  · EarlyStopping      : 조기 종료

CLI 사용:
  python main.py train --input_dir ./data --model_path ./local_model/kobart
                        --save_path ./saved_model/v1
                        --before_col 변경전 --after_col 변경후
                        --summary_col 요약 --reason_col 사유 --code_col 코드
"""

import os
import time
import pickle
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from typing import Callable, Optional, Dict, Tuple

from utils import (
    preprocess, preprocess_merged,
    split_data, compute_class_weights,
    DocChangeDataset, STANDARD_COLS,
)
from model import build_model
from model_loader import save_artifacts

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Early Stopping
# ══════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ══════════════════════════════════════════════════════════════
# 단일 에폭 실행
# ══════════════════════════════════════════════════════════════

def _run_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: str,
    training: bool = True,
    loss_weights: Dict = None,
) -> Dict:
    if loss_weights is None:
        loss_weights = {"summary": 1.0, "reason": 1.0, "code": 2.0}

    model.train(training)
    total_loss = sum_loss_tot = rea_loss_tot = code_loss_tot = 0.0
    correct = total_samples = 0

    for batch in loader:
        input_ids       = batch["input_ids"].to(device)
        attention_mask  = batch["attention_mask"].to(device)
        summary_labels  = batch["summary_labels"].to(device)
        reason_labels   = batch["reason_labels"].to(device)
        code_labels     = batch["code_labels"].to(device)

        code_logits, s_loss, r_loss, c_loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            summary_labels=summary_labels,
            reason_labels=reason_labels,
            code_labels=code_labels,
        )

        loss = torch.tensor(0.0, device=device)
        if s_loss is not None:
            loss = loss + loss_weights["summary"] * s_loss
            sum_loss_tot += s_loss.item()
        if r_loss is not None:
            loss = loss + loss_weights["reason"] * r_loss
            rea_loss_tot += r_loss.item()
        if c_loss is not None:
            loss = loss + loss_weights["code"] * c_loss
            code_loss_tot += c_loss.item()

        total_loss += loss.item()
        correct += (code_logits.argmax(-1) == code_labels).sum().item()
        total_samples += len(code_labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    n = max(len(loader), 1)
    return {
        "loss":     total_loss / n,
        "accuracy": correct / total_samples if total_samples else 0.0,
        "sum_loss": sum_loss_tot / n,
        "rea_loss": rea_loss_tot / n,
        "cod_loss": code_loss_tot / n,
    }


# ══════════════════════════════════════════════════════════════
# 학습/검증 루프
# ══════════════════════════════════════════════════════════════

def _run_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    epochs: int,
    patience: int,
    loss_weights: Dict,
    device: str,
    progress_fn: Optional[Callable] = None,
) -> Tuple:
    """
    학습 루프 실행.
    progress_fn: (epoch, total_epochs, log_str) → None  (tqdm 연동용)
    Returns: (model, history, best_val_loss)
    """
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    es = EarlyStopping(patience=patience)
    history = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if use_tqdm:
            train_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch:02d}/{epochs} [Train]",
                leave=False,
                unit="batch",
            )
        else:
            train_iter = train_loader

        # ── 학습 에폭 ──
        model.train()
        total_loss = sum_l = rea_l = cod_l = 0.0
        correct = total_s = 0
        lw = loss_weights or {"summary": 1.0, "reason": 1.0, "code": 2.0}

        for batch in train_iter:
            input_ids      = batch["input_ids"].to(device)
            attn_mask      = batch["attention_mask"].to(device)
            sum_labels     = batch["summary_labels"].to(device)
            rea_labels     = batch["reason_labels"].to(device)
            cod_labels     = batch["code_labels"].to(device)

            logits, sl, rl, cl = model(
                input_ids=input_ids, attention_mask=attn_mask,
                summary_labels=sum_labels, reason_labels=rea_labels,
                code_labels=cod_labels,
            )
            loss = torch.tensor(0.0, device=device)
            if sl is not None: loss = loss + lw["summary"] * sl; sum_l += sl.item()
            if rl is not None: loss = loss + lw["reason"]  * rl; rea_l += rl.item()
            if cl is not None: loss = loss + lw["code"]    * cl; cod_l += cl.item()
            total_loss += loss.item()
            correct += (logits.argmax(-1) == cod_labels).sum().item()
            total_s += len(cod_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        n = max(len(train_loader), 1)
        train_m = {
            "loss": total_loss / n,
            "accuracy": correct / total_s if total_s else 0.0,
        }

        # ── 검증 에폭 ──
        model.eval()
        val_total = val_correct = val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                ii  = batch["input_ids"].to(device)
                am  = batch["attention_mask"].to(device)
                sl_ = batch["summary_labels"].to(device)
                rl_ = batch["reason_labels"].to(device)
                cl_ = batch["code_labels"].to(device)
                logits, sv, rv, cv = model(
                    input_ids=ii, attention_mask=am,
                    summary_labels=sl_, reason_labels=rl_, code_labels=cl_,
                )
                v_loss = torch.tensor(0.0, device=device)
                if sv is not None: v_loss = v_loss + lw["summary"] * sv
                if rv is not None: v_loss = v_loss + lw["reason"]  * rv
                if cv is not None: v_loss = v_loss + lw["code"]    * cv
                val_total += v_loss.item()
                val_correct += (logits.argmax(-1) == cl_).sum().item()
                val_samples += len(cl_)

        nv = max(len(val_loader), 1)
        val_m = {
            "loss": val_total / nv,
            "accuracy": val_correct / val_samples if val_samples else 0.0,
        }

        scheduler.step()
        elapsed = time.time() - t0

        log_str = (
            f"[Epoch {epoch:02d}/{epochs}] "
            f"Train Loss={train_m['loss']:.4f} Acc={train_m['accuracy']:.3f} | "
            f"Val Loss={val_m['loss']:.4f} Acc={val_m['accuracy']:.3f} | "
            f"{elapsed:.1f}s"
        )
        logger.info(log_str)
        print(f"  {log_str}")

        record = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc":  train_m["accuracy"],
            "val_loss":   val_m["loss"],
            "val_acc":    val_m["accuracy"],
        }
        history.append(record)

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if progress_fn:
            progress_fn(epoch, epochs, log_str)

        if es.step(val_m["loss"]):
            print(f"  ⏹ Early stopping @ epoch {epoch}")
            logger.info(f"Early stopping @ epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_val_loss


# ══════════════════════════════════════════════════════════════
# DataLoader 생성
# ══════════════════════════════════════════════════════════════

def _make_loaders(
    train_df,
    val_df,
    tokenizer,
    max_input_len: int,
    max_target_len: int,
    batch_size: int,
    device: str,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = DocChangeDataset(train_df, tokenizer, max_input_len, max_target_len)
    val_ds   = DocChangeDataset(val_df,   tokenizer, max_input_len, max_target_len)

    eff_batch = min(batch_size, max(2, len(train_df) // 4))
    logger.info(f"유효 배치 크기: {eff_batch} (요청={batch_size}, 데이터={len(train_df)})")

    train_loader = DataLoader(
        train_ds, batch_size=eff_batch, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=eff_batch, shuffle=False, num_workers=0,
    )
    return train_loader, val_loader


# ══════════════════════════════════════════════════════════════
# 메인 학습 함수
# ══════════════════════════════════════════════════════════════

def train(
    df,
    col_map: Dict[str, str],
    base_model_path: str,
    save_path: str,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-5,
    max_input_len: int = 256,
    max_target_len: int = 128,
    patience: int = 3,
    loss_weights: Dict = None,
    progress_fn: Optional[Callable] = None,
    extra_config: Dict = None,
) -> Dict:
    """
    처음부터 학습.

    Args:
        df              : 표준화된 DataFrame 또는 원본 DataFrame
        col_map         : {"before": "실제컬럼명", ...}
                          (표준 컬럼이면 {"before":"before", ...} 사용)
        base_model_path : 로컬 pretrained 모델 경로
                          (./local_model/kobart 또는 ./local_model/t5)
        save_path       : 학습 결과 저장 경로
        extra_config    : config.json에 추가할 메타데이터

    Returns: 학습 결과 dict
    """
    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"디바이스: {device} | 데이터: {len(df)}행")
    print(f"\n{'━'*50}")
    print(f"  학습 시작")
    print(f"  디바이스  : {device}")
    print(f"  기반 모델 : {base_model_path}")
    print(f"  저장 경로 : {save_path}")
    print(f"  에폭/배치 : {epochs} / {batch_size}")
    print(f"{'━'*50}\n")

    # ── 전처리 ──
    print("  [1/5] 데이터 전처리...")
    processed_df, label_encoder = preprocess(
        df,
        col_map.get("before", "before"),
        col_map.get("after", "after"),
        col_map.get("summary", "summary"),
        col_map.get("reason", "reason"),
        col_map.get("code", "code"),
    )
    num_classes = len(label_encoder.classes_)
    print(f"       → {len(processed_df)}행, {num_classes}개 클래스: {list(label_encoder.classes_)}")

    # ── 분할 ──
    train_df, val_df = split_data(processed_df)

    # ── 클래스 가중치 ──
    cw = compute_class_weights(train_df["code_label"].values, num_classes)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    # ── 토크나이저 ──
    print(f"  [2/5] 토크나이저 로드: {base_model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, local_files_only=True
        )
    except OSError as e:
        raise OSError(
            f"토크나이저 로드 실패: {base_model_path}\n"
            "  → SETUP.md 참고해 로컬 모델을 다운로드하세요.\n"
            f"  오류: {e}"
        ) from e

    # ── DataLoader ──
    print("  [3/5] DataLoader 구성...")
    train_loader, val_loader = _make_loaders(
        train_df, val_df, tokenizer,
        max_input_len, max_target_len, batch_size, device,
    )

    # ── 모델 ──
    print(f"  [4/5] 모델 초기화: {base_model_path}")
    model = build_model(base_model_path, num_classes, class_weights, local_only=True)
    model.to(device)

    # ── Optimizer / Scheduler ──
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # ── 학습 루프 ──
    print(f"  [5/5] 학습 루프 시작 (early stopping patience={patience})")
    model, history, best_val_loss = _run_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        epochs, patience, loss_weights or {}, device, progress_fn,
    )

    # ── 저장 ──
    model_type = "bart" if "bart" in base_model_path.lower() else "t5"
    config = {
        "version": "1.0",
        "model_type": model_type,
        "base_model_path": base_model_path,
        "num_classes": num_classes,
        "column_mapping": col_map,
        "train_params": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_input_len": max_input_len,
            "max_target_len": max_target_len,
            "early_stopping_patience": patience,
        },
        "data_stats": {
            "total_rows": len(df),
            "train_rows": len(train_df),
            "val_rows":   len(val_df),
            "num_classes": num_classes,
            "label_classes": list(label_encoder.classes_),
        },
    }
    if extra_config:
        config.update(extra_config)

    save_artifacts(model, tokenizer, label_encoder, config, save_path)

    print(f"\n  ✅ 학습 완료!")
    print(f"     Best Val Loss : {best_val_loss:.4f}")
    print(f"     저장 경로     : {save_path}")

    return {
        "history": history,
        "label_encoder": label_encoder,
        "best_val_loss": best_val_loss,
        "save_path": save_path,
        "num_classes": num_classes,
    }


# ══════════════════════════════════════════════════════════════
# 증분 학습 (Incremental Fine-tuning)
# ══════════════════════════════════════════════════════════════

def incremental_train(
    new_df,
    col_map: Dict[str, str],
    model_dir: str,
    save_path: Optional[str] = None,       # None이면 model_dir 덮어씀
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-5,           # 낮은 LR → 기존 지식 보존
    max_input_len: int = 256,
    max_target_len: int = 128,
    patience: int = 3,
    loss_weights: Dict = None,
    progress_fn: Optional[Callable] = None,
) -> Dict:
    """
    기존 저장 모델을 불러와 새 데이터로 추가 학습.
    - 기존 label_encoder 재사용 (미지 클래스 행 자동 제외)
    - 낮은 LR로 catastrophic forgetting 방지
    - 완료 후 save_path(없으면 model_dir)에 저장
    """
    import os
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if save_path is None:
        save_path = model_dir

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"증분 학습 시작 | 새 데이터: {len(new_df)}행 | 모델: {model_dir}")
    print(f"\n{'━'*50}")
    print(f"  증분 학습 시작")
    print(f"  기존 모델  : {model_dir}")
    print(f"  디바이스   : {device}")
    print(f"  LR (낮음)  : {learning_rate}")
    print(f"{'━'*50}\n")

    # ── 기존 아티팩트 로드 ──
    from model_loader import load_artifacts, load_config
    model, tokenizer, label_encoder, config = load_artifacts(model_dir)

    existing_classes = set(label_encoder.classes_)
    num_classes = len(existing_classes)

    # ── 새 데이터 전처리 ──
    processed_df, new_le = preprocess(
        new_df,
        col_map.get("before", "before"),
        col_map.get("after", "after"),
        col_map.get("summary", "summary"),
        col_map.get("reason", "reason"),
        col_map.get("code", "code"),
    )

    # 미지 클래스 처리
    unseen = set(new_le.classes_) - existing_classes
    if unseen:
        print(f"  ⚠️  새 코드 클래스 발견: {unseen} → 완전 재학습 권장")
        logger.warning(f"미지 클래스: {unseen}")

    # 기존 label_encoder로 재매핑
    def safe_transform(code):
        if code in existing_classes:
            return int(label_encoder.transform([code])[0])
        return -1

    processed_df["code_label"] = processed_df["code"].apply(safe_transform)
    before = len(processed_df)
    processed_df = processed_df[processed_df["code_label"] >= 0].reset_index(drop=True)
    excluded = before - len(processed_df)
    if excluded:
        print(f"  ℹ️  미지 클래스 {excluded}행 제외")

    if len(processed_df) == 0:
        raise ValueError("증분 학습 가능한 데이터가 없습니다.")

    # ── 분할 / DataLoader ──
    train_df, val_df = split_data(processed_df)
    cw = compute_class_weights(train_df["code_label"].values, num_classes)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    model.to(device)
    # class_weights 버퍼 업데이트
    if hasattr(model, "class_weights"):
        model.class_weights = class_weights.to(device)

    train_loader, val_loader = _make_loaders(
        train_df, val_df, tokenizer,
        max_input_len, max_target_len, batch_size, device,
    )

    # ── Optimizer / Scheduler (낮은 LR) ──
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

    # ── 학습 루프 ──
    model, history, best_val_loss = _run_loop(
        model, train_loader, val_loader, optimizer, scheduler,
        epochs, patience, loss_weights or {}, device, progress_fn,
    )

    # ── 저장 ──
    config["train_params"]["incremental_lr"] = learning_rate
    config["data_stats"]["incremental_rows"] = len(processed_df)
    config["data_stats"]["excluded_rows"] = excluded
    from model_loader import save_artifacts as sa
    sa(model, tokenizer, label_encoder, config, save_path)

    print(f"\n  ✅ 증분 학습 완료!")
    print(f"     Best Val Loss : {best_val_loss:.4f}")
    print(f"     저장 경로     : {save_path}")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "save_path": save_path,
        "excluded_rows": excluded,
        "unseen_classes": list(unseen),
    }
