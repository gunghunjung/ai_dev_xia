"""
model.py ─ 멀티태스크 오프라인 모델 (T5 / KoBART)

핵심 조건:
  · 모든 from_pretrained() 호출에 local_files_only=True 사용
  · 인터넷 없이 ./local_model/{kobart|t5}/ 에서 로드
  · CPU / GPU 자동 선택

구조:
  · 공유 인코더 (T5 또는 BART)
  · Decoder 1 : summary 생성
  · Decoder 2 : reason  생성 (동일 가중치 공유)
  · Classifier: encoder 풀링 → code 분류
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# T5 기반 멀티태스크 모델
# ══════════════════════════════════════════════════════════════

class MultiTaskT5Model(nn.Module):
    """
    T5 기반 멀티태스크 모델 (오프라인).
    model_name: 로컬 경로 (예: './local_model/t5')
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        class_weights: torch.Tensor = None,
        dropout: float = 0.1,
        local_only: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        logger.info(f"T5 모델 로드: {model_name}  (local_only={local_only})")
        try:
            self.t5 = T5ForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=local_only,
            )
        except OSError as e:
            raise OSError(
                f"\n[오프라인 모델 없음] '{model_name}'\n"
                "  → SETUP.md를 참고해 모델을 ./local_model/ 에 미리 다운로드하세요.\n"
                f"  원본 오류: {e}"
            ) from e

        d_model = self.t5.config.d_model

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )
        logger.info(f"  hidden_size={d_model}, num_classes={num_classes}")

    # ── 인코더 풀링 ──────────────────────────────────────────

    def _encode(self, input_ids, attention_mask):
        enc_out = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state                       # (B, L, D)
        mask_ex = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_ex).sum(1) / mask_ex.sum(1).clamp(min=1e-9)
        return pooled, enc_out

    # ── 순전파 ───────────────────────────────────────────────

    def forward(
        self,
        input_ids,
        attention_mask,
        summary_labels=None,
        reason_labels=None,
        code_labels=None,
    ):
        pooled, enc_out = self._encode(input_ids, attention_mask)
        code_logits = self.classifier(pooled)

        summary_loss = None
        if summary_labels is not None:
            summary_loss = self.t5(
                encoder_outputs=enc_out,
                attention_mask=attention_mask,
                labels=summary_labels,
            ).loss

        reason_loss = None
        if reason_labels is not None:
            reason_loss = self.t5(
                encoder_outputs=enc_out,
                attention_mask=attention_mask,
                labels=reason_labels,
            ).loss

        code_loss = None
        if code_labels is not None:
            code_loss = F.cross_entropy(
                code_logits,
                code_labels,
                weight=self.class_weights.to(code_logits.device),
            )

        return code_logits, summary_loss, reason_loss, code_loss

    # ── 텍스트 생성 (inference) ──────────────────────────────

    def generate_summary(self, input_ids, attention_mask, tokenizer,
                         max_new_tokens=128, num_beams=4) -> list:
        self.eval()
        with torch.no_grad():
            ids = self.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

    def generate_reason(self, input_ids, attention_mask, tokenizer,
                        max_new_tokens=128, num_beams=4) -> list:
        return self.generate_summary(
            input_ids, attention_mask, tokenizer, max_new_tokens, num_beams
        )

    def predict_code(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            pooled, _ = self._encode(input_ids, attention_mask)
            logits = self.classifier(pooled)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_labels = probs.max(dim=-1)
        return pred_labels.cpu(), confidence.cpu()

    # ── 저장 ─────────────────────────────────────────────────

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.t5.save_pretrained(save_dir)
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
        meta = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "model_type": "t5",
        }
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(f"T5 모델 저장: {save_dir}")

    @classmethod
    def load(cls, save_dir: str, class_weights=None):
        meta_path = os.path.join(save_dir, "meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        model = cls(
            model_name=save_dir,          # fine-tuned weights (로컬 경로)
            num_classes=meta["num_classes"],
            class_weights=class_weights,
            local_only=True,              # 항상 로컬에서 로드
        )
        clf_path = os.path.join(save_dir, "classifier.pt")
        model.classifier.load_state_dict(
            torch.load(clf_path, map_location="cpu", weights_only=True)
        )
        logger.info(f"T5 모델 로드: {save_dir}")
        return model


# ══════════════════════════════════════════════════════════════
# BART 기반 멀티태스크 모델 (KoBART)
# ══════════════════════════════════════════════════════════════

class MultiTaskBartModel(nn.Module):
    """
    KoBART 기반 멀티태스크 모델 (오프라인).
    model_name: 로컬 경로 (예: './local_model/kobart')
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        class_weights: torch.Tensor = None,
        dropout: float = 0.1,
        local_only: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        logger.info(f"BART 모델 로드: {model_name}  (local_only={local_only})")
        try:
            self.bart = BartForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=local_only,
            )
        except OSError as e:
            raise OSError(
                f"\n[오프라인 모델 없음] '{model_name}'\n"
                "  → SETUP.md를 참고해 모델을 ./local_model/ 에 미리 다운로드하세요.\n"
                f"  원본 오류: {e}"
            ) from e

        d_model = self.bart.config.d_model

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )
        logger.info(f"  hidden_size={d_model}, num_classes={num_classes}")

    def _encode(self, input_ids, attention_mask):
        enc_out = self.bart.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = enc_out.last_hidden_state
        mask_ex = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask_ex).sum(1) / mask_ex.sum(1).clamp(min=1e-9)
        return pooled, enc_out

    def forward(self, input_ids, attention_mask,
                summary_labels=None, reason_labels=None, code_labels=None):
        pooled, enc_out = self._encode(input_ids, attention_mask)
        code_logits = self.classifier(pooled)

        summary_loss = None
        if summary_labels is not None:
            summary_loss = self.bart(
                encoder_outputs=enc_out,
                attention_mask=attention_mask,
                labels=summary_labels,
            ).loss

        reason_loss = None
        if reason_labels is not None:
            reason_loss = self.bart(
                encoder_outputs=enc_out,
                attention_mask=attention_mask,
                labels=reason_labels,
            ).loss

        code_loss = None
        if code_labels is not None:
            code_loss = F.cross_entropy(
                code_logits, code_labels,
                weight=self.class_weights.to(code_logits.device),
            )

        return code_logits, summary_loss, reason_loss, code_loss

    def generate_summary(self, input_ids, attention_mask, tokenizer,
                         max_new_tokens=128, num_beams=4) -> list:
        self.eval()
        with torch.no_grad():
            ids = self.bart.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return tokenizer.batch_decode(ids, skip_special_tokens=True)

    def generate_reason(self, input_ids, attention_mask, tokenizer,
                        max_new_tokens=128, num_beams=4) -> list:
        return self.generate_summary(
            input_ids, attention_mask, tokenizer, max_new_tokens, num_beams
        )

    def predict_code(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            pooled, _ = self._encode(input_ids, attention_mask)
            logits = self.classifier(pooled)
            probs = F.softmax(logits, dim=-1)
            confidence, pred_labels = probs.max(dim=-1)
        return pred_labels.cpu(), confidence.cpu()

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.bart.save_pretrained(save_dir)
        torch.save(self.classifier.state_dict(), os.path.join(save_dir, "classifier.pt"))
        with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "model_type": "bart",
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"BART 모델 저장: {save_dir}")

    @classmethod
    def load(cls, save_dir: str, class_weights=None):
        with open(os.path.join(save_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        model = cls(
            model_name=save_dir,
            num_classes=meta["num_classes"],
            class_weights=class_weights,
            local_only=True,
        )
        model.classifier.load_state_dict(
            torch.load(
                os.path.join(save_dir, "classifier.pt"),
                map_location="cpu",
                weights_only=True,
            )
        )
        logger.info(f"BART 모델 로드: {save_dir}")
        return model


# ══════════════════════════════════════════════════════════════
# 팩토리 함수
# ══════════════════════════════════════════════════════════════

def build_model(
    model_name: str,
    num_classes: int,
    class_weights: torch.Tensor = None,
    dropout: float = 0.1,
    local_only: bool = True,
) -> nn.Module:
    """
    model_name 경로에 따라 T5 또는 BART 모델 반환.
    'bart' 또는 'kobart'가 포함되면 BART 선택.
    """
    name_lower = model_name.lower()
    if "bart" in name_lower:
        return MultiTaskBartModel(
            model_name, num_classes, class_weights, dropout, local_only=local_only
        )
    else:
        return MultiTaskT5Model(
            model_name, num_classes, class_weights, dropout, local_only=local_only
        )
