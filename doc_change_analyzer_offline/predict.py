"""
predict.py ─ 학습된 모델로 예측 수행 (오프라인 CLI)

포함:
  · predict_single()   : 단일 파일 예측
  · predict_dir()      : 디렉터리 내 다중 파일 예측
  · _predict_df()      : 내부 DataFrame 예측 로직 (모델 재사용)

CLI 사용:
  # 단일 파일
  python main.py predict --model_path ./saved_model/v1
                          --input_file ./data/new.xlsx
                          --output_file ./predictions/result.xlsx

  # 다중 파일 (디렉터리)
  python main.py predict --model_path ./saved_model/v1
                          --input_dir ./data/predict
                          --output_mode separate
                          --output_dir ./predictions

  # 다중 파일 통합
  python main.py predict --model_path ./saved_model/v1
                          --input_dir ./data/predict
                          --output_mode combined
                          --output_file ./predictions/combined.xlsx
"""

import os
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Optional, List, Tuple, Callable, Dict

from utils import (
    clean_text, build_input_text, diff_highlight,
    load_excel, load_excel_dir, DocChangeDataset,
)
from model_loader import load_artifacts

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 내부 예측 로직 (모델 재사용)
# ══════════════════════════════════════════════════════════════

def _predict_df(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    model,
    tokenizer,
    label_encoder,
    device: str,
    batch_size: int = 8,
    max_input_len: int = 256,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    progress: bool = True,
) -> pd.DataFrame:
    """
    단일 DataFrame에 예측 결과 컬럼을 추가해 반환.
    model/tokenizer/label_encoder는 외부에서 주입 → 다중 파일에서 재사용 가능.

    추가 컬럼:
        pred_summary, pred_reason, pred_code, confidence_score, diff_highlight
    """
    if before_col not in df.columns:
        raise ValueError(f"before 컬럼 없음: '{before_col}'  | 사용 가능: {list(df.columns)}")
    if after_col not in df.columns:
        raise ValueError(f"after 컬럼 없음: '{after_col}'  | 사용 가능: {list(df.columns)}")

    result = df.copy()
    result["_b"] = df[before_col].apply(clean_text)
    result["_a"] = df[after_col].apply(clean_text)
    result["input_text"] = result.apply(
        lambda r: build_input_text(r["_b"], r["_a"]), axis=1
    )
    result["diff_highlight"] = result.apply(
        lambda r: diff_highlight(r["_b"], r["_a"]), axis=1
    )

    predict_ds = DocChangeDataset(
        result[["input_text"]].copy(),
        tokenizer,
        max_input_len=max_input_len,
        max_target_len=max_new_tokens,
        is_predict=True,
    )

    eff_batch = min(batch_size, max(1, len(result)))
    loader = DataLoader(predict_ds, batch_size=eff_batch, shuffle=False, num_workers=0)

    all_summaries, all_reasons, all_codes, all_conf = [], [], [], []
    total_batches = len(loader)

    # tqdm 래퍼
    iterator = loader
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(loader, desc="  배치 예측", unit="batch", leave=False)
        except ImportError:
            pass

    for batch in iterator:
        input_ids     = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        summaries = model.generate_summary(
            input_ids, attention_mask, tokenizer, max_new_tokens, num_beams
        )
        reasons = model.generate_reason(
            input_ids, attention_mask, tokenizer, max_new_tokens, num_beams
        )
        pred_labels, confidences = model.predict_code(input_ids, attention_mask)

        all_summaries.extend(summaries)
        all_reasons.extend(reasons)
        all_codes.extend(pred_labels.tolist())
        all_conf.extend(confidences.tolist())

    # 레이블 역변환
    try:
        decoded = label_encoder.inverse_transform(all_codes)
    except Exception as e:
        logger.warning(f"레이블 역변환 실패 ({e}) → 정수 그대로 사용")
        decoded = [str(c) for c in all_codes]

    result["pred_summary"]      = all_summaries
    result["pred_reason"]       = all_reasons
    result["pred_code"]         = decoded
    result["confidence_score"]  = [round(c, 4) for c in all_conf]

    result.drop(columns=["_b", "_a", "input_text"], inplace=True)
    return result


# ══════════════════════════════════════════════════════════════
# 단일 파일 예측 (공개 API)
# ══════════════════════════════════════════════════════════════

def predict_single(
    model_path: str,
    input_file: str,
    output_file: str = "prediction_result.xlsx",
    before_col: Optional[str] = None,
    after_col: Optional[str] = None,
    batch_size: int = 8,
    max_input_len: int = 256,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    단일 엑셀 파일에 대해 예측 후 결과 저장.

    before_col / after_col 이 None이면 config.json의 column_mapping에서 자동 읽음.
    """
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'━'*50}")
    print(f"  단일 파일 예측")
    print(f"  모델  : {model_path}")
    print(f"  입력  : {input_file}")
    print(f"  출력  : {output_file}")
    print(f"{'━'*50}\n")

    # ── 아티팩트 로드 ──
    print("  [1/3] 모델 로드...")
    model, tokenizer, label_encoder, config = load_artifacts(model_path)
    model.to(device)
    model.eval()

    # 컬럼 자동 매핑 (config 기반)
    col_map = config.get("column_mapping", {})
    before_col = before_col or col_map.get("before", "before")
    after_col  = after_col  or col_map.get("after", "after")
    print(f"       컬럼 매핑: before='{before_col}', after='{after_col}'")

    # ── 데이터 로드 ──
    print("  [2/3] 데이터 로드...")
    df = load_excel(input_file, chunksize=chunksize)
    print(f"       {len(df)}행 로드")

    # ── 예측 ──
    print("  [3/3] 예측 실행...")
    result_df = _predict_df(
        df, before_col, after_col,
        model, tokenizer, label_encoder, device,
        batch_size, max_input_len, max_new_tokens, num_beams,
    )

    # ── 저장 ──
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    result_df.to_excel(output_file, index=False, engine="openpyxl")

    avg_conf = result_df["confidence_score"].mean()
    print(f"\n  ✅ 예측 완료!")
    print(f"     결과 행수       : {len(result_df)}")
    print(f"     평균 신뢰도     : {avg_conf:.1%}")
    print(f"     저장 경로       : {output_file}")

    return result_df


# ══════════════════════════════════════════════════════════════
# 다중 파일 예측
# ══════════════════════════════════════════════════════════════

def predict_dir(
    model_path: str,
    input_dir: str,
    output_mode: str = "separate",          # "separate" | "combined"
    output_dir: str = "./predictions",
    output_file: str = "prediction_result_combined.xlsx",
    before_col: Optional[str] = None,
    after_col: Optional[str] = None,
    batch_size: int = 8,
    max_input_len: int = 256,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    chunksize: Optional[int] = None,
) -> Tuple[List[Dict], Optional[pd.DataFrame]]:
    """
    디렉터리 내 모든 xlsx 파일에 대해 예측 수행.

    output_mode:
        "separate" → 파일별 result_{원본명}.xlsx 저장
        "combined" → 하나의 파일로 통합 저장

    Returns:
        (results: [{"file": str, "df": DataFrame, "rows": int, "error": str}],
         combined_df: DataFrame or None)
    """
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'━'*50}")
    print(f"  다중 파일 예측")
    print(f"  모델      : {model_path}")
    print(f"  입력 폴더 : {input_dir}")
    print(f"  출력 방식 : {output_mode}")
    print(f"{'━'*50}\n")

    # ── 모델 1회 로드 ──
    print("  [1/?] 모델 로드 (1회)...")
    model, tokenizer, label_encoder, config = load_artifacts(model_path)
    model.to(device)
    model.eval()

    col_map = config.get("column_mapping", {})
    _before = before_col or col_map.get("before", "before")
    _after  = after_col  or col_map.get("after", "after")
    print(f"       컬럼 매핑: before='{_before}', after='{_after}'")

    # ── 파일 목록 로드 ──
    print(f"  [2/?] {input_dir} 파일 목록 로드...")
    loaded_files = load_excel_dir(input_dir, chunksize=chunksize, progress=True)
    ok_files  = [(fp, df) for fp, df, err in loaded_files if df is not None]
    err_files = [(fp, err) for fp, df, err in loaded_files if df is None]

    if err_files:
        for fp, err in err_files:
            print(f"  ⚠️  로드 실패: {os.path.basename(fp)}: {err}")
    if not ok_files:
        raise ValueError(f"예측 가능한 파일 없음: {input_dir}")

    print(f"       {len(ok_files)}개 파일 예측 준비 / {len(err_files)}개 실패")
    os.makedirs(output_dir, exist_ok=True)

    # ── 파일별 예측 ──
    results = []
    combined_parts = []

    try:
        from tqdm import tqdm
        file_iter = tqdm(ok_files, desc="  파일별 예측", unit="파일")
    except ImportError:
        file_iter = ok_files

    for fp, df in file_iter:
        fname = os.path.basename(fp)
        try:
            result_df = _predict_df(
                df, _before, _after,
                model, tokenizer, label_encoder, device,
                batch_size, max_input_len, max_new_tokens, num_beams,
                progress=False,          # 파일 레벨 tqdm과 중첩 방지
            )
            result_df["_source_file"] = fname

            out_path = ""
            if output_mode == "separate":
                out_path = os.path.join(output_dir, f"result_{fname}")
                result_df.to_excel(out_path, index=False, engine="openpyxl")
                logger.info(f"  → 저장: {out_path}")

            results.append({"file": fname, "df": result_df, "rows": len(result_df), "error": ""})
            combined_parts.append(result_df)

        except Exception as e:
            import traceback as tb
            msg = f"{type(e).__name__}: {e}\n{tb.format_exc()}"
            logger.error(f"[{fname}] 예측 실패: {msg}")
            results.append({"file": fname, "df": None, "rows": 0, "error": msg})

    # ── 통합 저장 ──
    combined_df = None
    if output_mode == "combined" and combined_parts:
        combined_df = pd.concat(combined_parts, ignore_index=True)
        combined_df.to_excel(output_file, index=False, engine="openpyxl")
        print(f"\n  📦 통합 결과 저장: {output_file} ({len(combined_df)}행)")

    # ── 요약 ──
    success = sum(1 for r in results if not r["error"])
    total_rows = sum(r["rows"] for r in results)
    print(f"\n  ✅ 다중 예측 완료!")
    print(f"     성공 파일 : {success}/{len(results)}개")
    print(f"     총 예측 행: {total_rows}건")
    if output_mode == "separate":
        print(f"     저장 폴더 : {output_dir}")

    # ── 실패 파일 목록 ──
    failed = [r for r in results if r["error"]]
    if failed:
        print(f"\n  ⚠️  실패한 파일 ({len(failed)}개):")
        for r in failed:
            print(f"     ❌ {r['file']}: {r['error'][:100]}")

    return results, combined_df


# ══════════════════════════════════════════════════════════════
# 단일 DataFrame 예측 (API 호출용)
# ══════════════════════════════════════════════════════════════

def predict(
    df: pd.DataFrame,
    before_col: str,
    after_col: str,
    model_path: str,
    batch_size: int = 8,
    max_input_len: int = 256,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    output_path: str = "prediction_result.xlsx",
) -> pd.DataFrame:
    """
    DataFrame을 직접 받아 예측 (Python API용).
    """
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, label_encoder, config = load_artifacts(model_path)
    model.to(device)
    model.eval()

    result_df = _predict_df(
        df, before_col, after_col,
        model, tokenizer, label_encoder, device,
        batch_size, max_input_len, max_new_tokens, num_beams,
    )

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        result_df.to_excel(output_path, index=False, engine="openpyxl")
        logger.info(f"예측 결과 저장: {output_path}")

    return result_df
