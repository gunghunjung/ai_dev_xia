"""
utils.py ─ 데이터 전처리 유틸리티 (오프라인 CLI 전용)

기능:
  · 엑셀 안전 로드 (단일/다중, 청크 지원)
  · 텍스트 정리 / input_text 생성 / diff 강조
  · 컬럼 표준화 · 다중 파일 통합
  · 중복 제거 (완전 / 유사도)
  · 데이터 검증 · 통계 리포트
  · PyTorch Dataset 클래스
"""

import os
import re
import logging
import pickle
import traceback
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Callable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── 내부 표준 컬럼명 ──────────────────────────────────────────
STANDARD_COLS = ["before", "after", "summary", "reason", "code"]


# ══════════════════════════════════════════════════════════════
# 엑셀 로드
# ══════════════════════════════════════════════════════════════

def _col_letter_to_index(letter: str) -> int:
    """엑셀 열 문자(A~Z, AA~ZZ)를 0-based 정수 인덱스로 변환."""
    letter = letter.upper().strip()
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


def get_excel_columns(file_path: str, header: int = 0) -> List[str]:
    """
    엑셀 파일의 컬럼명 목록만 빠르게 반환 (nrows=0).
    header: 헤더로 사용할 행 번호 (0-based, Excel 기준 header+1 행)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일 없음: {file_path}")
    df = pd.read_excel(file_path, engine="openpyxl", header=header, nrows=0)
    return [str(c) for c in df.columns.tolist()]


def load_excel(
    file_path: str,
    chunksize: Optional[int] = None,
    header = 0,
    skiprows = None,
) -> pd.DataFrame:
    """
    경로 문자열로 엑셀 로드.
    header   : 헤더 행 (0-based 정수 or None=헤더 없음)
    skiprows : 상단에서 건너뛸 행 수 (int or None)
    chunksize > 0 이면 청크 단위로 읽어 메모리 절약.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일 없음: {file_path}")
    try:
        kw = dict(engine="openpyxl", header=header)
        if skiprows:
            kw["skiprows"] = skiprows
        if chunksize:
            chunks = []
            for chunk in pd.read_excel(file_path, chunksize=chunksize, **kw):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_excel(file_path, **kw)
        logger.info(f"[{os.path.basename(file_path)}] 로드 완료: {len(df)}행 × {len(df.columns)}열")
        return df
    except Exception as e:
        logger.error(f"엑셀 로드 실패 [{file_path}]: {e}")
        raise


def load_excel_safe(
    file_path: str,
    chunksize: Optional[int] = None,
    header = 0,
    skiprows = None,
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    예외를 잡아 (DataFrame 또는 None, 오류 메시지) 반환.
    성공 시 오류 메시지는 빈 문자열.
    """
    try:
        df = load_excel(file_path, chunksize=chunksize, header=header, skiprows=skiprows)
        return df, ""
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        return None, msg


def load_excel_dir(
    dir_path: str,
    chunksize: Optional[int] = None,
    progress: bool = True,
    header = 0,
    skiprows = None,
) -> List[Tuple[str, Optional[pd.DataFrame], str]]:
    """
    디렉터리 내 모든 .xlsx 파일 로드.
    Returns: [(file_path, df_or_None, error_msg), ...]
    """
    xlsx_files = sorted([
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".xlsx")
    ])

    if not xlsx_files:
        logger.warning(f"디렉터리에 xlsx 파일 없음: {dir_path}")
        return []

    results = []
    iterator = xlsx_files
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(xlsx_files, desc="엑셀 로드", unit="파일")
        except ImportError:
            pass

    for fp in iterator:
        df, err = load_excel_safe(fp, chunksize=chunksize, header=header, skiprows=skiprows)
        results.append((fp, df, err))

    ok = sum(1 for _, d, _ in results if d is not None)
    logger.info(f"디렉터리 로드 완료: {ok}/{len(results)}개 성공")
    return results


# ══════════════════════════════════════════════════════════════
# 텍스트 정리
# ══════════════════════════════════════════════════════════════

def clean_text(text) -> str:
    """앞뒤 공백, 연속 공백, 제어문자 제거"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return text.strip()


def build_input_text(before: str, after: str) -> str:
    """[BEFORE] {before} [AFTER] {after} 형식 생성"""
    return f"[BEFORE] {clean_text(before)} [AFTER] {clean_text(after)}"


def diff_highlight(before: str, after: str) -> str:
    """토큰 집합 차이 기반 변경 강조 (+추가 / -삭제)"""
    b_tokens = set(clean_text(before).split())
    a_tokens = set(clean_text(after).split())
    added = sorted(a_tokens - b_tokens)
    removed = sorted(b_tokens - a_tokens)
    parts = []
    if added:
        parts.append("+" + " +".join(added))
    if removed:
        parts.append("-" + " -".join(removed))
    return " / ".join(parts) if parts else "(변경없음)"


# ══════════════════════════════════════════════════════════════
# 데이터 검증
# ══════════════════════════════════════════════════════════════

def validate_dataframe(
    df: pd.DataFrame,
    col_map: Dict[str, str],
    file_name: str = "unknown",
    null_threshold: float = 0.5,
    min_text_len: int = 2,
    max_text_len: int = 2000,
) -> Tuple[bool, List[str]]:
    """
    DataFrame 품질 검증.
    Returns: (통과여부, [경고/에러 메시지 목록])
    """
    messages = []
    ok = True

    # 필수 컬럼 존재 확인
    for key in ["before", "after", "code"]:
        col = col_map.get(key, "")
        if not col or col not in df.columns:
            messages.append(f"[{file_name}] ❌ 필수 컬럼 없음: key='{key}', 매핑값='{col}'")
            ok = False

    if not ok:
        return False, messages

    # null 비율
    for key in ["before", "after"]:
        col = col_map[key]
        ratio = df[col].isna().mean()
        if ratio > null_threshold:
            messages.append(
                f"[{file_name}] ⚠️ '{col}' null 비율 {ratio:.1%} > 임계값 {null_threshold:.0%}"
            )

    # 텍스트 길이
    for key in ["before", "after"]:
        col = col_map[key]
        lengths = df[col].dropna().astype(str).str.len()
        short = (lengths < min_text_len).sum()
        long_ = (lengths > max_text_len).sum()
        if short:
            messages.append(f"[{file_name}] ⚠️ '{col}' 너무 짧은 텍스트 {short}건 → 제거 예정")
        if long_:
            messages.append(f"[{file_name}] ⚠️ '{col}' 너무 긴 텍스트 {long_}건 → 잘림 처리")

    return ok, messages


# ══════════════════════════════════════════════════════════════
# 컬럼 표준화
# ══════════════════════════════════════════════════════════════

def standardize_columns(
    df: pd.DataFrame,
    col_map: Dict[str, str],
    source_file: str = "",
    min_text_len: int = 2,
    max_text_len: int = 2000,
) -> Tuple[pd.DataFrame, Dict]:
    """
    col_map에 따라 컬럼을 표준 이름으로 변환 후 정제.

    col_map 예:
        {"before": "변경전", "after": "변경후",
         "summary": "요약", "reason": "사유", "code": "코드"}

    없는 선택 컬럼(summary, reason)은 빈 문자열로 채움.
    """
    stats = {"original": len(df), "null_removed": 0, "length_removed": 0, "final": 0}

    col_data = {}
    df_cols = list(df.columns)
    for std_col in STANDARD_COLS:
        actual = col_map.get(std_col, "")
        resolved = None

        if actual:
            # 1) 컬럼명 직접 매칭
            if actual in df.columns:
                resolved = actual
            else:
                # 2) 열 문자(E, F, G...) → 인덱스로 변환
                stripped = actual.strip().upper()
                if stripped and all(c.isalpha() for c in stripped):
                    idx = _col_letter_to_index(stripped)
                    if 0 <= idx < len(df_cols):
                        resolved = df_cols[idx]
                        logger.info(f"열 문자 '{actual}' → '{resolved}' 로 매핑")
                # 3) 숫자 인덱스 문자열 (예: "4")
                elif stripped.lstrip('-').isdigit():
                    idx = int(stripped)
                    if 0 <= idx < len(df_cols):
                        resolved = df_cols[idx]
                        logger.info(f"열 번호 '{actual}' → '{resolved}' 로 매핑")

        if resolved:
            col_data[std_col] = df[resolved].copy()
        else:
            if actual:
                logger.warning(f"컬럼 '{actual}' 없음 → 빈값으로 대체 (전체 컬럼: {df_cols})")
            col_data[std_col] = pd.Series([""] * len(df), dtype=str)

    out = pd.DataFrame(col_data)
    if source_file:
        out["_source_file"] = source_file

    # null 제거 (필수 컬럼)
    before_len = len(out)
    out.dropna(subset=["before", "after", "code"], inplace=True)
    stats["null_removed"] = before_len - len(out)

    # 텍스트 정리
    for col in ["before", "after", "summary", "reason"]:
        out[col] = out[col].apply(clean_text)

    # 길이 필터
    before_len2 = len(out)
    mask = (
        (out["before"].str.len() >= min_text_len) &
        (out["after"].str.len() >= min_text_len) &
        (out["before"].str.len() <= max_text_len) &
        (out["after"].str.len() <= max_text_len)
    )
    out = out[mask].reset_index(drop=True)
    stats["length_removed"] = before_len2 - len(out)

    out["code"] = out["code"].astype(str).str.strip()
    stats["final"] = len(out)

    logger.info(
        f"[{source_file or 'unnamed'}] 표준화: "
        f"{stats['original']} → null제거-{stats['null_removed']} / "
        f"길이제거-{stats['length_removed']} = {stats['final']}행"
    )
    return out, stats


# ══════════════════════════════════════════════════════════════
# 다중 파일 통합
# ══════════════════════════════════════════════════════════════

def merge_files(
    dataframes: List[Tuple[pd.DataFrame, Dict[str, str], str]],
    dedup_exact: bool = True,
    progress: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    여러 DataFrame을 표준화 후 하나로 통합.

    dataframes: [(df, col_map, file_name), ...]

    Returns: (통합 DataFrame, 통계 dict)
    """
    merged_parts = []
    all_stats = []
    error_files = []
    total = len(dataframes)

    iterator = enumerate(dataframes)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="파일 통합", unit="파일")
        except ImportError:
            pass

    for idx, (df, col_map, fname) in iterator:
        try:
            std_df, stats = standardize_columns(df, col_map, source_file=fname)
            stats["file"] = fname
            all_stats.append(stats)
            if len(std_df) > 0:
                merged_parts.append(std_df)
            else:
                logger.warning(f"[{fname}] 유효 데이터 없음 → 제외")
                error_files.append((fname, "유효 데이터 없음 (0행)"))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{fname}] 표준화 실패: {msg}")
            error_files.append((fname, msg))

    if not merged_parts:
        raise ValueError(
            "통합 가능한 데이터가 없습니다. 컬럼 매핑을 확인하세요.\n"
            f"오류 파일: {error_files}"
        )

    merged = pd.concat(merged_parts, ignore_index=True)
    before_dedup = len(merged)

    if dedup_exact:
        merged = deduplicate_exact(merged)

    summary = {
        "total_files": total,
        "success_files": len(merged_parts),
        "error_files": error_files,
        "file_stats": all_stats,
        "total_before_dedup": before_dedup,
        "dedup_removed": before_dedup - len(merged),
        "final_rows": len(merged),
    }
    logger.info(
        f"통합 완료: {total}개 파일 → {len(merged)}행 "
        f"(중복제거: {summary['dedup_removed']}건)"
    )
    return merged, summary


# ══════════════════════════════════════════════════════════════
# 중복 제거
# ══════════════════════════════════════════════════════════════

def deduplicate_exact(df: pd.DataFrame) -> pd.DataFrame:
    """before + after 기준 완전 중복 제거"""
    before = len(df)
    df = df.drop_duplicates(subset=["before", "after"], keep="first").reset_index(drop=True)
    logger.info(f"완전 중복 제거: {before} → {len(df)}행 ({before-len(df)}건 제거)")
    return df


def deduplicate_similarity(
    df: pd.DataFrame,
    threshold: float = 0.95,
    sample_limit: int = 5000,
) -> pd.DataFrame:
    """
    TF-IDF 코사인 유사도 기반 중복 제거.
    sample_limit 초과 시 완전 중복만 적용.
    """
    if len(df) > sample_limit:
        logger.warning(
            f"유사도 중복: 데이터 {len(df)}건 > {sample_limit}건 → 완전 중복만 적용"
        )
        return deduplicate_exact(df)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = (df["before"] + " " + df["after"]).tolist()
        vec = TfidfVectorizer(max_features=5000).fit_transform(texts)
        sim = cosine_similarity(vec)

        to_remove = set()
        n = len(df)
        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    to_remove.add(j)

        df = df.drop(index=list(to_remove)).reset_index(drop=True)
        logger.info(f"유사도 중복 제거: {n} → {len(df)}행 ({len(to_remove)}건 제거)")
    except Exception as e:
        logger.warning(f"유사도 중복 실패 ({e}) → 완전 중복만 적용")
        df = deduplicate_exact(df)

    return df


# ══════════════════════════════════════════════════════════════
# 전처리 (학습용)
# ══════════════════════════════════════════════════════════════

def preprocess(
    df: pd.DataFrame,
    before_col: str = "before",
    after_col: str = "after",
    summary_col: str = "summary",
    reason_col: str = "reason",
    code_col: str = "code",
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    원본 또는 표준화된 DataFrame → 학습용 DataFrame + LabelEncoder.
    표준 컬럼 DataFrame(merge 결과)은 기본값 그대로 사용.
    원본 파일은 실제 컬럼명을 인자로 전달.
    """
    required = [before_col, after_col, summary_col, reason_col, code_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 없음: {missing}\n사용 가능한 컬럼: {list(df.columns)}")

    work = df[required].copy()
    work.columns = STANDARD_COLS

    before_len = len(work)
    work.dropna(subset=["before", "after", "code"], inplace=True)
    logger.info(f"결측치 제거: {before_len} → {len(work)}행")

    for col in ["before", "after", "summary", "reason"]:
        work[col] = work[col].apply(clean_text)

    work = work[(work["before"] != "") & (work["after"] != "")].reset_index(drop=True)

    work["input_text"] = work.apply(
        lambda r: build_input_text(r["before"], r["after"]), axis=1
    )
    work["diff_highlight"] = work.apply(
        lambda r: diff_highlight(r["before"], r["after"]), axis=1
    )

    work["code"] = work["code"].astype(str).str.strip()
    le = LabelEncoder()
    work["code_label"] = le.fit_transform(work["code"])

    logger.info(f"전처리 완료: {len(work)}행, 클래스: {list(le.classes_)}")
    return work, le


def preprocess_merged(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """merge_files() 결과물(표준 컬럼)을 바로 전처리"""
    return preprocess(df, "before", "after", "summary", "reason", "code")


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """train/validation 분할 (stratify by code_label)"""
    try:
        train_df, val_df = train_test_split(
            df, test_size=test_size, random_state=random_state,
            stratify=df["code_label"],
        )
    except ValueError:
        logger.warning("Stratify 불가 → 랜덤 분할")
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    logger.info(f"학습: {len(train_df)}행 / 검증: {len(val_df)}행")
    return train_df, val_df


def compute_class_weights(
    code_labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """클래스 불균형 처리: total / (num_classes × class_count)"""
    weights = np.zeros(num_classes, dtype=np.float32)
    total = len(code_labels)
    for c in range(num_classes):
        count = (code_labels == c).sum()
        weights[c] = total / (num_classes * count) if count > 0 else 1.0
    logger.info(f"클래스 가중치: {dict(enumerate(weights.tolist()))}")
    return weights


# ══════════════════════════════════════════════════════════════
# 통계 리포트
# ══════════════════════════════════════════════════════════════

def build_merge_report(summary: Dict) -> str:
    """merge_files() 반환 summary → 사람이 읽기 쉬운 텍스트"""
    lines = [
        "═" * 50,
        "  데이터 통합 리포트",
        "═" * 50,
        f"  처리 파일   : {summary['success_files']} / {summary['total_files']}개 성공",
        f"  통합 전 데이터: {summary['total_before_dedup']}건",
        f"  중복 제거   : {summary['dedup_removed']}건",
        f"  최종 데이터 : {summary['final_rows']}건",
        "",
    ]
    if summary["error_files"]:
        lines.append("  ─ 오류 파일 ─")
        for fname, msg in summary["error_files"]:
            lines.append(f"    ❌ {fname}: {msg}")
        lines.append("")
    lines.append("  ─ 파일별 결과 ─")
    for s in summary["file_stats"]:
        name = os.path.basename(s["file"])
        lines.append(
            f"    {name}: 원본 {s['original']}행 → "
            f"null-{s['null_removed']} / 길이-{s['length_removed']} → "
            f"최종 {s['final']}행"
        )
    lines.append("═" * 50)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 데이터셋 저장
# ══════════════════════════════════════════════════════════════

def save_dataset(
    df: pd.DataFrame,
    output_path: str = "merged_dataset.xlsx",
    fmt: str = "xlsx",
) -> str:
    """통합 DataFrame을 xlsx 또는 csv로 저장"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if fmt == "csv":
        path = output_path if output_path.endswith(".csv") else output_path + ".csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        path = output_path if output_path.endswith(".xlsx") else output_path + ".xlsx"
        df.to_excel(path, index=False, engine="openpyxl")
    logger.info(f"데이터셋 저장: {path} ({len(df)}행)")
    return path


# ══════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════

import torch
from torch.utils.data import Dataset


class DocChangeDataset(Dataset):
    """
    PyTorch Dataset.
    is_predict=True 이면 input_text 컬럼만 있으면 됨.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_input_len: int = 256,
        max_target_len: int = 128,
        is_predict: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.is_predict = is_predict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["input_text"]),
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if not self.is_predict:
            def _encode(text):
                t = self.tokenizer(
                    str(text),
                    max_length=self.max_target_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                labels = t["input_ids"].squeeze(0)
                labels[labels == self.tokenizer.pad_token_id] = -100
                return labels

            item["summary_labels"] = _encode(row.get("summary", ""))
            item["reason_labels"] = _encode(row.get("reason", ""))
            item["code_labels"] = torch.tensor(int(row["code_label"]), dtype=torch.long)

        return item
