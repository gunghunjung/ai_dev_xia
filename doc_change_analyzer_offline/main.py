"""
main.py ─ 오프라인 문서 변경 분석 AI  (완전 CLI 모드)

사용법:
  # ① 단일 파일 학습
  python main.py train \\
      --input_file ./data/sample.xlsx \\
      --before_col 변경전 --after_col 변경후 \\
      --summary_col 요약 --reason_col 사유 --code_col 코드 \\
      --model_path ./local_model/kobart \\
      --save_path ./saved_model/v1

  # ② 디렉터리 학습 (다중 파일 자동 통합)
  python main.py train \\
      --input_dir ./data \\
      --before_col 변경전 --after_col 변경후 \\
      --summary_col 요약 --reason_col 사유 --code_col 코드 \\
      --model_path ./local_model/kobart \\
      --save_path ./saved_model/v1

  # ③ 증분 학습 (기존 모델에 추가 데이터)
  python main.py train \\
      --input_file ./data/new.xlsx \\
      --before_col 변경전 --after_col 변경후 \\
      --summary_col 요약 --reason_col 사유 --code_col 코드 \\
      --model_path ./saved_model/v1 \\
      --save_path ./saved_model/v1 \\
      --incremental

  # ④ 단일 파일 예측
  python main.py predict \\
      --model_path ./saved_model/v1 \\
      --input_file ./data/new.xlsx \\
      --output_file ./predictions/result.xlsx

  # ⑤ 디렉터리 예측 (다중 파일)
  python main.py predict \\
      --model_path ./saved_model/v1 \\
      --input_dir ./data/predict \\
      --output_mode separate \\
      --output_dir ./predictions

  # ⑥ 저장 모델 목록 조회
  python main.py list --models_root ./saved_model

  # ⑦ 모델 상세 정보
  python main.py info --model_path ./saved_model/v1
"""

import os
import sys
import logging
import argparse
import traceback

# ─────────────────────────────────────────────────────────────────
# 오프라인 강제 설정 (import 전에 먼저 설정)
# ─────────────────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ─────────────────────────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────────────────────────

def _setup_logging(log_level: str = "INFO", log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
        force=True,
    )

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 서브커맨드: train
# ═══════════════════════════════════════════════════════════════════

def cmd_train(args):
    """학습 명령 처리 (신규 학습 / 증분 학습)"""
    import pandas as pd
    from utils import (
        load_excel, load_excel_dir,
        merge_files, build_merge_report,
        save_dataset,
    )
    from train import train as do_train, incremental_train
    from model_loader import print_model_info

    col_map = {
        "before":  args.before_col,
        "after":   args.after_col,
        "summary": args.summary_col,
        "reason":  args.reason_col,
        "code":    args.code_col,
    }

    # ── 데이터 로드 ──────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  [데이터 로드]")
    print("═" * 60)

    if args.input_file:
        # 단일 파일
        if not os.path.exists(args.input_file):
            print(f"  ❌ 파일 없음: {args.input_file}")
            sys.exit(1)
        df = load_excel(args.input_file)
        print(f"  파일 로드: {args.input_file} ({len(df)}행)")

    elif args.input_dir:
        # 디렉터리 → 다중 파일 통합
        if not os.path.isdir(args.input_dir):
            print(f"  ❌ 디렉터리 없음: {args.input_dir}")
            sys.exit(1)
        loaded = load_excel_dir(args.input_dir, progress=True)
        ok = [(fp, d, "") for fp, d, e in loaded if d is not None]
        if not ok:
            print(f"  ❌ 로드 성공 파일 없음: {args.input_dir}")
            sys.exit(1)

        # 각 파일에 동일 col_map 적용
        triples = [(d, col_map, os.path.basename(fp)) for fp, d, _ in ok]
        df, merge_summary = merge_files(triples, dedup_exact=True)
        print(build_merge_report(merge_summary))

        # 통합 데이터셋 저장 (옵션)
        if args.save_merged:
            merged_path = os.path.join(args.save_path, "merged_dataset.xlsx")
            save_dataset(df, merged_path)
            print(f"  통합 데이터셋 저장: {merged_path}")
    else:
        print("  ❌ --input_file 또는 --input_dir 중 하나를 지정하세요.")
        sys.exit(1)

    print(f"\n  최종 학습 데이터: {len(df)}행\n")

    # ── 학습 / 증분 학습 ───────────────────────────────────
    if args.incremental:
        print("  [모드] 증분 학습 (Incremental Fine-tuning)")
        result = incremental_train(
            new_df=df,
            col_map=col_map,
            model_dir=args.model_path,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
            patience=args.patience,
        )
    else:
        print("  [모드] 신규 학습")
        result = do_train(
            df=df,
            col_map=col_map,
            base_model_path=args.model_path,
            save_path=args.save_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_input_len=args.max_input_len,
            max_target_len=args.max_target_len,
            patience=args.patience,
        )

    # ── 학습 이력 출력 ──────────────────────────────────────
    history = result.get("history", [])
    if history:
        print("\n" + "─" * 60)
        print("  Epoch 학습 이력")
        print("─" * 60)
        print(f"  {'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
        for h in history:
            print(
                f"  {h['epoch']:>6}  "
                f"{h['train_loss']:>11.4f}  "
                f"{h['train_acc']:>9.3f}  "
                f"{h['val_loss']:>9.4f}  "
                f"{h['val_acc']:>8.3f}"
            )
        print("─" * 60)

    print(f"\n  Best Val Loss : {result['best_val_loss']:.4f}")
    print(f"  저장 경로     : {result['save_path']}")
    print_model_info(result["save_path"])


# ═══════════════════════════════════════════════════════════════════
# 서브커맨드: predict
# ═══════════════════════════════════════════════════════════════════

def cmd_predict(args):
    """예측 명령 처리 (단일 파일 / 디렉터리)"""
    from predict import predict_single, predict_dir

    if not os.path.isdir(args.model_path):
        print(f"  ❌ 모델 경로 없음: {args.model_path}")
        sys.exit(1)

    if args.input_file:
        # 단일 파일 예측
        if not os.path.exists(args.input_file):
            print(f"  ❌ 파일 없음: {args.input_file}")
            sys.exit(1)
        result_df = predict_single(
            model_path=args.model_path,
            input_file=args.input_file,
            output_file=args.output_file,
            before_col=args.before_col or None,
            after_col=args.after_col or None,
            batch_size=args.batch_size,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        _print_prediction_sample(result_df)

    elif args.input_dir:
        # 다중 파일 예측
        if not os.path.isdir(args.input_dir):
            print(f"  ❌ 디렉터리 없음: {args.input_dir}")
            sys.exit(1)
        results, combined_df = predict_dir(
            model_path=args.model_path,
            input_dir=args.input_dir,
            output_mode=args.output_mode,
            output_dir=args.output_dir,
            output_file=args.output_file,
            before_col=args.before_col or None,
            after_col=args.after_col or None,
            batch_size=args.batch_size,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        if combined_df is not None:
            _print_prediction_sample(combined_df)
    else:
        print("  ❌ --input_file 또는 --input_dir 중 하나를 지정하세요.")
        sys.exit(1)


def _print_prediction_sample(df, n: int = 3):
    """예측 결과 샘플 콘솔 출력"""
    cols = ["pred_summary", "pred_reason", "pred_code", "confidence_score"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return
    print("\n" + "─" * 60)
    print(f"  예측 결과 샘플 (최대 {n}행)")
    print("─" * 60)
    sample = df[available].head(n)
    for i, row in sample.iterrows():
        print(f"\n  [{i+1}번 행]")
        if "pred_summary" in row:
            print(f"    요약     : {str(row['pred_summary'])[:80]}")
        if "pred_reason" in row:
            print(f"    사유     : {str(row['pred_reason'])[:80]}")
        if "pred_code" in row:
            print(f"    코드     : {row['pred_code']}")
        if "confidence_score" in row:
            print(f"    신뢰도   : {row['confidence_score']:.1%}")
    print("─" * 60)


# ═══════════════════════════════════════════════════════════════════
# 서브커맨드: list
# ═══════════════════════════════════════════════════════════════════

def cmd_list(args):
    """저장된 모델 목록 출력"""
    from model_loader import print_model_list
    print_model_list(args.models_root)


# ═══════════════════════════════════════════════════════════════════
# 서브커맨드: info
# ═══════════════════════════════════════════════════════════════════

def cmd_info(args):
    """단일 모델 상세 정보 출력"""
    from model_loader import print_model_info
    print_model_info(args.model_path)


# ═══════════════════════════════════════════════════════════════════
# ArgumentParser 구성
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="오프라인 문서 변경 분석 AI (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="로그 레벨 (기본: INFO)",
    )
    parser.add_argument("--log_file", default=None, help="로그 파일 경로 (선택)")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ─────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="모델 학습 (신규 / 증분)")

    # 입력
    inp = p_train.add_mutually_exclusive_group()
    inp.add_argument("--input_file", help="단일 엑셀 파일 경로")
    inp.add_argument("--input_dir",  help="여러 엑셀 파일이 있는 디렉터리 경로")

    # 컬럼 매핑
    p_train.add_argument("--before_col",  default="변경전", help="변경 전 컬럼명 (기본: 변경전)")
    p_train.add_argument("--after_col",   default="변경후", help="변경 후 컬럼명 (기본: 변경후)")
    p_train.add_argument("--summary_col", default="요약",   help="요약 컬럼명 (기본: 요약)")
    p_train.add_argument("--reason_col",  default="사유",   help="사유 컬럼명 (기본: 사유)")
    p_train.add_argument("--code_col",    default="코드",   help="코드 컬럼명 (기본: 코드)")

    # 모델 경로
    p_train.add_argument("--model_path",  required=True,
                         help="베이스 모델 로컬 경로 또는 기존 저장 모델 경로 (--incremental 시)")
    p_train.add_argument("--save_path",   default="./saved_model/model_v1",
                         help="학습 결과 저장 경로 (기본: ./saved_model/model_v1)")

    # 학습 파라미터
    p_train.add_argument("--epochs",        type=int,   default=10,   help="에폭 수 (기본: 10)")
    p_train.add_argument("--batch_size",    type=int,   default=8,    help="배치 크기 (기본: 8)")
    p_train.add_argument("--lr",            type=float, default=3e-5, help="학습률 (기본: 3e-5)")
    p_train.add_argument("--max_input_len", type=int,   default=256,  help="입력 최대 토큰 (기본: 256)")
    p_train.add_argument("--max_target_len",type=int,   default=128,  help="생성 최대 토큰 (기본: 128)")
    p_train.add_argument("--patience",      type=int,   default=3,    help="Early Stopping patience (기본: 3)")

    # 모드
    p_train.add_argument("--incremental", action="store_true",
                         help="기존 모델에 추가 데이터로 증분 학습")
    p_train.add_argument("--save_merged", action="store_true",
                         help="다중 파일 통합 시 merged_dataset.xlsx 저장")

    # ── predict ───────────────────────────────────────────────
    p_pred = sub.add_parser("predict", help="저장된 모델로 예측")

    # 입력
    pinp = p_pred.add_mutually_exclusive_group()
    pinp.add_argument("--input_file", help="예측할 단일 엑셀 파일")
    pinp.add_argument("--input_dir",  help="예측할 엑셀 파일들이 있는 디렉터리")

    p_pred.add_argument("--model_path", required=True, help="저장된 모델 경로")
    p_pred.add_argument("--output_file", default="./predictions/prediction_result.xlsx",
                        help="결과 저장 파일 경로 (단일 파일 또는 combined 모드)")
    p_pred.add_argument("--output_dir",  default="./predictions",
                        help="결과 저장 디렉터리 (separate 모드)")
    p_pred.add_argument("--output_mode", default="separate",
                        choices=["separate", "combined"],
                        help="다중 파일 출력 방식 (기본: separate)")

    # 컬럼 (config 자동 참조 가능 → 선택)
    p_pred.add_argument("--before_col", default="", help="변경 전 컬럼명 (미지정 시 config 자동)")
    p_pred.add_argument("--after_col",  default="", help="변경 후 컬럼명 (미지정 시 config 자동)")

    # 예측 파라미터
    p_pred.add_argument("--batch_size",     type=int, default=8,   help="배치 크기 (기본: 8)")
    p_pred.add_argument("--max_input_len",  type=int, default=256, help="입력 최대 토큰 (기본: 256)")
    p_pred.add_argument("--max_new_tokens", type=int, default=128, help="생성 최대 토큰 (기본: 128)")
    p_pred.add_argument("--num_beams",      type=int, default=4,   help="Beam search 크기 (기본: 4)")

    # ── list ──────────────────────────────────────────────────
    p_list = sub.add_parser("list", help="저장된 모델 목록 조회")
    p_list.add_argument("--models_root", default="./saved_model",
                        help="모델 루트 경로 (기본: ./saved_model)")

    # ── info ──────────────────────────────────────────────────
    p_info = sub.add_parser("info", help="모델 상세 정보 출력")
    p_info.add_argument("--model_path", required=True, help="모델 경로")

    return parser


# ═══════════════════════════════════════════════════════════════════
# 진입점
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.log_level, args.log_file)
    logger.info(f"명령: {args.command} | args: {vars(args)}")

    print("\n" + "═" * 60)
    print("  오프라인 문서 변경 분석 AI")
    print(f"  명령: {args.command.upper()}")
    print("═" * 60)

    try:
        if args.command == "train":
            cmd_train(args)
        elif args.command == "predict":
            cmd_predict(args)
        elif args.command == "list":
            cmd_list(args)
        elif args.command == "info":
            cmd_info(args)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n\n  ⛔ 사용자 중단 (Ctrl+C)")
        sys.exit(0)

    except OSError as e:
        # 오프라인 모델 없음 등 설정 오류
        print(f"\n  ❌ [설정 오류] {e}")
        logger.error(f"OSError: {e}", exc_info=True)
        sys.exit(2)

    except ValueError as e:
        # 컬럼 없음, 데이터 없음 등 입력 오류
        print(f"\n  ❌ [입력 오류] {e}")
        logger.error(f"ValueError: {e}", exc_info=True)
        sys.exit(2)

    except Exception as e:
        print(f"\n  ❌ [예기치 않은 오류] {type(e).__name__}: {e}")
        print("  상세 정보:")
        traceback.print_exc()
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
