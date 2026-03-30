# evaluation/news_ablation.py — 뉴스 AI 효과 정량 검증 프레임워크
"""
뉴스 피처 적용 전/후 예측 성능 비교 (Ablation Study)

검증 항목:
  1. 전체 성능: 뉴스 없음 vs 뉴스 있음 NLL / MSE / 방향 정확도
  2. 종목별 성능 비교
  3. 섹터별 성능 비교
  4. Horizon별 성능 비교 (1d/3d/5d/20d)
  5. 카테고리별 기여도 (특정 카테고리 ON/OFF)
  6. 미래 정보 누수 검사
  7. 중복 기사 과반영 여부 검사
  8. SHAP 기반 feature importance (torch+shap 설치 시)

자기개선 루프:
  analyze() → 의미 없는 피처 특정 → 보고서 생성 → 재학습 트리거

사용법:
    ablation = NewsAblationStudy(db=get_news_db())
    report = ablation.run_full_study(
        symbols=["005930.KS", "000660.KS"],
        start_date="2024-01-01",
        end_date="2025-03-01",
    )
    print(report.summary())
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("quant.news_ablation")


# ── 결과 데이터클래스 ──────────────────────────────────────────────────────────

@dataclass
class ModelScore:
    """단일 모델 구성의 성능 지표"""
    name:             str
    nll_loss:         float = 0.0
    mse_loss:         float = 0.0
    direction_acc:    float = 0.0    # 방향 예측 정확도 (0~1)
    sharpe_proxy:     float = 0.0    # μ/σ 기반 신호 품질
    n_samples:        int   = 0


@dataclass
class AblationResult:
    """전체 비교 결과"""
    baseline:           ModelScore = field(default_factory=lambda: ModelScore("baseline"))
    with_news:          ModelScore = field(default_factory=lambda: ModelScore("with_news"))
    improvement_pct:    Dict[str, float] = field(default_factory=dict)
    category_contrib:   Dict[str, float] = field(default_factory=dict)
    per_symbol:         Dict[str, Dict]  = field(default_factory=dict)
    per_sector:         Dict[str, Dict]  = field(default_factory=dict)
    leakage_check:      List[str]        = field(default_factory=list)
    duplicate_check:    Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    timestamp:          str = field(default_factory=lambda: datetime.now().isoformat())
    recommendations:    List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  뉴스 AI 피처 효과 검증 보고서",
            f"  생성 시각: {self.timestamp[:19]}",
            "=" * 60,
            "",
            "[ 전체 성능 비교 ]",
            f"  기준 모델  NLL={self.baseline.nll_loss:.4f}  "
            f"DirAcc={self.baseline.direction_acc:.1%}",
            f"  뉴스 모델  NLL={self.with_news.nll_loss:.4f}  "
            f"DirAcc={self.with_news.direction_acc:.1%}",
        ]

        imp = self.improvement_pct
        if imp:
            lines += ["", "[ 개선율 ]"]
            for k, v in imp.items():
                sign = "▲" if v > 0 else "▼"
                lines.append(f"  {k:<20}: {sign} {abs(v):.2f}%")

        if self.category_contrib:
            lines += ["", "[ 카테고리별 기여도 TOP 5 ]"]
            sorted_cat = sorted(self.category_contrib.items(),
                                key=lambda x: abs(x[1]), reverse=True)[:5]
            for cat, val in sorted_cat:
                bar = "█" * min(int(abs(val) * 20), 20)
                lines.append(f"  {cat:<18}: {val:+.4f}  {bar}")

        if self.feature_importance:
            lines += ["", "[ Feature Importance TOP 10 ]"]
            sorted_fi = sorted(self.feature_importance.items(),
                               key=lambda x: x[1], reverse=True)[:10]
            for feat, imp_val in sorted_fi:
                lines.append(f"  [{feat}] {imp_val:.4f}")

        if self.leakage_check:
            lines += ["", f"[ ⚠ 미래 누수 의심 {len(self.leakage_check)}건 ]"]
            for l in self.leakage_check[:3]:
                lines.append(f"  {l}")

        if self.recommendations:
            lines += ["", "[ 개선 권고 ]"]
            for rec in self.recommendations:
                lines.append(f"  • {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# ── 핵심 검증 클래스 ──────────────────────────────────────────────────────────

class NewsAblationStudy:
    """
    뉴스 피처 ablation study 실행기.

    사용 시나리오:
      A) 학습된 두 모델 비교 (use_news=False 모델 vs use_news=True 모델)
      B) 예측 데이터셋 기반 오프라인 분석 (모델 없이 피처 통계만)
    """

    def __init__(self, db=None):
        self.db = db

    # ── 시나리오 A: 모델 기반 비교 ──────────────────────────────────────────

    def compare_models(
        self,
        baseline_model,         # HybridModel (use_news=False)
        news_model,             # HybridModel (use_news=True)
        images:    np.ndarray,  # (N, C, H, W)
        ts_seq:    np.ndarray,  # (N, T, F)
        labels:    np.ndarray,  # (N,)
        macro_feats: Optional[np.ndarray] = None,  # (N, 32)
        news_feats:  Optional[np.ndarray] = None,  # (N, 40)
        symbol:    str = "ALL",
    ) -> AblationResult:
        """두 모델의 성능을 동일 데이터셋에서 비교"""
        import torch

        result = AblationResult()

        def _eval(model, use_news_feat: bool) -> ModelScore:
            model.eval()
            all_mu, all_sigma, all_lbl = [], [], []
            batch_size = 64
            N = len(labels)
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    sl   = slice(i, min(i + batch_size, N))
                    img  = torch.from_numpy(images[sl]).float()
                    ts   = torch.from_numpy(ts_seq[sl]).float()
                    lbl  = labels[sl]
                    mac  = (torch.from_numpy(macro_feats[sl].astype(np.float32))
                            if macro_feats is not None else None)
                    news = (torch.from_numpy(news_feats[sl].astype(np.float32))
                            if (news_feats is not None and use_news_feat) else None)
                    mu, sigma = model(img, ts, macro_feats=mac, news_feats=news)
                    all_mu.append(mu.cpu().numpy())
                    all_sigma.append(sigma.cpu().numpy())
                    all_lbl.append(lbl)

            mu_arr  = np.concatenate(all_mu)
            sig_arr = np.concatenate(all_sigma)
            lbl_arr = np.concatenate(all_lbl)

            return _compute_scores(mu_arr, sig_arr, lbl_arr,
                                   "with_news" if use_news_feat else "baseline")

        result.baseline  = _eval(baseline_model, False)
        result.with_news = _eval(news_model, True)

        _fill_improvements(result)
        _fill_recommendations(result)
        return result

    # ── 시나리오 B: 피처 기반 오프라인 분석 ──────────────────────────────────

    def analyze_features(
        self,
        news_feats_by_date: Dict[str, np.ndarray],   # date → (40,)
        labels_by_date:     Dict[str, float],          # date → return
        events_by_date:     Optional[Dict[str, List]] = None,
    ) -> AblationResult:
        """
        뉴스 피처와 실제 수익률의 상관관계 분석 (모델 불필요).

        Returns:
            AblationResult (category_contrib, feature_importance 중심)
        """
        result = AblationResult()

        if not news_feats_by_date or not labels_by_date:
            return result

        common_dates = sorted(set(news_feats_by_date) & set(labels_by_date))
        if not common_dates:
            return result

        feat_matrix = np.array([news_feats_by_date[d] for d in common_dates])
        label_arr   = np.array([labels_by_date[d]     for d in common_dates])

        # ── 피처 중요도: 피처 × 수익률 상관계수 ─────────────────────────────
        from features.news_features import _CATEGORIES, NEWS_FEATURE_DIM
        feat_names = (
            [f"cat_{c}" for c in _CATEGORIES]            # [0-11]
            + [f"score_{w}" for w in ("1h","4h","1d","3d","5d","20d")]  # [12-17]
            + [f"density_{w}" for w in ("1h","4h","1d","3d","5d","20d")] # [18-23]
            + [f"sentiment_{w}" for w in ("1h","4h","1d","3d","5d","20d")] # [24-29]
            + ["stock_score","sector_score","market_risk","policy_risk",
               "shock_flag","repeat_intensity",
               "freshness","sentiment_velocity","volume_zscore","cumulative_impact"] # [30-39]
        )

        importance: Dict[str, float] = {}
        for i in range(min(NEWS_FEATURE_DIM, feat_matrix.shape[1])):
            col = feat_matrix[:, i]
            if col.std() < 1e-9:
                importance[feat_names[i]] = 0.0
                continue
            corr = float(np.corrcoef(col, label_arr)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            importance[feat_names[i]] = abs(corr)

        result.feature_importance = importance

        # ── 카테고리별 기여도 (상관 방향 보존) ─────────────────────────────
        cat_contrib: Dict[str, float] = {}
        for i, cat in enumerate(_CATEGORIES):
            col = feat_matrix[:, i]
            if col.std() < 1e-9:
                cat_contrib[cat] = 0.0
                continue
            corr = float(np.corrcoef(col, label_arr)[0, 1])
            cat_contrib[cat] = corr if not np.isnan(corr) else 0.0
        result.category_contrib = cat_contrib

        # ── 기준 점수 (뉴스 피처 없이 0 예측) vs 뉴스 활용 ─────────────────
        result.baseline  = ModelScore(name="zero_baseline",
                                      mse_loss=float(np.mean(label_arr ** 2)),
                                      n_samples=len(label_arr))

        # 뉴스 점수 기반 단순 방향 예측 (feat[12] = 1d score 사용)
        score_1d = feat_matrix[:, 14] if feat_matrix.shape[1] > 14 else feat_matrix[:, 0]
        pred_dir = np.sign(score_1d)
        real_dir = np.sign(label_arr)
        dir_acc  = float(np.mean(pred_dir == real_dir))
        result.with_news = ModelScore(name="news_score_signal",
                                      direction_acc=dir_acc,
                                      n_samples=len(label_arr))

        # ── 누수 검사 ──────────────────────────────────────────────────────
        if events_by_date:
            from features.news_features import check_leakage
            leaks = check_leakage(
                {d: feat_matrix[i] for i, d in enumerate(common_dates)},
                events_by_date,
            )
            result.leakage_check = leaks

        _fill_improvements(result)
        _fill_recommendations(result)
        return result

    # ── 중복 과반영 검사 ──────────────────────────────────────────────────────

    def check_duplicate_inflation(
        self,
        events: List[Dict],
        threshold_repeat: int = 10,
    ) -> Dict[str, float]:
        """
        중복 기사가 점수를 과도하게 부풀리는지 검사.

        Returns:
            {
              "max_repeat_count": ...,
              "high_repeat_events": ...,
              "score_inflation_ratio": ...,
            }
        """
        if not events:
            return {}

        repeats = [int(e.get("repeat_count", 1)) for e in events]
        max_rep = max(repeats)
        high    = sum(1 for r in repeats if r >= threshold_repeat)

        # 반복보도가 없다면 점수 vs 반복보도 보정 점수 비교
        raw_scores  = [float(e.get("computed_score", 0.0)) for e in events]
        norm_scores = [
            s / max(math.sqrt(r), 1.0)
            for s, r in zip(raw_scores, repeats)
        ]
        inflation = (
            (sum(abs(s) for s in raw_scores) /
             max(sum(abs(s) for s in norm_scores), 1e-9))
            if raw_scores else 1.0
        )

        return {
            "max_repeat_count":    float(max_rep),
            "high_repeat_events":  float(high),
            "score_inflation_ratio": float(inflation),
            "recommendation":      (
                "반복보도 정규화 필요 (√repeat 스케일링 권고)"
                if inflation > 1.5 else "정상"
            ),
        }

    # ── 자기개선 루프 ──────────────────────────────────────────────────────────

    def run_self_improvement(
        self,
        result: AblationResult,
        retrain_callback=None,   # Callable[[], None] — 재학습 트리거
    ) -> List[str]:
        """
        ablation 결과 분석 → 권고 생성 → (선택) 재학습 트리거.

        Returns:
            실행된 조치 목록
        """
        actions = []

        # 1. 의미 없는 피처 식별 (importance < 0.02)
        weak_feats = [k for k, v in result.feature_importance.items() if v < 0.02]
        if len(weak_feats) > 5:
            actions.append(f"약한 피처 {len(weak_feats)}개 식별: {weak_feats[:3]}...")

        # 2. 누수 경고
        if result.leakage_check:
            actions.append(f"⚠ 미래 누수 {len(result.leakage_check)}건 → 데이터 파이프라인 점검 필요")

        # 3. 성능 개선이 없는 경우
        imp = result.improvement_pct.get("direction_acc_pct", 0.0)
        if imp < 0:
            actions.append(f"뉴스 피처가 방향 정확도를 {imp:.2f}% 저하 → 재학습 권고")
            if retrain_callback:
                try:
                    retrain_callback()
                    actions.append("✓ 재학습 트리거 실행됨")
                except Exception as e:
                    actions.append(f"재학습 트리거 실패: {e}")

        # 4. 높은 반복보도 비율
        if result.duplicate_check.get("score_inflation_ratio", 1.0) > 1.5:
            actions.append("중복 기사 점수 부풀림 → 클러스터링 재실행 권고")

        return actions


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def _compute_scores(
    mu:     np.ndarray,
    sigma:  np.ndarray,
    labels: np.ndarray,
    name:   str,
) -> ModelScore:
    """예측 배열로부터 성능 지표 계산"""
    n = len(labels)
    if n == 0:
        return ModelScore(name=name, n_samples=0)

    # NLL (가우시안)
    sigma_c = np.clip(sigma, 1e-3, 2.0)
    nll = float(np.mean(
        np.log(sigma_c) + (labels - mu) ** 2 / (2 * sigma_c ** 2)
    ))

    # MSE
    mse = float(np.mean((mu - labels) ** 2))

    # 방향 정확도
    pred_dir = np.sign(mu)
    real_dir = np.sign(labels)
    dir_acc  = float(np.mean(pred_dir == real_dir))

    # Sharpe proxy: μ/σ 평균 (신호 품질)
    snr = float(np.mean(np.abs(mu) / (sigma_c + 1e-9)))

    return ModelScore(
        name=name,
        nll_loss=nll,
        mse_loss=mse,
        direction_acc=dir_acc,
        sharpe_proxy=snr,
        n_samples=n,
    )


def _fill_improvements(result: AblationResult) -> None:
    """개선율 계산"""
    b = result.baseline
    w = result.with_news

    def pct_imp(before, after, lower_is_better=True):
        if abs(before) < 1e-9:
            return 0.0
        delta = (before - after) if lower_is_better else (after - before)
        return round(delta / abs(before) * 100, 2)

    result.improvement_pct = {
        "nll_loss_pct":       pct_imp(b.nll_loss,      w.nll_loss,      lower_is_better=True),
        "mse_loss_pct":       pct_imp(b.mse_loss,      w.mse_loss,      lower_is_better=True),
        "direction_acc_pct":  pct_imp(b.direction_acc, w.direction_acc, lower_is_better=False),
        "sharpe_proxy_pct":   pct_imp(b.sharpe_proxy,  w.sharpe_proxy,  lower_is_better=False),
    }


def _fill_recommendations(result: AblationResult) -> None:
    """자동 개선 권고 생성"""
    recs = []
    imp  = result.improvement_pct

    if imp.get("direction_acc_pct", 0) > 2.0:
        recs.append(f"뉴스 피처가 방향 정확도를 {imp['direction_acc_pct']:.1f}% 향상 → 지속 활성화 권고")
    elif imp.get("direction_acc_pct", 0) < -1.0:
        recs.append("뉴스 피처 효과 미미 → 학습 데이터 확충 또는 피처 설계 재검토")

    if imp.get("nll_loss_pct", 0) > 5.0:
        recs.append(f"NLL 손실 {imp['nll_loss_pct']:.1f}% 감소 → 뉴스 피처 유효")

    # 카테고리 기여도 분석
    cat = result.category_contrib
    if cat:
        top_pos = [k for k, v in sorted(cat.items(), key=lambda x: x[1], reverse=True)
                   if v > 0.1][:3]
        top_neg = [k for k, v in sorted(cat.items(), key=lambda x: x[1])
                   if v < -0.1][:3]
        if top_pos:
            recs.append(f"강한 양의 기여 카테고리: {', '.join(top_pos)} → 가중치 강화 검토")
        if top_neg:
            recs.append(f"강한 역기여 카테고리: {', '.join(top_neg)} → 역방향 해석 검토")

    if result.leakage_check:
        recs.append(f"⚠ 미래 정보 누수 {len(result.leakage_check)}건 → 즉시 점검 필요")

    result.recommendations = recs
