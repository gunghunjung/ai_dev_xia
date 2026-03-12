"""
ReportGenerator — 사람이 읽을 수 있는 예측 보고서 생성
──────────────────────────────────────────────────────────
입력: ForecastEngine 출력 dict
출력: 구조화된 텍스트 보고서 (GUI 표시 또는 파일 저장)

보고서 섹션:
  1. 종목 기본 정보
  2. 현재 시장 국면 판단
  3. 예측 요약 (방향/확률/신뢰구간)
  4. 다중 시계 예측 표
  5. 주요 예측 근거 (SHAP/피처 중요도)
  6. 리스크 요인
  7. 종합 판단 및 고지 사항

⚠ "투자 권유"가 아닌 "정보 제공" 목적임을 항상 명시.
⚠ 모든 예측은 과거 패턴 기반 확률 추정이며 미래를 보장하지 않음.
"""
from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Any
from ..logger_config import get_logger

log = get_logger("explain.report")

_DISCLAIMER = (
    "\n" + "═" * 60 + "\n"
    "⚠ 법적 고지 사항\n"
    "이 보고서는 과거 가격 데이터와 AI 모델에 기반한\n"
    "확률적 정보 제공 목적이며, 투자 권유가 아닙니다.\n"
    "모든 예측은 미래를 보장하지 않으며,\n"
    "투자 결정은 반드시 본인의 판단으로 이루어져야 합니다.\n"
    "═" * 60
)


class ReportGenerator:
    """
    Parameters
    ----------
    symbol      : 종목 코드
    company_name: 회사명 (있으면)
    currency    : 통화 (USD, KRW 등)
    """

    def __init__(
        self,
        symbol:       str = "UNKNOWN",
        company_name: str = "",
        currency:     str = "USD",
    ) -> None:
        self._symbol  = symbol
        self._cname   = company_name or symbol
        self._cur     = currency

    # ── 메인 보고서 ───────────────────────────────────────────────────────
    def generate(
        self,
        forecast:      Dict[str, Any],
        current_price: float = 0.0,
        shap_info:     Optional[Dict[str, Any]] = None,
        perf_metrics:  Optional[Dict[str, float]] = None,
    ) -> str:
        """
        완전한 예측 보고서 생성.

        Parameters
        ----------
        forecast      : ForecastEngine.forecast() 반환값
        current_price : 현재 주가
        shap_info     : SHAPExplainer.local_explanation() 반환값 (선택)
        perf_metrics  : 백테스트/검증 성능 (선택)
        """
        now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = []

        lines += self._header(now, current_price)
        lines += self._regime_section(forecast)
        lines += self._prediction_summary(forecast, current_price)
        lines += self._multi_horizon_table(forecast)
        lines += self._risk_section(forecast)
        if shap_info:
            lines += self._shap_section(shap_info)
        if perf_metrics:
            lines += self._performance_section(perf_metrics)
        lines += self._final_verdict(forecast)
        lines.append(_DISCLAIMER)

        return "\n".join(lines)

    # ── 섹션별 생성 ───────────────────────────────────────────────────────
    def _header(self, now: str, price: float) -> List[str]:
        price_str = f"{price:,.2f} {self._cur}" if price else "N/A"
        return [
            "═" * 60,
            f" AI 주가 예측 보고서",
            f" 종목  : {self._cname} ({self._symbol})",
            f" 현재가: {price_str}",
            f" 생성일: {now}",
            "═" * 60,
            "",
        ]

    def _regime_section(self, fc: Dict[str, Any]) -> List[str]:
        regime = fc.get("regime_label", "감지 불가")
        return [
            "【1. 시장 국면 판단】",
            f"  현재 국면: {regime}",
            "",
        ]

    def _prediction_summary(self, fc: Dict[str, Any], price: float) -> List[str]:
        prob_up   = fc.get("prob_up",   float("nan"))
        prob_down = fc.get("prob_down", float("nan"))
        point     = fc.get("point_return", float("nan"))
        ci90_l    = fc.get("ci_90_lower", float("nan"))
        ci90_u    = fc.get("ci_90_upper", float("nan"))
        ci50_l    = fc.get("ci_50_lower", float("nan"))
        ci50_u    = fc.get("ci_50_upper", float("nan"))
        uncertainty = fc.get("uncertainty", float("nan"))
        vol        = fc.get("forecast_volatility", float("nan"))

        target_p  = fc.get("target_prob",    float("nan"))
        stop_p    = fc.get("stop_loss_prob", float("nan"))
        t_pct     = fc.get("target_pct",     0.05)
        s_pct     = fc.get("stop_loss_pct", -0.03)

        lines = [
            "【2. 예측 요약】",
            f"  방향 확률:   ↑ {prob_up:.1%}  /  ↓ {prob_down:.1%}",
            f"  예측 수익률: {point:+.2%}",
            f"  90% 신뢰구간: [{ci90_l:+.2%}, {ci90_u:+.2%}]",
            f"  50% 신뢰구간: [{ci50_l:+.2%}, {ci50_u:+.2%}]",
            f"  예측 불확실성 (σ): {uncertainty:.4f}",
            f"  예측 변동성 (연율): {vol:.1%}" if not self._is_nan(vol) else "",
            "",
            f"  목표가 +{t_pct:.0%} 도달 확률: {target_p:.1%}",
            f"  손절선 {s_pct:.0%} 이탈 확률: {stop_p:.1%}",
            "",
        ]
        if price:
            t_price = fc.get("target_price")
            s_price = fc.get("stop_loss_price")
            if t_price:
                lines.insert(-1, f"  목표가 (절대): {t_price:,.2f} {self._cur}")
            if s_price:
                lines.insert(-1, f"  손절가 (절대): {s_price:,.2f} {self._cur}")
        return [l for l in lines if l is not None]

    def _multi_horizon_table(self, fc: Dict[str, Any]) -> List[str]:
        mh = fc.get("multi_horizon", {})
        if not mh:
            return []
        lines = [
            "【3. 다중 시계 예측 (단순 외삽 근사)】",
            f"  {'기간':<8} {'예측수익률':>12} {'90%CI 하단':>12} {'90%CI 상단':>12} {'상승확률':>10}",
            "  " + "-" * 58,
        ]
        for h, info in sorted(mh.items()):
            lines.append(
                f"  {h:>2}일 후  "
                f"  {info.get('point', float('nan')):>+10.2%}  "
                f"  {info.get('lower_90', float('nan')):>+10.2%}  "
                f"  {info.get('upper_90', float('nan')):>+10.2%}  "
                f"  {info.get('prob_up', 0.5):>8.1%}"
            )
        lines.append("")
        return lines

    def _risk_section(self, fc: Dict[str, Any]) -> List[str]:
        warnings = fc.get("warnings", [])
        drift    = fc.get("drift", {})
        lines = ["【4. 리스크 요인】"]
        if warnings:
            for w in warnings:
                lines.append(f"  {w}")
        else:
            lines.append("  특이 리스크 없음 (현재 시점 기준)")
        if drift.get("is_drifted"):
            lines.append(
                f"  ⚠ 분포 변화 감지 (KS={drift.get('ks_stat',0):.3f}, "
                f"PSI={drift.get('psi',0):.3f})"
            )
        lines.append("")
        return lines

    def _shap_section(self, shap_info: Dict[str, Any]) -> List[str]:
        lines = ["【5. 예측 근거 (주요 피처 기여)】"]
        if "error" in shap_info:
            lines.append(f"  {shap_info['error']}")
            lines.append("")
            return lines

        top_pos = shap_info.get("top_positive", [])
        top_neg = shap_info.get("top_negative", [])

        if top_pos:
            lines.append("  📈 상승 기여 요인:")
            for name, val in top_pos[:5]:
                lines.append(f"      {name:<25} +{val:.4f}")
        if top_neg:
            lines.append("  📉 하락 기여 요인:")
            for name, val in top_neg[:5]:
                lines.append(f"      {name:<25} {val:.4f}")

        txt = shap_info.get("explanation_text", "")
        if txt:
            lines.append("")
            lines.append(f"  요약: {txt.split(chr(10))[0]}")

        lines.append("")
        return lines

    def _performance_section(self, metrics: Dict[str, float]) -> List[str]:
        lines = ["【6. 모델 백테스트 성능 요약】"]
        items = [
            ("DA (방향정확도)", metrics.get("da", float("nan")), ".1%"),
            ("Sharpe",          metrics.get("sharpe", float("nan")), ".2f"),
            ("Sortino",         metrics.get("sortino", float("nan")), ".2f"),
            ("Max Drawdown",    metrics.get("max_drawdown", float("nan")), ".1%"),
            ("누적수익률",      metrics.get("cumulative_return", float("nan")), ".1%"),
            ("RMSE",            metrics.get("rmse", float("nan")), ".5f"),
        ]
        for name, val, fmt in items:
            if not self._is_nan(val):
                lines.append(f"  {name:<20}: {val:{fmt}}")
        lines.append("")
        return lines

    def _final_verdict(self, fc: Dict[str, Any]) -> List[str]:
        signal = fc.get("signal", "N/A")
        unc    = fc.get("uncertainty", 0)
        drift  = fc.get("drift", {})

        lines = [
            "【7. 종합 판단】",
            f"  신호: {signal}",
        ]

        if unc > 0.02:
            lines.append("  ⚠ 불확실성이 높아 적극적 포지션 진입은 위험합니다.")
        if drift.get("is_drifted"):
            lines.append("  ⚠ 현재 시장이 학습 데이터와 다른 특성을 보입니다. 모델 재학습 권장.")
        lines.append("")
        return lines

    # ── 유틸 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_nan(v: Any) -> bool:
        try:
            import math
            return math.isnan(float(v))
        except Exception:
            return True

    def short_summary(self, fc: Dict[str, Any]) -> str:
        """GUI 상단 표시용 한 줄 요약"""
        sig  = fc.get("signal", "N/A")
        pu   = fc.get("prob_up", 0.5)
        pd_  = fc.get("prob_down", 0.5)
        ret  = fc.get("point_return", 0.0)
        unc  = fc.get("uncertainty", 0.0)
        return (
            f"{sig}  |  ↑{pu:.0%} / ↓{pd_:.0%}  |  "
            f"예측수익률 {ret:+.2%}  |  불확실성 σ={unc:.4f}"
        )
