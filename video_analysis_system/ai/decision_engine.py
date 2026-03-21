"""
ai/decision_engine.py — Combines AI scores + temporal analysis + rules
                         to produce the final SystemState.

Design principle:
  AI prediction alone does NOT produce the final answer.
  Rules and temporal validation always have the final say.

Decision pipeline per frame:
  1. Evaluate each IDecisionRule against FrameContext.
  2. Aggregate rule results into a combined_score.
  3. Apply hysteresis (consecutive frame counting).
  4. Update per-ROI state.
  5. Write system_state to FrameContext.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from config import DecisionConfig
from core.data_models import FrameContext, ROIData, RuleResult, TemporalSummary
from core.interfaces import IDecisionRule
from core.states import AbnormalityType, ROIState, SystemState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in decision rules
# ---------------------------------------------------------------------------

class AIScoreRule(IDecisionRule):
    """Fires when a ROI's AI confidence for 'abnormal' exceeds threshold."""

    @property
    def name(self) -> str:
        return "ai_score_rule"

    def evaluate(self, ctx: FrameContext) -> Optional[RuleResult]:
        max_score = 0.0
        for roi in ctx.rois:
            if roi.detection and roi.detection.is_abnormal:
                max_score = max(max_score, roi.detection.confidence)
        if max_score > 0:
            return RuleResult(
                rule_name=self.name,
                triggered=True,
                severity=max_score,
                abnormality_type=AbnormalityType.AI_DETECTION,
                message=f"AI abnormal confidence={max_score:.2f}",
            )
        return None


class SuddenChangeRule(IDecisionRule):
    """Fires on any ROI with a large frame-to-frame intensity delta."""

    def __init__(self, threshold: float = 40.0):
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "sudden_change_rule"

    def evaluate(self, ctx: FrameContext) -> Optional[RuleResult]:
        for roi in ctx.rois:
            delta = roi.features.get("mean_abs_diff", 0.0)
            if delta > self._threshold:
                severity = min(delta / (self._threshold * 2), 1.0)
                return RuleResult(
                    rule_name=self.name,
                    triggered=True,
                    severity=severity,
                    abnormality_type=AbnormalityType.SUDDEN_CHANGE,
                    message=f"ROI {roi.roi_id}: sudden change delta={delta:.1f}",
                )
        return None


class StuckRule(IDecisionRule):
    """Fires when temporal summary reports a stuck ROI."""

    @property
    def name(self) -> str:
        return "stuck_rule"

    def evaluate(self, ctx: FrameContext) -> Optional[RuleResult]:
        for roi in ctx.rois:
            if roi.temporal_summary and roi.temporal_summary.is_stuck:
                return RuleResult(
                    rule_name=self.name,
                    triggered=True,
                    severity=0.8,
                    abnormality_type=AbnormalityType.STUCK,
                    message=f"ROI {roi.roi_id} appears stuck (std={roi.temporal_summary.recent_std:.2f})",
                )
        return None


class DriftRule(IDecisionRule):
    """Fires when temporal summary reports a drifting ROI."""

    @property
    def name(self) -> str:
        return "drift_rule"

    def evaluate(self, ctx: FrameContext) -> Optional[RuleResult]:
        for roi in ctx.rois:
            if roi.temporal_summary and roi.temporal_summary.is_drifting:
                slope = roi.temporal_summary.trend_slope
                severity = min(abs(slope) / 5.0, 1.0)
                return RuleResult(
                    rule_name=self.name,
                    triggered=True,
                    severity=severity,
                    abnormality_type=AbnormalityType.DRIFT,
                    message=f"ROI {roi.roi_id} drifting, slope={slope:.2f}",
                )
        return None


class OscillationRule(IDecisionRule):
    """Fires when temporal summary reports oscillation."""

    @property
    def name(self) -> str:
        return "oscillation_rule"

    def evaluate(self, ctx: FrameContext) -> Optional[RuleResult]:
        for roi in ctx.rois:
            if roi.temporal_summary and roi.temporal_summary.is_oscillating:
                amp = roi.temporal_summary.oscillation_amplitude
                return RuleResult(
                    rule_name=self.name,
                    triggered=True,
                    severity=min(amp / 50.0, 1.0),
                    abnormality_type=AbnormalityType.OSCILLATION,
                    message=f"ROI {roi.roi_id} oscillating, amplitude={amp:.1f}",
                )
        return None


class LowConfidenceWarningRule(IDecisionRule):
    """Issues a WARNING when AI confidence is in the marginal range."""

    def __init__(self, warning_lo: float = 0.35, warning_hi: float = 0.60):
        self._lo = warning_lo
        self._hi = warning_hi

    @property
    def name(self) -> str:
        return "low_confidence_warning"

    def evaluate(self, ctx: FrameContext) -> Optional[RuleResult]:
        for roi in ctx.rois:
            if roi.detection:
                conf = roi.detection.confidence
                if self._lo < conf < self._hi:
                    return RuleResult(
                        rule_name=self.name,
                        triggered=True,
                        severity=conf,
                        message=f"ROI {roi.roi_id}: marginal confidence={conf:.2f}",
                    )
        return None


# ---------------------------------------------------------------------------
# ROI state mapper
# ---------------------------------------------------------------------------

def _derive_roi_state(roi: ROIData, cfg: DecisionConfig) -> ROIState:
    """Combine temporal summary + detection result into a per-ROI state."""
    ts = roi.temporal_summary

    if ts is not None:
        if ts.has_sudden_change:
            return ROIState.SUDDEN_CHANGE
        if ts.is_stuck:
            return ROIState.STUCK
        if ts.is_oscillating:
            return ROIState.OSCILLATING
        if ts.is_drifting:
            return ROIState.DRIFTING

    if roi.detection:
        if roi.detection.is_abnormal and roi.detection.confidence >= cfg.abnormal_score_threshold:
            return ROIState.ABNORMAL
        if roi.detection.confidence >= cfg.warning_score_threshold:
            return ROIState.WARNING

    return ROIState.NORMAL


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Evaluates all registered rules, aggregates scores, applies hysteresis,
    and writes the final SystemState into FrameContext.
    """

    def __init__(self, cfg: DecisionConfig):
        self._cfg = cfg
        self._rules: List[IDecisionRule] = self._default_rules(cfg)
        self._consecutive_abnormal: int = 0
        self._consecutive_normal: int = 0
        self._current_state: SystemState = SystemState.INITIALIZING
        self._frames_processed: int = 0
        self._min_warmup: int = 5

    # ── Public API ────────────────────────────────────────────────────────

    def add_rule(self, rule: IDecisionRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, name: str) -> None:
        self._rules = [r for r in self._rules if r.name != name]

    def decide(self, ctx: FrameContext) -> None:
        """
        Main entry point. Mutates *ctx* with:
          - ctx.triggered_rules
          - ctx.system_state
          - ctx.state_confidence
          - per-roi.roi_state
        """
        self._frames_processed += 1

        # Update per-ROI states first
        for roi in ctx.rois:
            roi.roi_state = _derive_roi_state(roi, self._cfg)

        # Evaluate rules
        triggered: List[RuleResult] = []
        for rule in self._rules:
            try:
                result = rule.evaluate(ctx)
                if result and result.triggered:
                    triggered.append(result)
            except Exception as exc:
                logger.error("Rule '%s' raised: %s", rule.name, exc)

        ctx.triggered_rules = triggered

        # Aggregate combined severity score
        combined_score = max((r.severity for r in triggered), default=0.0)

        # Hysteresis state machine
        new_state = self._hysteresis(combined_score, ctx)

        ctx.system_state = new_state
        ctx.state_confidence = combined_score
        ctx.consecutive_abnormal = self._consecutive_abnormal
        ctx.consecutive_normal = self._consecutive_normal

        # Mark event when state first goes abnormal
        if new_state == SystemState.ABNORMAL and self._current_state != SystemState.ABNORMAL:
            ctx.mark_event("state_change_abnormal", severity=combined_score)

        self._current_state = new_state

    def reset(self) -> None:
        self._consecutive_abnormal = 0
        self._consecutive_normal = 0
        self._current_state = SystemState.INITIALIZING
        self._frames_processed = 0

    # ── Internal ──────────────────────────────────────────────────────────

    def _hysteresis(self, score: float, ctx: FrameContext) -> SystemState:
        """Prevent flickering by requiring N consecutive frames."""
        cfg = self._cfg

        # Warm-up period
        if self._frames_processed < self._min_warmup:
            return SystemState.INITIALIZING

        if score >= cfg.abnormal_score_threshold:
            self._consecutive_abnormal += 1
            self._consecutive_normal = 0
        elif score >= cfg.warning_score_threshold:
            self._consecutive_abnormal = 0
            self._consecutive_normal = 0
            return SystemState.WARNING
        else:
            self._consecutive_normal += 1
            self._consecutive_abnormal = 0

        if self._consecutive_abnormal >= cfg.consecutive_abnormal_frames:
            return SystemState.ABNORMAL

        if self._consecutive_normal >= cfg.consecutive_normal_frames:
            return SystemState.NORMAL

        # Stay in current state if hysteresis not yet satisfied
        return self._current_state if self._current_state != SystemState.INITIALIZING else SystemState.NORMAL

    @staticmethod
    def _default_rules(cfg: DecisionConfig) -> List[IDecisionRule]:
        return [
            AIScoreRule(),
            SuddenChangeRule(cfg.sudden_change_threshold),
            StuckRule(),
            DriftRule(),
            OscillationRule(),
            LowConfidenceWarningRule(cfg.warning_score_threshold, cfg.abnormal_score_threshold),
        ]
