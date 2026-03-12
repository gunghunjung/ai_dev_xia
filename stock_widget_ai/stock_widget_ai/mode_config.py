"""
ModeConfig — Lightweight / Advanced 모드 설정 팩토리
────────────────────────────────────────────────────
Lightweight : CPU only, 저사양 PC, 30초~2분 학습
Advanced    : GPU 권장, 고성능 PC, 5~30분 학습

우선순위:
  Lightweight: 안정성 > 재현성 > 속도 > 정확도
  Advanced:    정확도 > 멀티모달 > 앙상블 > 계산비용
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class ModeConfig:
    # ── 식별 ─────────────────────────────────────────────────────────
    name: str = "lightweight"

    # ── 모델 셋 ──────────────────────────────────────────────────────
    primary_models: List[str] = field(default_factory=list)
    # DL 모델 (Trainer 기반)
    dl_models: List[str] = field(default_factory=list)
    # GBM 모델 (fit/predict 기반)
    gbm_models: List[str] = field(default_factory=list)

    # ── 학습 하이퍼파라미터 기본값 ───────────────────────────────────
    epochs: int = 30
    batch_size: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    lr: float = 5e-4
    scheduler: str = "onecycle"
    use_augmentation: bool = True
    use_mixed_precision: bool = True  # CUDA에서만 활성화됨

    # ── 검증 ─────────────────────────────────────────────────────────
    use_walk_forward: bool = False
    wf_n_splits: int = 3
    wf_gap: int = 10             # 학습-테스트 gap (영업일)
    wf_window: str = "expanding"
    wf_min_train: int = 252

    # ── 앙상블 ──────────────────────────────────────────────────────
    use_regime_ensemble: bool = False
    use_meta_learner: bool = False    # stacking
    ensemble_weights: Dict[str, float] = field(default_factory=dict)

    # ── 불확실성 ─────────────────────────────────────────────────────
    use_mc_dropout: bool = False
    mc_samples: int = 10
    use_conformal: bool = False
    conformal_alpha: float = 0.1      # 90% PI
    use_bootstrap_ci: bool = True
    bootstrap_n: int = 200

    # ── 설명 ─────────────────────────────────────────────────────────
    shap_method: str = "tree"         # "tree" | "gradient" | "kernel"
    shap_max_samples: int = 200

    # ── Foundation model ─────────────────────────────────────────────
    use_foundation_model: bool = False
    foundation_model: str = "chronos"

    # ── 피처 선택 ────────────────────────────────────────────────────
    use_feature_selection: bool = False
    max_features: int = 50

    # ── 시퀀스 ──────────────────────────────────────────────────────
    sequence_length: int = 30

    # ── GBM 설정 ─────────────────────────────────────────────────────
    gbm_n_estimators: int = 200
    gbm_max_depth: int = 6
    gbm_early_stopping: int = 20

    @classmethod
    def lightweight(cls) -> "ModeConfig":
        """
        저사양 PC 최적화 설정.
        XGBoost + LightGBM 위주, LSTM 보조.
        Bootstrap CI, SHAP TreeExplainer.
        Hold-out 검증.
        """
        return cls(
            name="lightweight",
            primary_models=["xgboost", "lightgbm", "lstm"],
            dl_models=["lstm"],
            gbm_models=["xgboost", "lightgbm"],
            epochs=30,
            batch_size=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.2,
            lr=5e-4,
            scheduler="cosine",
            use_augmentation=False,      # 속도 우선
            use_mixed_precision=True,
            use_walk_forward=False,
            wf_n_splits=3,
            wf_gap=10,
            use_regime_ensemble=False,
            use_meta_learner=False,
            ensemble_weights={"xgboost": 0.4, "lightgbm": 0.4, "lstm": 0.2},
            use_mc_dropout=False,
            mc_samples=10,
            use_conformal=False,
            use_bootstrap_ci=True,
            bootstrap_n=200,
            shap_method="tree",
            shap_max_samples=100,
            use_foundation_model=False,
            use_feature_selection=False,
            max_features=50,
            sequence_length=30,
            gbm_n_estimators=200,
            gbm_max_depth=6,
            gbm_early_stopping=20,
        )

    @classmethod
    def advanced(cls) -> "ModeConfig":
        """
        고성능 GPU 설정.
        Transformer + PatchTST + GBM 멀티모달.
        Walk-forward CV, Conformal PI, MC Dropout.
        RegimeEnsemble + MetaLearner stacking.
        SHAP GradientExplainer.
        """
        return cls(
            name="advanced",
            primary_models=["transformer", "patchtst", "xgboost", "lightgbm"],
            dl_models=["transformer", "patchtst"],
            gbm_models=["xgboost", "lightgbm"],
            epochs=80,
            batch_size=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.2,
            lr=5e-4,
            scheduler="onecycle",
            use_augmentation=True,
            use_mixed_precision=True,
            use_walk_forward=True,
            wf_n_splits=5,
            wf_gap=20,
            wf_window="expanding",
            wf_min_train=252,
            use_regime_ensemble=True,
            use_meta_learner=True,
            ensemble_weights={
                "transformer": 0.3, "patchtst": 0.25,
                "xgboost": 0.25, "lightgbm": 0.2,
            },
            use_mc_dropout=True,
            mc_samples=30,
            use_conformal=True,
            conformal_alpha=0.1,
            use_bootstrap_ci=False,
            bootstrap_n=0,
            shap_method="gradient",    # DL 모델용
            shap_max_samples=300,
            use_foundation_model=False,  # 별도 설치 필요
            foundation_model="chronos",
            use_feature_selection=True,
            max_features=40,
            sequence_length=60,
            gbm_n_estimators=500,
            gbm_max_depth=8,
            gbm_early_stopping=30,
        )

    @classmethod
    def from_name(cls, name: str) -> "ModeConfig":
        """이름으로 ModeConfig 생성"""
        if name == "advanced":
            return cls.advanced()
        return cls.lightweight()

    def to_train_params(self) -> dict:
        """TrainingPanel / AppController.train() 에 전달할 params dict"""
        return {
            "model":       self.dl_models[0] if self.dl_models else "lstm",
            "epochs":      self.epochs,
            "batch_size":  self.batch_size,
            "hidden_dim":  self.hidden_dim,
            "num_layers":  self.num_layers,
            "num_heads":   self.num_heads,
            "dropout":     self.dropout,
            "lr":          self.lr,
            "scheduler":   self.scheduler,
            "augmentation": self.use_augmentation,
        }

    def describe(self) -> str:
        """로그/UI 표시용 설명 문자열"""
        lines = [
            f"[{self.name.upper()} 모드]",
            f"  모델: {', '.join(self.primary_models)}",
            f"  검증: {'Walk-forward(' + str(self.wf_n_splits) + '폴드)' if self.use_walk_forward else 'Hold-out'}",
            f"  앙상블: {'Regime+Meta' if self.use_regime_ensemble else '단순 가중합'}",
            f"  불확실성: {'MC Dropout+Conformal' if self.use_mc_dropout else 'Bootstrap CI'}",
            f"  SHAP: {self.shap_method}",
            f"  피처선택: {'사용' if self.use_feature_selection else '미사용'}",
        ]
        return "\n".join(lines)
