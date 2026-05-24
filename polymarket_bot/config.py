"""
Production-grade configuration for Polymarket BTC 5-minute directional bot.
All parameters are mathematically grounded and regime-adaptive.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, IntEnum


class RegimeState(IntEnum):
    """Hidden Markov Model regime states for BTC microstructure."""
    CALM_TRENDING = 0
    CALM_RANGE = 1
    VOLATILE_TRENDING = 2
    VOLATILE_RANGE = 3
    CRISIS = 4
    LIQUIDATION_CASCADE = 5


class Direction(Enum):
    """Directional outcome for settlement."""
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class SignalType(Enum):
    """Signal classification types."""
    OBSERVATION = "observation"
    CONTINUATION = "continuation"
    REVERSAL = "reversal"
    EXHAUSTION = "exhaustion"
    ABSORPTION = "absorption"
    VACUUM = "vacuum"
    BURST_FAILURE = "burst_failure"
    SUPPRESSED = "suppressed"


class ExecutionQuality(Enum):
    """Execution quality classification."""
    OPTIMAL = "optimal"
    ACCEPTABLE = "acceptable"
    DEGRADED = "degraded"
    REJECT = "reject"


@dataclass(frozen=True)
class PolymarketConfig:
    """Polymarket API and authentication configuration."""
    private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
    funder: str = os.getenv("POLYMARKET_FUNDER", os.getenv("DEPOSIT_WALLET", ""))
    sig_type: int = int(os.getenv("POLYMARKET_SIG_TYPE", "3"))
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    relayer_url: str = os.getenv("RELAYER_URL", "https://relayer-v2.polymarket.com")
    clob_api_key: str = os.getenv("CLOB_API_KEY", "")
    clob_api_secret: str = os.getenv("CLOB_SECRET", "")
    clob_api_passphrase: str = os.getenv("CLOB_PASS_PHRASE", "")
    builder_api_key: str = os.getenv("BUILDER_API_KEY", "")
    builder_secret: str = os.getenv("BUILDER_SECRET", "")
    builder_passphrase: str = os.getenv("BUILDER_PASS_PHRASE", "")
    deposit_wallet: str = os.getenv("DEPOSIT_WALLET", "")
    clob_api_url: str = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
    gamma_api_url: str = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com")
    price_to_beat_api_url: str = os.getenv(
        "PRICE_TO_BEAT_API_URL",
        "https://polymarket.com/api/equity/price-to-beat",
    )
    rtds_ws_url: str = os.getenv("RTDS_WS_URL", "wss://ws-live-data.polymarket.com")
    chain_id: int = int(os.getenv("CHAIN_ID", "137"))
    rpc_url: str = os.getenv("RPC_URL", "https://polygon-rpc.com")
    ctf_exchange_address: str = os.getenv("CTF_EXCHANGE_ADDRESS", "0xE111180000d2663C0091e4f400237545B87B996B")
    neg_risk_exchange_address: str = os.getenv("NEG_RISK_EXCHANGE_ADDRESS", "0xe2222d279d744050d28e00520010520000310F59")
    conditional_tokens_address: str = os.getenv("CONDITIONAL_TOKENS_ADDRESS", "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
    collateral_token_address: str = os.getenv("COLLATERAL_TOKEN_ADDRESS", "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
    ctf_collateral_adapter_address: str = os.getenv("CTF_COLLATERAL_ADAPTER_ADDRESS", "0xAdA100Db00Ca00073811820692005400218FcE1f")
    neg_risk_ctf_collateral_adapter_address: str = os.getenv("NEG_RISK_CTF_COLLATERAL_ADAPTER_ADDRESS", "0xadA2005600Dec949baf300f4C6120000bDB6eAab")


@dataclass(frozen=True)
class MarketConfig:
    """BTC 5-minute market specification."""
    settlement_interval_seconds: float = 300.0  # 5 minutes
    condition_id: str = ""  # Will be resolved dynamically
    up_token_id: str = ""   # Will be resolved dynamically
    down_token_id: str = ""  # Will be resolved dynamically
    min_order_size: float = float(os.getenv("POLYMARKET_MIN_ORDER_SIZE", "5.0"))  # shares fallback
    enforce_max_entry_price: bool = True
    max_entry_price: float = 0.65
    max_position_size: float = 50.0
    price_tick: float = 0.01
    max_slippage_bps: float = 50.0


@dataclass
class HMMConfig:
    """Hidden Markov Model configuration for regime detection."""
    n_regimes: int = 6
    emission_dim: int = 8  # volatility, flow_imbalance, spread, taker_ratio, ...
    transition_smoothing: float = 1e-3
    emission_smoothing: float = 1e-2
    min_observations_fit: int = 200
    refit_interval_seconds: float = 600.0
    regime_persistence_prior: Dict[int, float] = field(default_factory=lambda: {
        RegimeState.CALM_TRENDING: 0.92,
        RegimeState.CALM_RANGE: 0.88,
        RegimeState.VOLATILE_TRENDING: 0.75,
        RegimeState.VOLATILE_RANGE: 0.70,
        RegimeState.CRISIS: 0.55,
        RegimeState.LIQUIDATION_CASCADE: 0.40,
    })
    regime_volatility_prior: Dict[int, float] = field(default_factory=lambda: {
        RegimeState.CALM_TRENDING: 0.0001,
        RegimeState.CALM_RANGE: 0.00005,
        RegimeState.VOLATILE_TRENDING: 0.0005,
        RegimeState.VOLATILE_RANGE: 0.0003,
        RegimeState.CRISIS: 0.001,
        RegimeState.LIQUIDATION_CASCADE: 0.003,
    })

    @property
    def num_regimes(self) -> int:
        return self.n_regimes


@dataclass
class KalmanConfig:
    """Kalman filter configuration for price state estimation."""
    process_noise_scale: float = 1e-4
    measurement_noise_scale: float = 1e-3
    initial_state_variance: float = 1.0
    adaptive_noise_window: int = 50
    innovation_threshold: float = 3.0  # Chi-squared threshold for outlier rejection
    state_dim: int = 3  # [price, velocity, acceleration]


@dataclass
class BayesianConfig:
    """Bayesian evidence fusion configuration."""
    prior_strength: float = 0.5
    evidence_decay_halflife: float = 60.0  # seconds
    min_evidence_weight: float = 0.01
    max_evidence_weight: float = 1.0
    posterior_smoothing: float = 0.95
    calibration_bins: int = 20
    calibration_window: int = 500


@dataclass
class MicrostructureConfig:
    """Microstructure analysis configuration."""
    # Orderbook analysis
    orderbook_depth_levels: int = 20
    spoof_detection_threshold: float = 0.6  # Cancellation ratio threshold
    spoof_cancellation_velocity_threshold: float = 2.0  # Cancel rate vs add rate
    fake_liquidity_ratio_threshold: float = 0.4
    orderbook_pressure_smoothing: float = 0.85
    
    # Flow analysis
    taker_aggression_window: int = 20  # ticks
    flow_imbalance_smoothing: float = 0.90
    flow_acceleration_window: int = 10
    taker_ratio_threshold: float = 0.65
    
    # Queue modeling
    queue_survival_decay_rate: float = 0.05
    queue_persistence_window: int = 30
    
    # Spread analysis
    spread_stability_window: int = 50
    spread_expansion_threshold: float = 2.0  # Standard deviations
    effective_spread_weight: float = 0.7
    
    # Liquidity analysis
    liquidity_stability_window: int = 30
    real_liquidity_min_survival: float = 0.5  # Fraction that must survive
    vacuum_detection_threshold: float = 0.3  # Fraction of liquidity removed
    absorption_detection_volume_ratio: float = 3.0  # Absorbed vs displayed


@dataclass
class SignalConfig:
    """Signal engine configuration."""
    # Continuation engine
    continuation_min_regime_persistence: float = 0.7
    continuation_flow_threshold: float = 0.55
    continuation_momentum_decay: float = 0.02
    continuation_max_entropy: float = 0.7
    
    # Reversal engine
    reversal_taker_inversion_threshold: float = 0.4
    reversal_continuation_collapse_rate: float = 0.5
    reversal_vacuum_sensitivity: float = 0.6
    reversal_absorption_wall_ratio: float = 2.5
    reversal_microburst_velocity: float = 3.0  # Standard deviations
    
    # Exhaustion detection
    exhaustion_taker_decay_rate: float = 0.15
    exhaustion_flow_deceleration_threshold: float = 0.3
    exhaustion_volume_climax_ratio: float = 2.0
    
    # Burst detection
    burst_failure_reversal_depth: float = 0.5
    burst_velocity_threshold: float = 2.5
    burst_time_window_seconds: float = 30.0
    
    # Entropy filter
    entropy_suppression_threshold: float = 0.8
    entropy_window: int = 30
    decorrelation_correlation_threshold: float = 0.7
    decorrelation_window: int = 50
    decorrelation_min_trade_interval_seconds: float = 10.0
    decorrelation_late_window_interval_seconds: float = 4.0
    decorrelation_late_window_start_seconds: float = 90.0
    decorrelation_edge_similarity_bps: float = 50.0
    decorrelation_max_flips_in_window: int = 3
    decorrelation_flip_window_seconds: float = 300.0


@dataclass
class ExecutionConfig:
    """Execution realism configuration."""
    max_latency_ms: float = 500.0
    typical_latency_ms: float = 200.0
    latency_uncertainty_ms: float = 100.0
    fill_probability_base: float = 0.85
    slippage_model_sigma: float = 0.001
    spread_expansion_regime_multiplier: Dict[int, float] = field(default_factory=lambda: {
        RegimeState.CALM_TRENDING: 1.0,
        RegimeState.CALM_RANGE: 1.0,
        RegimeState.VOLATILE_TRENDING: 1.5,
        RegimeState.VOLATILE_RANGE: 1.3,
        RegimeState.CRISIS: 2.5,
        RegimeState.LIQUIDATION_CASCADE: 4.0,
    })
    min_fill_probability: float = 0.5
    max_spread_bps_for_execution: float = 200.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_notional: float = 100.0
    max_loss_per_settlement: float = 10.0
    max_consecutive_losses: int = 5
    cooldown_after_max_losses_seconds: float = 1800.0  # 30 minutes
    min_confidence_threshold: float = 0.55
    min_edge_bps: float = 100.0  # Minimum 1% edge in probability
    position_sizing_kelly_fraction: float = 0.25  # Quarter-Kelly for safety
    max_drawdown_fraction: float = 0.15
    settlement_time_buffer_seconds: float = 30.0  # Don't trade within 30s of settlement


@dataclass
class VolatilityConfig:
    """Dynamic volatility engine configuration."""
    garch_p: int = 1
    garch_q: int = 1
    volatility_smoothing: float = 0.94
    volatility_clustering_window: int = 100
    regime_transition_volatility_threshold: float = 2.0
    stochastic_volatility_drift: float = 0.0
    stochastic_volatility_mean_reversion: float = 0.1
    stochastic_volatility_vol_of_vol: float = 0.3


@dataclass
class BotConfig:
    """Master configuration container."""
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    microstructure: MicrostructureConfig = field(default_factory=MicrostructureConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    
    # Data feed configuration
    btc_price_source: str = "multi_exchange"
    btc_price_symbol: str = "BTCUSDT"
    coinbase_product_id: str = "BTC-USD"
    settlement_price_source: str = "coinbase"
    btc_price_interval_ms: float = 1000.0
    btc_price_max_age_seconds: float = float(os.getenv("BTC_PRICE_MAX_AGE_SECONDS", "8.0"))
    orderbook_interval_ms: float = 500.0
    polymarket_poll_interval_ms: float = 2000.0
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "polymarket_bot.log"
    trade_history_file: str = os.getenv("TRADE_HISTORY_FILE", "polymarket_trade_history.jsonl")
    heartbeat_interval_seconds: float = float(os.getenv("BOT_HEARTBEAT_INTERVAL_SECONDS", "15.0"))
    run_cycle_timeout_seconds: float = float(os.getenv("BOT_RUN_CYCLE_TIMEOUT_SECONDS", "15.0"))
    rollover_prefetch_seconds: float = float(os.getenv("ROLLOVER_PREFETCH_SECONDS", "90.0"))
    rollover_prefetch_poll_seconds: float = float(os.getenv("ROLLOVER_PREFETCH_POLL_SECONDS", "1.0"))
    rollover_fetch_timeout_seconds: float = float(os.getenv("ROLLOVER_FETCH_TIMEOUT_SECONDS", "2.5"))
    rollover_prefetch_rounds: int = int(os.getenv("ROLLOVER_PREFETCH_ROUNDS", "2"))
    rollover_orderbook_warmup_seconds: float = float(os.getenv("ROLLOVER_ORDERBOOK_WARMUP_SECONDS", "8.0"))
    price_to_beat_refresh_timeout_seconds: float = float(os.getenv("PRICE_TO_BEAT_REFRESH_TIMEOUT_SECONDS", "1.5"))
    use_official_settlement_close: bool = os.getenv("USE_OFFICIAL_SETTLEMENT_CLOSE", "true").lower() != "false"
    official_settlement_poll_seconds: float = 1.0
    local_settlement_grace_seconds: float = float(os.getenv("LOCAL_SETTLEMENT_GRACE_SECONDS", "1.0"))
    wallet_maintenance_enabled: bool = os.getenv("WALLET_MAINTENANCE_ENABLED", "false").lower() == "true"
    wallet_maintenance_interval_seconds: float = float(os.getenv("WALLET_MAINTENANCE_INTERVAL_SECONDS", "60.0"))
    auto_update_allowance_enabled: bool = os.getenv("AUTO_UPDATE_ALLOWANCE_ENABLED", "false").lower() == "true"
    auto_claim_enabled: bool = os.getenv("AUTO_CLAIM_ENABLED", "false").lower() == "true"
    
    # Strategy
    strategy_name: str = "btc_5min_directional"
    strategy_version: str = "1.0.0"
    trade_expected_direction_only: bool = True
    one_entry_per_round: bool = True
    allow_direction_flip_entries: bool = False
    directional_flip_min_probability: float = 0.66
    directional_min_probability: float = 0.62
    directional_min_confidence: float = 0.55
    post_observation_confirmation_seconds: float = float(
        os.getenv("POST_OBSERVATION_CONFIRMATION_SECONDS", "0.0")
    )
    observation_window_seconds: float = 40.0
    require_observation_window: bool = True
    observation_min_samples: int = int(os.getenv("OBSERVATION_MIN_SAMPLES", "32"))
    observation_min_coverage: float = float(os.getenv("OBSERVATION_MIN_COVERAGE", "0.86"))
    observation_max_gap_seconds: float = float(os.getenv("OBSERVATION_MAX_GAP_SECONDS", "6.0"))
    observation_min_validation_score: float = float(os.getenv("OBSERVATION_MIN_VALIDATION_SCORE", "0.60"))
    observation_min_abs_log_odds: float = 0.45
    observation_min_direction_confidence: float = 0.58
    observation_ema_slope_enabled: bool = os.getenv(
        "OBSERVATION_EMA_SLOPE_ENABLED", "true"
    ).lower() == "true"
    observation_ema_slope_fusion_weight: float = float(
        os.getenv("OBSERVATION_EMA_SLOPE_FUSION_WEIGHT", "0.42")
    )
    observation_ema_slope_max_log_odds: float = float(
        os.getenv("OBSERVATION_EMA_SLOPE_MAX_LOG_ODDS", "2.4")
    )
    observation_ema_slope_min_quality: float = float(
        os.getenv("OBSERVATION_EMA_SLOPE_MIN_QUALITY", "0.38")
    )
    round_direction_lock_enabled: bool = True
    require_round_direction_lock_for_entry: bool = False
    round_direction_min_confidence: float = 0.68
    round_direction_min_abs_log_odds: float = 0.50
    round_direction_max_technical_conflict: float = 0.35
    round_direction_lock_conflict_log_odds: float = float(
        os.getenv("ROUND_DIRECTION_LOCK_CONFLICT_LOG_ODDS", "0.75")
    )
    round_direction_lock_conflict_confidence: float = float(
        os.getenv("ROUND_DIRECTION_LOCK_CONFLICT_CONFIDENCE", "0.55")
    )
    settlement_tie_break_bps: float = float(
        os.getenv("SETTLEMENT_TIE_BREAK_BPS", "1.5")
    )
    settlement_tie_break_abs_usd: float = float(
        os.getenv("SETTLEMENT_TIE_BREAK_ABS_USD", "1.0")
    )
    technical_momentum_enabled: bool = True
    technical_momentum_min_samples: int = 32
    technical_momentum_min_timeframes: int = 2
    technical_momentum_min_consensus: float = 0.62
    technical_momentum_max_conflict: float = 0.38
    technical_momentum_max_log_odds: float = 2.6
    technical_momentum_fusion_weight: float = 0.38
    technical_momentum_smoothing_alpha: float = 0.62
    technical_momentum_hysteresis_log_odds: float = 0.35
    technical_momentum_tick_buffer_slack_seconds: float = float(
        os.getenv("TECHNICAL_MOMENTUM_TICK_BUFFER_SLACK_SECONDS", "8.0")
    )
    technical_momentum_rsi50_macd50_required: bool = True
    technical_momentum_rsi50_min_distance: float = 1.0
    technical_momentum_macd50_min_z: float = 0.15
