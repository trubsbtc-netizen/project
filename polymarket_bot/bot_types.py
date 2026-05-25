"""
Core type definitions for the Polymarket BTC 5-minute directional bot.
All data structures are designed for low-latency, numerically stable operations.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum, IntEnum
from datetime import datetime, timezone
from collections import deque

from polymarket_bot.config import RegimeState, Direction, SignalType, ExecutionQuality


# ============================================================
# RAW DATA TYPES
# ============================================================

@dataclass
class TickData:
    """Single price tick from BTC feed."""
    timestamp: float  # Unix epoch seconds
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    trade_id: str = ""
    is_snapshot: bool = False


@dataclass
class OrderbookLevel:
    """Single level in the orderbook."""
    price: float
    size: float
    order_count: int = 0
    timestamp: float = 0.0
    is_real: bool = True  # Flag for spoof detection


@dataclass
class OrderbookSnapshot:
    """Full orderbook snapshot at a point in time."""
    timestamp: float
    bids: List[OrderbookLevel]  # Sorted descending by price
    asks: List[OrderbookLevel]  # Sorted ascending by price
    sequence_number: int = 0
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None
    
    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_bps(self) -> Optional[float]:
        if self.spread and self.mid_price and self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000.0
        return None


@dataclass
class OrderbookDelta:
    """Incremental orderbook update (add/cancel/trade)."""
    timestamp: float
    side: str  # 'bid' or 'ask'
    price: float
    size_delta: float  # Positive = add, Negative = cancel/trade
    order_count_delta: int = 0
    is_trade: bool = False  # True if this delta is from a trade execution
    sequence_number: int = 0


@dataclass
class PolymarketOrderbook:
    """Polymarket CLOB orderbook for a specific token."""
    timestamp: float
    token_id: str
    bids: List[OrderbookLevel]  # Sorted descending
    asks: List[OrderbookLevel]  # Sorted ascending
    condition_id: str = ""
    
    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None
    
    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def best_bid_price(self) -> Optional[float]:
        return self.best_bid

    @property
    def best_ask_price(self) -> Optional[float]:
        return self.best_ask

    @property
    def best_bid_size(self) -> Optional[float]:
        return self.bids[0].size if self.bids else None

    @property
    def best_ask_size(self) -> Optional[float]:
        return self.asks[0].size if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None


@dataclass
class PriceToBeatSnapshot:
    """Price-to-beat value with RTDS/source timing and local receipt time."""
    price: float
    source_timestamp: Optional[float] = None
    received_timestamp: float = 0.0
    rtds_message_timestamp: Optional[float] = None
    source: str = ""
    symbol: str = ""
    market_slug: str = ""
    is_carried_forward: bool = False
    is_valid: bool = True
    reason: str = ""

    @property
    def effective_timestamp(self) -> float:
        return self.source_timestamp or 0.0

    @property
    def age_seconds(self) -> float:
        if not self.received_timestamp or not self.source_timestamp:
            return 0.0
        return max(0.0, self.received_timestamp - self.source_timestamp)


@dataclass
class PolymarketMarketInfo:
    """Polymarket market metadata."""
    condition_id: str
    question: str
    tokens: Dict[str, str]  # {"UP": token_id, "DOWN": token_id}
    end_date_iso: str
    slug: str = ""
    active: bool = True
    closed: bool = False
    minimum_order_size: float = 5.0  # CLOB minimum size in outcome-token shares
    minimum_tick_size: float = 0.01
    fee_rate: float = 0.0
    fee_exponent: float = 1.0
    price_to_beat: Optional[float] = None
    price_to_beat_snapshot: Optional[PriceToBeatSnapshot] = None

    @property
    def token_id_up(self) -> str:
        return self.tokens.get("UP", "")

    @property
    def token_id_down(self) -> str:
        return self.tokens.get("DOWN", "")

    @property
    def price_to_beat_timestamp(self) -> Optional[float]:
        if self.price_to_beat_snapshot is None:
            return None
        return self.price_to_beat_snapshot.effective_timestamp


# ============================================================
# MICROSTRUCTURE ANALYSIS TYPES
# ============================================================

@dataclass
class OrderbookPressure:
    """Computed orderbook pressure metrics."""
    timestamp: float
    bid_pressure: float  # Weighted bid volume metric
    ask_pressure: float  # Weighted ask volume metric
    net_pressure: float  # bid_pressure - ask_pressure
    pressure_imbalance: float  # Normalized: (bid-ask)/(bid+ask)
    real_bid_pressure: float  # After spoof filtering
    real_ask_pressure: float  # After spoof filtering
    real_net_pressure: float  # real_bid - real_ask
    real_imbalance: float  # After spoof filtering normalized
    spoof_score: float  # 0-1, higher = more spoofing detected
    fake_liquidity_ratio: float  # Fraction of displayed liquidity deemed fake
    depth_skew: float  # Asymmetry in depth distribution
    weighted_mid: float  # Volume-weighted midpoint
    microprice: float  # Orderbook microprice (Stoikov)
    
    @staticmethod
    def compute_microprice(bid_price: float, bid_size: float,
                           ask_price: float, ask_size: float) -> float:
        """
        Stoikov microprice: P_micro = bid*Q_ask + ask*Q_bid / (Q_bid + Q_ask)
        This is the fair price given the orderbook imbalance.
        """
        if bid_size + ask_size > 0:
            return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
        return (bid_price + ask_price) / 2.0


@dataclass
class FlowMetrics:
    """Order flow and taker aggression metrics."""
    timestamp: float
    taker_buy_volume: float
    taker_sell_volume: float
    taker_ratio: float  # buy_volume / total_volume
    taker_aggression_score: float  # Normalized aggression metric
    flow_imbalance: float  # (buy - sell) / (buy + sell)
    flow_acceleration: float  # Rate of change of flow imbalance
    cumulative_flow_delta: float  # Running cumulative flow
    maker_volume: float
    total_volume: float
    trade_count: int
    avg_trade_size: float
    large_trade_count: int  # Trades above threshold
    large_trade_ratio: float  # Fraction of volume from large trades
    taker_inversion: float  # Recent inversion in taker direction
    flow_persistence: float  # Autocorrelation of flow imbalance
    flow_decay_rate: float  # Exponential decay rate of flow signal


@dataclass
class LiquidityMetrics:
    """Liquidity stability and depth metrics."""
    timestamp: float
    total_bid_depth: float  # Sum of bid sizes up to depth levels
    total_ask_depth: float  # Sum of ask sizes up to depth levels
    real_bid_depth: float  # After spoof filtering
    real_ask_depth: float  # After spoof filtering
    depth_ratio: float  # bid_depth / ask_depth
    liquidity_stability_score: float  # 0-1, how stable liquidity has been
    cancellation_velocity: float  # Rate of order cancellations
    add_cancel_ratio: float  # Order additions vs cancellations
    absorption_score: float  # 0-1, absorption detected
    absorption_volume: float  # Volume absorbed at key levels
    vacuum_score: float  # 0-1, liquidity vacuum detected
    vacuum_depth: float  # Depth of liquidity vacuum
    real_liquidity_fraction: float  # Fraction of displayed that's real
    liquidity_concentration: float  # How concentrated depth is at top levels
    depth_skew_coefficient: float  # Power-law skew of depth distribution


@dataclass
class QueueMetrics:
    """Queue persistence and survival modeling."""
    timestamp: float
    queue_survival_probability: float  # Estimated probability queue position survives
    queue_position_estimate: float  # Estimated queue position
    cancellation_rate: float  # Orders per second being cancelled
    fill_rate: float  # Orders per second being filled
    queue_decay_rate: float  # Exponential decay rate of queue
    queue_imbalance: float  # Bid queue vs ask queue survival
    effective_queue_depth: float  # Queue depth adjusted for cancellation
    queue_age_distribution: float  # Average age of orders in queue
    front_of_queue_probability: float  # Probability of being at front of queue

    @property
    def bid_queue_survival(self) -> float:
        return float(np.clip(self.queue_survival_probability * (1.0 + self.queue_imbalance), 0.01, 0.99))

    @property
    def ask_queue_survival(self) -> float:
        return float(np.clip(self.queue_survival_probability * (1.0 - self.queue_imbalance), 0.01, 0.99))

    @property
    def bid_queue_position(self) -> float:
        return max(0.0, self.queue_position_estimate * (1.0 - self.queue_imbalance))

    @property
    def ask_queue_position(self) -> float:
        return max(0.0, self.queue_position_estimate * (1.0 + self.queue_imbalance))


@dataclass
class SpreadMetrics:
    """Spread analysis metrics."""
    timestamp: float
    spread: float  # Current spread
    spread_bps: float  # Spread in basis points
    effective_spread: float  # Effective spread (trade-weighted)
    realized_spread: float  # Realized spread after price movement
    spread_stability: float  # 0-1, how stable spread has been
    spread_expansion_score: float  # 0-1, spread expansion detected
    spread_asymmetry: float  # Bid-ask spread asymmetry
    quoted_spread: float  # Quoted spread at current time
    time_weighted_spread: float  # Time-weighted average spread

    @property
    def is_expanding(self) -> bool:
        return self.spread_expansion_score >= 0.5


# ============================================================
# REGIME AND VOLATILITY TYPES
# ============================================================

@dataclass
class RegimeStateEstimate:
    """Current regime state estimate from HMM."""
    timestamp: float
    current_regime: RegimeState
    regime_probabilities: Dict[RegimeState, float]  # P(regime_i | observations)
    regime_persistence: float  # Probability current regime persists
    regime_transition_probability: Dict[RegimeState, float]  # P(next_regime | current)
    regime_duration_estimate: float  # Expected duration of current regime (seconds)
    regime_volatility_estimate: float  # Expected volatility in current regime
    regime_flow_characteristic: float  # Expected flow imbalance in regime
    regime_confidence: float  # Confidence in regime classification
    regime_entropy: float  # Shannon entropy of regime distribution


@dataclass
class VolatilityEstimate:
    """Multi-model volatility estimate."""
    timestamp: float
    realized_volatility: float  # Realized vol from recent returns
    garch_volatility: float  # GARCH(1,1) estimate
    stochastic_volatility: float  # Stochastic vol model estimate
    implied_volatility: float  # From Polymarket prices (if available)
    kalman_volatility: float  # Kalman-filtered volatility
    regime_adjusted_volatility: float  # Adjusted for regime
    volatility_of_volatility: float  # Second moment (vol clustering)
    volatility_skew: float  # Asymmetric volatility (up vs down)
    volatility_forecast_5min: float  # 5-minute forward forecast
    volatility_confidence_interval: Tuple[float, float]  # (low, high) bounds
    volatility_regime_alignment: float  # How well vol matches regime


# ============================================================
# SIGNAL TYPES
# ============================================================

@dataclass
class ContinuationSignal:
    """Momentum continuation signal."""
    timestamp: float
    direction: Direction
    continuation_probability: float  # P(continuation | data)
    momentum_strength: float  # 0-1
    flow_confirmation: float  # 0-1, flow supports continuation
    regime_support: float  # 0-1, regime supports continuation
    persistence_score: float  # 0-1, momentum persistence
    exhaustion_risk: float  # 0-1, risk of exhaustion
    entropy_score: float  # Signal entropy
    kalman_velocity: float  # Price velocity from Kalman
    kalman_acceleration: float  # Price acceleration from Kalman
    signal_quality: float  # Overall signal quality 0-1
    is_valid: bool  # Whether signal passes all filters

    @property
    def continuation_direction(self) -> Direction:
        return self.direction


@dataclass
class ReversalSignal:
    """Fast reversal detection signal."""
    timestamp: float
    direction: Direction  # Direction of reversal
    reversal_probability: float  # P(reversal | data)
    reversal_speed: float  # How fast reversal is forming
    taker_inversion_score: float  # Taker flow inversion metric
    continuation_collapse_score: float  # Continuation momentum collapse
    vacuum_score: float  # Liquidity vacuum on opposite side
    absorption_score: float  # Absorption wall detected
    microburst_score: float  # Microburst reversal detected
    exhaustion_score: float  # Taker exhaustion
    failed_breakout_score: float  # Failed breakout burst
    reversal_depth: float  # Expected reversal depth
    signal_quality: float  # Overall signal quality 0-1
    is_valid: bool

    @property
    def reversal_direction(self) -> Direction:
        return self.direction


@dataclass
class ExhaustionSignal:
    """Continuation exhaustion detection."""
    timestamp: float
    direction: Direction  # Direction being exhausted
    exhaustion_probability: float
    taker_decay_rate: float  # Rate of taker aggression decay
    flow_deceleration: float  # Flow imbalance deceleration
    volume_climax_score: float  # Volume climax detection
    absorption_behind: float  # Absorption behind the move
    spread_expansion: float  # Spread widening during exhaustion
    depth_imbalance_shift: float  # Depth shifting against move
    exhaustion_severity: float  # 0-1 severity
    is_valid: bool


@dataclass
class BurstFailureSignal:
    """Failed breakout burst detection."""
    timestamp: float
    direction: Direction  # Original burst direction
    burst_failure_probability: float
    burst_velocity: float  # Initial burst speed
    reversal_depth: float  # How far it reversed
    failure_speed: float  # How fast the failure occurred
    volume_profile: float  # Volume during burst vs reversal
    liquidity_absorption: float  # Liquidity that stopped the burst
    is_valid: bool


@dataclass
class EntropyFilterResult:
    """Entropy-aware signal suppression result."""
    timestamp: float
    raw_signal_strength: float  # Original signal strength
    entropy_score: float  # Shannon entropy of recent signals
    decorrelation_score: float  # Signal decorrelation metric
    suppression_factor: float  # 0-1, how much signal is suppressed
    is_suppressed: bool  # Whether signal is fully suppressed
    effective_signal_strength: float  # After suppression
    correlation_with_history: float  # Correlation with recent signals
    novelty_score: float  # How novel this signal is


@dataclass
class ObservationSignal:
    """Opening observation signal for a single 5-minute market."""
    timestamp: float
    market_slug: str
    market_start_timestamp: float
    observation_start_timestamp: float
    observation_end_timestamp: float
    observed_seconds: float
    required_seconds: float
    sample_count: int
    effective_sample_size: float
    coverage_ratio: float
    max_gap_seconds: float
    outlier_fraction: float
    is_ready: bool
    is_valid: bool
    direction: Direction
    p_up: float
    p_down: float
    confidence: float
    validation_score: float
    uncertainty: float
    terminal_z_score: float
    drift_per_second: float
    sigma_per_sqrt_second: float
    student_t_df: float
    terminal_log_odds: float
    microstructure_log_odds: float
    combined_log_odds: float
    ema_slope_log_odds: float = 0.0
    ema_slope_velocity: float = 0.0
    ema_spread_acceleration: float = 0.0
    ema_curvature: float = 0.0
    ema_turning_pressure: float = 0.0
    reason: str = ""


@dataclass
class TechnicalMomentumSignal:
    """Multi-timeframe RSI/MACD directional evidence."""
    timestamp: float
    market_slug: str
    is_valid: bool
    direction: Direction
    p_up: float
    p_down: float
    confidence: float
    validation_score: float
    consensus_score: float
    conflict_score: float
    log_odds: float
    rsi_log_odds: float
    macd_log_odds: float
    timeframe_probabilities: Dict[str, float]
    timeframe_weights: Dict[str, float]
    rsi_values: Dict[str, float]
    macd_histograms: Dict[str, float]
    dominant_timeframe: str
    reason: str = ""


# ============================================================
# INFERENCE TYPES
# ============================================================

@dataclass
class KalmanState:
    """Kalman filter state estimate."""
    timestamp: float
    state: np.ndarray  # [price, velocity, acceleration]
    covariance: np.ndarray  # State covariance matrix
    innovation: float  # Latest innovation (measurement residual)
    innovation_variance: float  # Innovation variance
    kalman_gain: np.ndarray  # Latest Kalman gain
    log_likelihood: float  # Filter log-likelihood
    is_outlier: bool  # Whether latest observation was rejected
    velocity_estimate: float  # Price velocity (trend speed)
    acceleration_estimate: float  # Price acceleration (trend change)
    velocity_confidence: float  # Confidence in velocity estimate
    acceleration_confidence: float  # Confidence in acceleration estimate

    @property
    def velocity(self) -> float:
        return self.velocity_estimate

    @property
    def acceleration(self) -> float:
        return self.acceleration_estimate


@dataclass
class BayesianPosterior:
    """Bayesian posterior for directional probability."""
    timestamp: float
    p_up: float  # P(BTC settles UP | evidence)
    p_down: float  # P(BTC settles DOWN | evidence)
    p_neutral: float  # P(neutral/no clear direction | evidence)
    posterior_entropy: float  # Shannon entropy of posterior
    evidence_strength: float  # Total evidence weight
    prior_weight: float  # How much prior contributes
    likelihood_weight: float  # How much likelihood contributes
    posterior_variance: float  # Uncertainty in posterior
    credible_interval_up: Tuple[float, float]  # (low, high) for P(UP)
    credible_interval_down: Tuple[float, float]  # (low, high) for P(DOWN)
    bayes_factor_up_vs_down: float  # Bayes factor favoring UP
    calibration_score: float  # How well calibrated the posterior is


@dataclass
class HazardEstimate:
    """Directional hazard rate estimate."""
    timestamp: float
    up_hazard_rate: float  # Hazard rate for UP direction
    down_hazard_rate: float  # Hazard rate for DOWN direction
    continuation_decay_rate: float  # Rate at which continuation probability decays
    reversal_hazard_rate: float  # Rate at which reversal occurs
    time_to_reversal_estimate: float  # Expected time to reversal (seconds)
    settlement_hazard_up: float  # Hazard of settling UP at current state
    settlement_hazard_down: float  # Hazard of settling DOWN at current state
    hazard_volatility_adjustment: float  # Volatility adjustment to hazard rates


@dataclass
class MultiTimescaleEstimate:
    """Multi-timescale inference aggregation."""
    timestamp: float
    timescale_estimates: Dict[str, float]  # {"1s": prob, "5s": prob, "30s": prob, "1m": prob, "5m": prob}
    timescale_weights: Dict[str, float]  # Weight for each timescale
    weighted_direction_probability: float  # Aggregated probability
    timescale_consistency: float  # How consistent signals are across timescales
    dominant_timescale: str  # Which timescale has most weight
    short_term_signal: float  # 1-5 second signal
    medium_term_signal: float  # 30-60 second signal
    long_term_signal: float  # 1-5 minute signal
    timescale_conflict_score: float  # How much timescales disagree


@dataclass
class SettlementForecast:
    """Final settlement probability forecast."""
    timestamp: float
    p_up_settlement: float  # P(BTC UP at settlement)
    p_down_settlement: float  # P(BTC DOWN at settlement)
    time_to_settlement_seconds: float  # Remaining time
    settlement_confidence: float  # Confidence in forecast
    settlement_uncertainty: float  # Uncertainty range
    settlement_probability_range: Tuple[float, float]  # (low, high) for UP
    expected_settlement_direction: Direction  # Most likely outcome
    edge_estimate: float  # Directional edge: max(p_up, p_down) - 0.5
    edge_confidence: float  # Confidence in directional edge
    forecast_quality: float  # Overall forecast quality
    execution_recommended: bool  # Whether to execute
    execution_direction: Direction  # Recommended directional side, or NEUTRAL if signal is weak
    execution_size_fraction: float  # Recommended size (0-1)
    execution_confidence: float  # Confidence in execution recommendation
    btc_price: Optional[float] = None  # Spot price used by the terminal model
    price_to_beat: Optional[float] = None  # Polymarket/Chainlink PTB
    terminal_z_score: float = 0.0  # Standardized log-distance to PTB at expiry
    terminal_sigma_per_sqrt_second: float = 0.0  # Log-return volatility scale
    terminal_drift_per_second: float = 0.0  # Log-price drift used by forecast
    terminal_distance_log: float = 0.0  # log(spot / PTB)
    market_price_up: Optional[float] = None  # Current UP ask used for edge
    market_price_down: Optional[float] = None  # Current DOWN ask used for edge
    selected_market_price: Optional[float] = None  # Ask for execution_direction, used only for execution filters
    selected_probability: Optional[float] = None  # P(selected outcome)
    min_edge_bps_required: float = 0.0  # Directional-edge gate
    reject_reason: str = ""  # Why execution_recommended is false
    observation_ready: bool = False
    observation_valid: bool = False
    observation_seconds: float = 0.0
    observation_p_up: Optional[float] = None
    observation_confidence: float = 0.0
    observation_validation_score: float = 0.0
    observation_reason: str = ""
    technical_valid: bool = False
    technical_p_up: Optional[float] = None
    technical_confidence: float = 0.0
    technical_consensus_score: float = 0.0
    technical_log_odds: float = 0.0
    technical_dominant_timeframe: str = ""
    technical_reason: str = ""
    round_direction_locked: bool = False
    round_direction: Direction = Direction.NEUTRAL
    round_direction_confidence: float = 0.0
    round_direction_p_up: Optional[float] = None
    round_direction_reason: str = ""


# ============================================================
# EXECUTION TYPES
# ============================================================

@dataclass
class ExecutionEstimate:
    """Execution realism estimate."""
    timestamp: float
    expected_fill_probability: float  # P(order gets filled)
    expected_slippage: float  # Expected slippage
    expected_slippage_bps: float  # Expected slippage in bps
    expected_latency_ms: float  # Expected latency
    effective_price: float  # Price after slippage adjustment
    spread_impact: float  # Impact of current spread
    liquidity_impact: float  # Impact of available liquidity
    regime_impact: float  # Impact of current regime on execution
    execution_quality: ExecutionQuality  # Overall execution quality
    max_recommended_size: float  # Maximum position size given conditions
    fill_time_estimate: float  # Estimated time to fill (seconds)
    limit_price: float = 0.0  # Limit price needed for estimated average fill
    visible_depth_notional: float = 0.0  # Visible ask notional used by model
    best_ask_price: Optional[float] = None  # Best ask on the selected token


@dataclass
class TradeDecision:
    """Final trade decision output."""
    timestamp: float
    direction: Direction  # UP or DOWN
    token_id: str  # Polymarket token to buy
    probability: float  # Our estimated probability
    market_price: float  # Current market price for this direction
    edge: float  # probability - market_price
    edge_bps: float  # Edge in basis points
    confidence: float  # Overall confidence 0-1
    position_size: float  # Position size in USDC
    fill_probability: float  # Expected fill probability
    expected_slippage_bps: float  # Expected slippage
    settlement_forecast: SettlementForecast  # Full forecast
    execution_estimate: ExecutionEstimate  # Execution estimate
    regime_state: RegimeState  # Current regime
    signal_type: SignalType  # What drove this decision
    is_dry_run: bool  # Whether this is a dry run
    reason: str  # Human-readable reason for decision
    should_trade: bool  # Whether to actually execute


# ============================================================
# RISK TYPES
# ============================================================

@dataclass
class RiskState:
    """Current risk management state."""
    timestamp: float
    current_position: float  # Current position size
    current_position_direction: Direction  # Current position direction
    unrealized_pnl: float  # Unrealized P&L
    realized_pnl_session: float  # Session realized P&L
    consecutive_losses: int  # Consecutive loss count
    total_trades_session: int  # Total trades this session
    win_rate_session: float  # Session win rate
    max_drawdown: float  # Maximum drawdown
    current_drawdown: float  # Current drawdown from peak
    risk_score: float  # 0-1 overall risk score
    is_in_cooldown: bool  # Whether in cooldown period
    cooldown_remaining_seconds: float  # Remaining cooldown time
    available_capital: float  # Available capital for trading
    position_limit_remaining: float  # Remaining position limit
    should_stop_trading: bool  # Whether to stop trading entirely


# ============================================================
# COMPOSITE STATE TYPES
# ============================================================

@dataclass
class MarketState:
    """Complete market state at a point in time - the central data structure."""
    timestamp: float
    
    # Raw data
    btc_price: float
    market_slug: str = ""
    market_question: str = ""
    market_end_iso: str = ""
    price_to_beat: Optional[float] = None
    price_source: str = ""
    btc_orderbook: Optional[OrderbookSnapshot] = None
    polymarket_orderbook_up: Optional[PolymarketOrderbook] = None
    polymarket_orderbook_down: Optional[PolymarketOrderbook] = None
    
    # Microstructure
    orderbook_pressure: Optional[OrderbookPressure] = None
    flow_metrics: Optional[FlowMetrics] = None
    liquidity_metrics: Optional[LiquidityMetrics] = None
    queue_metrics: Optional[QueueMetrics] = None
    spread_metrics: Optional[SpreadMetrics] = None
    
    # Regime & Volatility
    regime_state: Optional[RegimeStateEstimate] = None
    volatility_estimate: Optional[VolatilityEstimate] = None
    
    # Signals
    continuation_signal: Optional[ContinuationSignal] = None
    reversal_signal: Optional[ReversalSignal] = None
    exhaustion_signal: Optional[ExhaustionSignal] = None
    burst_failure_signal: Optional[BurstFailureSignal] = None
    entropy_filter: Optional[EntropyFilterResult] = None
    observation_signal: Optional[ObservationSignal] = None
    technical_signal: Optional[TechnicalMomentumSignal] = None
    
    # Inference
    kalman_state: Optional[KalmanState] = None
    bayesian_posterior: Optional[BayesianPosterior] = None
    hazard_estimate: Optional[HazardEstimate] = None
    multi_timescale: Optional[MultiTimescaleEstimate] = None
    settlement_forecast: Optional[SettlementForecast] = None
    
    # Execution
    execution_estimate: Optional[ExecutionEstimate] = None
    
    # Risk
    risk_state: Optional[RiskState] = None
    
    # Decision
    trade_decision: Optional[TradeDecision] = None


# ============================================================
# HISTORY BUFFERS
# ============================================================

class TickBuffer:
    """Circular buffer for tick data with numpy-backed arrays for performance."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.timestamps = deque(maxlen=max_size)
        self.prices = deque(maxlen=max_size)
        self.volumes = deque(maxlen=max_size)
        self.sides = deque(maxlen=max_size)
        self._price_array_cache = None
        self._cache_valid = False
    
    def append(self, tick: TickData):
        self.timestamps.append(tick.timestamp)
        self.prices.append(tick.price)
        self.volumes.append(tick.volume)
        self.sides.append(tick.side)
        self._cache_valid = False
    
    def get_price_array(self) -> np.ndarray:
        if not self._cache_valid:
            self._price_array_cache = np.array(self.prices, dtype=np.float64)
            self._cache_valid = True
        return self._price_array_cache
    
    def get_volume_array(self) -> np.ndarray:
        return np.array(self.volumes, dtype=np.float64)
    
    def get_timestamp_array(self) -> np.ndarray:
        return np.array(self.timestamps, dtype=np.float64)
    
    def __len__(self) -> int:
        return len(self.timestamps)

    def len(self) -> int:
        return len(self)
    
    def is_empty(self) -> bool:
        return len(self.timestamps) == 0
    
    def last_price(self) -> Optional[float]:
        return self.prices[-1] if self.prices else None
    
    def last_timestamp(self) -> Optional[float]:
        return self.timestamps[-1] if self.timestamps else None
    
    def recent_slice(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get last n (timestamps, prices, volumes) as numpy arrays."""
        n = min(n, len(self))
        return (
            np.array(list(self.timestamps)[-n:], dtype=np.float64),
            np.array(list(self.prices)[-n:], dtype=np.float64),
            np.array(list(self.volumes)[-n:], dtype=np.float64),
        )

    def price_at_timestamp(
        self,
        target_timestamp: float,
        max_skew_seconds: float = 2.0,
        max_gap_seconds: float = 3.0,
    ) -> Tuple[Optional[float], str]:
        """Return nearest/interpolated price for a timestamp if surrounding ticks are sane."""
        if not self.timestamps:
            return None, "no_ticks"

        timestamps = np.array(self.timestamps, dtype=np.float64)
        prices = np.array(self.prices, dtype=np.float64)
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        prices = prices[order]

        now_min = timestamps[0]
        now_max = timestamps[-1]
        if target_timestamp < now_min - max_skew_seconds:
            return None, "target_before_ticks"
        if target_timestamp > now_max + max_skew_seconds:
            return None, "target_after_ticks"

        idx = int(np.searchsorted(timestamps, target_timestamp))
        if idx < len(timestamps) and abs(timestamps[idx] - target_timestamp) <= max_skew_seconds:
            return float(prices[idx]), "nearest"
        if idx > 0 and abs(timestamps[idx - 1] - target_timestamp) <= max_skew_seconds:
            return float(prices[idx - 1]), "nearest"

        if 0 < idx < len(timestamps):
            t0 = float(timestamps[idx - 1])
            t1 = float(timestamps[idx])
            if t1 - t0 <= max_gap_seconds and t0 <= target_timestamp <= t1:
                weight = (target_timestamp - t0) / max(t1 - t0, 1e-9)
                price = float(prices[idx - 1] + weight * (prices[idx] - prices[idx - 1]))
                return price, "interpolated"
            return None, "gap_too_large"

        return None, "no_bracket"

    def has_recent_gap(self, max_gap_seconds: float = 5.0, lookback: int = 20) -> bool:
        n = min(lookback, len(self.timestamps))
        if n < 2:
            return False
        timestamps = np.array(list(self.timestamps)[-n:], dtype=np.float64)
        timestamps = np.sort(timestamps)
        gaps = np.diff(timestamps)
        return bool(np.any(gaps > max_gap_seconds))


class OrderbookBuffer:
    """Buffer for orderbook snapshots with delta tracking."""
    
    def __init__(self, max_snapshots: int = 100, max_deltas: int = 5000):
        self.max_snapshots = max_snapshots
        self.max_deltas = max_deltas
        self.snapshots = deque(maxlen=max_snapshots)
        self.deltas = deque(maxlen=max_deltas)
        self.cancellation_events = deque(maxlen=1000)
        self.addition_events = deque(maxlen=1000)
    
    def add_snapshot(self, snapshot: OrderbookSnapshot):
        self.snapshots.append(snapshot)
    
    def add_delta(self, delta: OrderbookDelta):
        self.deltas.append(delta)
        if delta.size_delta < 0 and not delta.is_trade:
            self.cancellation_events.append(delta)
        elif delta.size_delta > 0:
            self.addition_events.append(delta)
    
    def recent_snapshots(self, n: int) -> List[OrderbookSnapshot]:
        n = min(n, len(self.snapshots))
        return list(self.snapshots)[-n:]
    
    def recent_deltas(self, n: int) -> List[OrderbookDelta]:
        n = min(n, len(self.deltas))
        return list(self.deltas)[-n:]


class SignalBuffer:
    """Buffer for signal history with decorrelation tracking."""
    
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.continuation_signals = deque(maxlen=max_size)
        self.reversal_signals = deque(maxlen=max_size)
        self.direction_probabilities = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
    
    def add_signal(self, timestamp: float, direction_prob: float,
                   continuation: Optional[ContinuationSignal] = None,
                   reversal: Optional[ReversalSignal] = None):
        self.timestamps.append(timestamp)
        self.direction_probabilities.append(direction_prob)
        if continuation:
            self.continuation_signals.append(continuation)
        if reversal:
            self.reversal_signals.append(reversal)

    def __len__(self) -> int:
        return len(self.direction_probabilities)

    def len(self) -> int:
        return len(self)
    
    def recent_probabilities(self, n: int) -> np.ndarray:
        n = min(n, len(self.direction_probabilities))
        return np.array(list(self.direction_probabilities)[-n:], dtype=np.float64)
    
    def recent_timestamps(self, n: int) -> np.ndarray:
        n = min(n, len(self.timestamps))
        return np.array(list(self.timestamps)[-n:], dtype=np.float64)


class RegimeBuffer:
    """Buffer for regime state history."""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.regime_states = deque(maxlen=max_size)
        self.regime_probabilities = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
    
    def add(self, regime: RegimeStateEstimate):
        self.regime_states.append(regime.current_regime)
        self.regime_probabilities.append(regime.regime_probabilities)
        self.timestamps.append(regime.timestamp)
    
    def recent_regimes(self, n: int) -> List[RegimeState]:
        n = min(n, len(self.regime_states))
        return list(self.regime_states)[-n:]
