import math
import time

import numpy as np

from polymarket_bot.config import BotConfig, Direction, RegimeState
from polymarket_bot.bot import TradingBot
from polymarket_bot.decision.decision_engine import DecisionEngine
from polymarket_bot.inference.bayesian_fusion import BayesianFusion
from polymarket_bot.inference.posterior_calibrator import PosteriorCalibrator
from polymarket_bot.inference.settlement_forecast import SettlementForecaster
from polymarket_bot.strategy.observation_window import ObservationWindowStrategy
from polymarket_bot.strategy.technical_momentum import TechnicalMomentumStrategy
from polymarket_bot.tui import _time_to_settlement_from_state
from polymarket_bot.bot_types import (
    BayesianPosterior,
    ExecutionEstimate,
    ExecutionQuality,
    FlowMetrics,
    HazardEstimate,
    KalmanState,
    MarketState,
    MultiTimescaleEstimate,
    ObservationSignal,
    OrderbookPressure,
    RegimeStateEstimate,
    SettlementForecast,
    TechnicalMomentumSignal,
    TickBuffer,
    TickData,
    VolatilityEstimate,
)


def _volatility(
    second_vol: float = 6.0e-5,
    forecast_5min: float = 1.0e-3,
) -> VolatilityEstimate:
    return VolatilityEstimate(
        timestamp=time.time(),
        realized_volatility=second_vol,
        garch_volatility=second_vol,
        stochastic_volatility=second_vol,
        implied_volatility=0.0,
        kalman_volatility=second_vol,
        regime_adjusted_volatility=second_vol,
        volatility_of_volatility=0.0,
        volatility_skew=0.0,
        volatility_forecast_5min=forecast_5min,
        volatility_confidence_interval=(second_vol * 0.75, second_vol * 1.25),
        volatility_regime_alignment=1.0,
    )


def _posterior(p_up: float = 0.5) -> BayesianPosterior:
    return BayesianPosterior(
        timestamp=time.time(),
        p_up=p_up,
        p_down=1.0 - p_up,
        p_neutral=0.0,
        posterior_entropy=0.7,
        evidence_strength=5.0,
        prior_weight=0.5,
        likelihood_weight=0.5,
        posterior_variance=0.05,
        credible_interval_up=(max(0.0, p_up - 0.1), min(1.0, p_up + 0.1)),
        credible_interval_down=(max(0.0, 1.0 - p_up - 0.1), min(1.0, 1.0 - p_up + 0.1)),
        bayes_factor_up_vs_down=1.0,
        calibration_score=0.8,
    )


def _multi(p_up: float = 0.5) -> MultiTimescaleEstimate:
    return MultiTimescaleEstimate(
        timestamp=time.time(),
        timescale_estimates={"1s": p_up, "30s": p_up},
        timescale_weights={"1s": 0.5, "30s": 0.5},
        weighted_direction_probability=p_up,
        timescale_consistency=0.9,
        dominant_timescale="30s",
        short_term_signal=p_up,
        medium_term_signal=p_up,
        long_term_signal=p_up,
        timescale_conflict_score=0.05,
    )


def _hazard(p_up: float = 0.5) -> HazardEstimate:
    return HazardEstimate(
        timestamp=time.time(),
        up_hazard_rate=p_up,
        down_hazard_rate=1.0 - p_up,
        continuation_decay_rate=0.0,
        reversal_hazard_rate=0.0,
        time_to_reversal_estimate=999.0,
        settlement_hazard_up=max(p_up, 1.0e-6),
        settlement_hazard_down=max(1.0 - p_up, 1.0e-6),
        hazard_volatility_adjustment=1.0,
    )


def _kalman(
    price: float = 100.0,
    velocity: float = 0.0,
    acceleration: float = 0.0,
) -> KalmanState:
    return KalmanState(
        timestamp=time.time(),
        state=np.array([price, velocity, acceleration], dtype=np.float64),
        covariance=np.eye(3),
        innovation=0.0,
        innovation_variance=1.0,
        kalman_gain=np.zeros(3),
        log_likelihood=0.0,
        is_outlier=False,
        velocity_estimate=velocity,
        acceleration_estimate=acceleration,
        velocity_confidence=0.5,
        acceleration_confidence=0.5,
    )


def _regime() -> RegimeStateEstimate:
    return RegimeStateEstimate(
        timestamp=time.time(),
        current_regime=RegimeState.CALM_RANGE,
        regime_probabilities={RegimeState.CALM_RANGE: 0.9},
        regime_persistence=0.8,
        regime_transition_probability={RegimeState.CALM_RANGE: 0.9},
        regime_duration_estimate=120.0,
        regime_volatility_estimate=6.0e-5,
        regime_flow_characteristic=0.0,
        regime_confidence=0.9,
        regime_entropy=0.1,
    )


def _flow(imbalance: float = 0.0, persistence: float = 1.0) -> FlowMetrics:
    return FlowMetrics(
        timestamp=time.time(),
        taker_buy_volume=1.0,
        taker_sell_volume=1.0,
        taker_ratio=0.5,
        taker_aggression_score=0.0,
        flow_imbalance=imbalance,
        flow_acceleration=0.0,
        cumulative_flow_delta=0.0,
        maker_volume=0.0,
        total_volume=2.0,
        trade_count=2,
        avg_trade_size=1.0,
        large_trade_count=0,
        large_trade_ratio=0.0,
        taker_inversion=0.0,
        flow_persistence=persistence,
        flow_decay_rate=0.0,
    )


def _pressure(real_imbalance: float = 0.0) -> OrderbookPressure:
    return OrderbookPressure(
        timestamp=time.time(),
        bid_pressure=1.0,
        ask_pressure=1.0,
        net_pressure=0.0,
        pressure_imbalance=real_imbalance,
        real_bid_pressure=1.0 + max(real_imbalance, 0.0),
        real_ask_pressure=1.0 + max(-real_imbalance, 0.0),
        real_net_pressure=real_imbalance,
        real_imbalance=real_imbalance,
        spoof_score=0.0,
        fake_liquidity_ratio=0.0,
        depth_skew=0.0,
        weighted_mid=100.0,
        microprice=100.0,
    )


def _observation(
    direction: Direction,
    p_up: float,
    confidence: float,
    valid: bool = True,
) -> ObservationSignal:
    now = time.time()
    return ObservationSignal(
        timestamp=now,
        market_slug="btc-updown-5m-1234567890",
        market_start_timestamp=now - 60.0,
        observation_start_timestamp=now - 60.0,
        observation_end_timestamp=now,
        observed_seconds=60.0,
        required_seconds=40.0,
        sample_count=60,
        effective_sample_size=30.0,
        coverage_ratio=1.0,
        max_gap_seconds=1.0,
        outlier_fraction=0.0,
        is_ready=True,
        is_valid=valid,
        direction=direction,
        p_up=p_up,
        p_down=1.0 - p_up,
        confidence=confidence,
        validation_score=0.9,
        uncertainty=0.15,
        terminal_z_score=0.0,
        drift_per_second=0.0,
        sigma_per_sqrt_second=6.0e-5,
        student_t_df=20.0,
        terminal_log_odds=math.log(p_up / (1.0 - p_up)),
        microstructure_log_odds=0.0,
        combined_log_odds=math.log(p_up / (1.0 - p_up)),
        reason="test",
    )


def _execution_estimate(price: float = 0.55) -> ExecutionEstimate:
    return ExecutionEstimate(
        timestamp=time.time(),
        expected_fill_probability=0.9,
        expected_slippage=0.0,
        expected_slippage_bps=0.0,
        expected_latency_ms=10.0,
        effective_price=price,
        spread_impact=0.0,
        liquidity_impact=0.0,
        regime_impact=0.0,
        execution_quality=ExecutionQuality.ACCEPTABLE,
        max_recommended_size=20.0,
        fill_time_estimate=1.0,
        limit_price=price,
        visible_depth_notional=100.0,
        best_ask_price=price,
    )


def _technical_signal(
    direction: Direction,
    p_up: float,
    confidence: float,
    validation_score: float = 0.9,
    consensus_score: float = 0.9,
) -> TechnicalMomentumSignal:
    log_odds = math.log(p_up / (1.0 - p_up))
    return TechnicalMomentumSignal(
        timestamp=time.time(),
        market_slug="btc-updown-5m-1234567890",
        is_valid=True,
        direction=direction,
        p_up=p_up,
        p_down=1.0 - p_up,
        confidence=confidence,
        validation_score=validation_score,
        consensus_score=consensus_score,
        conflict_score=0.05,
        log_odds=log_odds,
        rsi_log_odds=log_odds,
        macd_log_odds=log_odds,
        timeframe_probabilities={"5s": p_up},
        timeframe_weights={"5s": 1.0},
        rsi_values={"5s": 50.0},
        macd_histograms={"5s": 0.0},
        dominant_timeframe="5s",
        reason="test",
    )


def _trend_ticks(
    start: float,
    slope: float,
    curvature: float = 0.0,
    count: int = 120,
) -> TickBuffer:
    ticks = TickBuffer()
    log_price = math.log(100.0)
    for i in range(count):
        if i > 0:
            log_price += slope + curvature * i / max(count - 1, 1)
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))
    return ticks


def test_observation_neutral_zone() -> None:
    config = BotConfig()
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"

    flat = TickBuffer()
    for i in range(65):
        flat.append(TickData(timestamp=start + i, price=100.0, volume=1.0, side="buy"))

    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=flat,
        btc_price=100.0,
        price_to_beat=100.0,
        time_to_settlement=200.0,
        volatility_estimate=_volatility(),
        now=start + 65.0,
    )
    assert signal.is_ready
    assert not signal.is_valid
    assert signal.direction == Direction.NEUTRAL
    assert 0.48 <= signal.p_up <= 0.52

    strategy = ObservationWindowStrategy(config)
    trend_start = start + 600.0
    trend_slug = f"btc-updown-5m-{int(trend_start)}"
    trend = TickBuffer()
    log_price = math.log(100.0)
    for i in range(80):
        if i > 0:
            r = 2.0e-5 + 4.0e-5 * math.sin(i * 2.399) + 2.0e-5 * math.sin(i * 0.917 + 0.4)
            log_price += r
        trend.append(TickData(timestamp=trend_start + i, price=math.exp(log_price), volume=1.0, side="buy"))
    signal = strategy.evaluate(
        market_slug=trend_slug,
        tick_buffer=trend,
        btc_price=trend.last_price() or 100.0,
        price_to_beat=100.0,
        time_to_settlement=200.0,
        volatility_estimate=_volatility(),
        now=trend_start + 80.0,
    )
    assert signal.is_valid
    assert signal.direction == Direction.UP
    assert signal.p_up >= 0.62

    assert signal.observed_seconds <= 40.5

    strategy = ObservationWindowStrategy(config)
    early_start = start + 1_200.0
    early_slug = f"btc-updown-5m-{int(early_start)}"
    early_strong = TickBuffer()
    for i in range(24):
        early_strong.append(TickData(timestamp=early_start + i, price=100.0 + 0.06 * i, volume=1.0, side="buy"))
    signal = strategy.evaluate(
        market_slug=early_slug,
        tick_buffer=early_strong,
        btc_price=101.4,
        price_to_beat=100.0,
        time_to_settlement=240.0,
        volatility_estimate=_volatility(),
        now=early_start + 45.0,
    )
    assert not signal.is_valid
    assert signal.direction == Direction.NEUTRAL


def test_observation_window_cuts_at_forty_seconds() -> None:
    config = BotConfig()
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"
    ticks = TickBuffer()
    log_price = math.log(100.0)
    for i in range(120):
        if i > 0:
            log_price += 1.8e-5
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))

    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=101.0,
        price_to_beat=100.0,
        time_to_settlement=240.0,
        volatility_estimate=_volatility(),
        now=start + 119.0,
    )
    assert signal.is_ready
    assert signal.observed_seconds <= 40.5


def test_observation_accepts_full_forty_second_return_count() -> None:
    config = BotConfig()
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"
    ticks = TickBuffer()
    log_price = math.log(100.0)
    for i in range(41):
        if i > 0:
            log_price += 2.7e-5 + 2.0e-5 * math.sin(i * 0.73)
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))

    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=ticks.last_price() or 100.0,
        price_to_beat=100.0,
        time_to_settlement=220.0,
        volatility_estimate=_volatility(),
        now=start + 40.1,
    )
    assert signal.is_ready
    assert signal.sample_count >= 32
    assert "samples<" not in signal.reason


def test_observation_interpolates_cutoff_boundary() -> None:
    config = BotConfig()
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"
    ticks = TickBuffer()
    log_price = math.log(100.0)
    for i in range(40):
        if i > 0:
            log_price += 2.4e-5 + 1.6e-5 * math.sin(i * 0.67)
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))
    log_price += 2.4e-5
    ticks.append(TickData(timestamp=start + 41.0, price=math.exp(log_price), volume=1.0, side="buy"))

    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=100.20,
        price_to_beat=100.0,
        time_to_settlement=220.0,
        volatility_estimate=_volatility(),
        now=start + 40.2,
    )

    assert signal.is_ready
    assert abs(signal.observed_seconds - config.observation_window_seconds) < 1.0e-9
    assert signal.sample_count >= 32
    assert "samples<" not in signal.reason
    assert "coverage<" not in signal.reason


def test_observation_signal_is_immutable_after_cutoff() -> None:
    config = BotConfig()
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"
    ticks = TickBuffer()
    log_price = math.log(100.0)
    for i in range(41):
        if i > 0:
            log_price += 2.8e-5 + 1.2e-5 * math.sin(i * 0.71)
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))

    first = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=101.0,
        price_to_beat=100.0,
        time_to_settlement=220.0,
        volatility_estimate=_volatility(),
        now=start + 40.2,
    )

    for i in range(41, 221):
        log_price -= 2.2e-4
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="sell"))

    late_flow = _flow(0.95)
    late_flow.timestamp = start + 220.0
    late_pressure = _pressure(-0.95)
    late_pressure.timestamp = start + 220.0
    second = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=70.0,
        price_to_beat=100.0,
        time_to_settlement=20.0,
        flow_metrics=late_flow,
        orderbook_pressure=late_pressure,
        volatility_estimate=_volatility(second_vol=2.5e-4),
        now=start + 220.0,
    )

    assert second.timestamp != first.timestamp
    assert second.is_valid == first.is_valid
    assert second.direction == first.direction
    assert second.reason == first.reason
    assert math.isclose(second.p_up, first.p_up, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(second.confidence, first.confidence, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(second.combined_log_odds, first.combined_log_odds, rel_tol=0.0, abs_tol=1.0e-12)

    late_strategy = ObservationWindowStrategy(config)
    late_only = late_strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=70.0,
        price_to_beat=100.0,
        time_to_settlement=20.0,
        flow_metrics=late_flow,
        orderbook_pressure=late_pressure,
        volatility_estimate=_volatility(second_vol=2.5e-4),
        now=start + 220.0,
    )
    assert abs(late_only.microstructure_log_odds) < 1.0e-12


def test_multi_speed_ema_slope_tracks_direction_without_crossing() -> None:
    config = BotConfig()
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"

    up_ticks = _trend_ticks(start, slope=2.5e-5, count=120)
    up_signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=up_ticks,
        btc_price=up_ticks.last_price() or 100.0,
        price_to_beat=100.0,
        time_to_settlement=180.0,
        volatility_estimate=_volatility(),
        now=start + 119.0,
    )
    assert up_signal.is_ready
    assert up_signal.direction == Direction.UP
    assert up_signal.ema_slope_log_odds > 0.0
    assert up_signal.ema_slope_velocity > 0.0
    assert abs(up_signal.ema_spread_acceleration) > 0.0
    assert up_signal.ema_turning_pressure < 0.8

    down_start = start + 600.0
    down_slug = f"btc-updown-5m-{int(down_start)}"
    down_ticks = _trend_ticks(down_start, slope=-2.5e-5, count=120)
    down_signal = strategy.evaluate(
        market_slug=down_slug,
        tick_buffer=down_ticks,
        btc_price=down_ticks.last_price() or 100.0,
        price_to_beat=100.0,
        time_to_settlement=180.0,
        volatility_estimate=_volatility(),
        now=down_start + 119.0,
    )
    assert down_signal.is_ready
    assert down_signal.direction == Direction.DOWN
    assert down_signal.ema_slope_log_odds < 0.0
    assert down_signal.ema_slope_velocity < 0.0
    assert abs(down_signal.ema_spread_acceleration) > 0.0


def test_multi_speed_ema_slope_detects_turning_pressure() -> None:
    config = BotConfig()
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"

    ticks = TickBuffer()
    log_price = math.log(100.0)
    for i in range(120):
        if i < 60:
            log_price += 3.0e-5
        else:
            log_price -= 3.5e-5
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))

    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=ticks.last_price() or 100.0,
        price_to_beat=100.0,
        time_to_settlement=180.0,
        volatility_estimate=_volatility(),
        now=start + 119.0,
    )
    assert signal.is_ready
    assert signal.ema_turning_pressure >= 0.0
    assert signal.ema_slope_log_odds != 0.0


def test_volatility_forecast_unit_conversion() -> None:
    config = BotConfig()
    forecaster = SettlementForecaster(config)
    vol = _volatility(second_vol=6.0e-5, forecast_5min=1.0e-3)
    sigma = forecaster._forecast_volatility_per_sqrt_second(vol)
    naive_wrong = 0.88 * 6.0e-5 + 0.12 * 1.0e-3
    assert sigma < naive_wrong * 0.60
    assert 1.0e-6 <= sigma <= 0.005


def test_lock_enforces_direction() -> None:
    config = BotConfig()
    forecaster = SettlementForecaster(config)
    forecast = forecaster.forecast(
        posterior=_posterior(0.8),
        multi_timescale=_multi(0.8),
        hazard=_hazard(0.8),
        kalman=_kalman(100.0),
        regime_estimate=_regime(),
        volatility_estimate=_volatility(),
        time_to_settlement=180.0,
        btc_price=101.0,
        price_to_beat=100.0,
        empirical_sigma_per_sqrt_second=6.0e-5,
        empirical_drift_per_second=0.0,
        empirical_sample_count=60,
        observation_signal=_observation(Direction.UP, 0.8, 0.8),
        round_direction=Direction.DOWN,
        round_direction_p_up=0.30,
        round_direction_confidence=0.85,
        round_direction_reason="test",
    )
    assert forecast.round_direction_locked
    assert forecast.expected_settlement_direction == Direction.DOWN
    assert forecast.execution_direction == Direction.DOWN
    assert forecast.p_up_settlement < 0.5
    assert forecast.p_down_settlement > 0.5


def test_terminal_probability_penalizes_down_tie_zone() -> None:
    config = BotConfig()
    forecaster = SettlementForecaster(config)
    tie = forecaster.forecast(
        posterior=_posterior(0.5),
        multi_timescale=_multi(0.5),
        hazard=_hazard(0.5),
        kalman=_kalman(100.0),
        regime_estimate=_regime(),
        volatility_estimate=_volatility(),
        time_to_settlement=90.0,
        btc_price=100.0,
        price_to_beat=100.0,
        empirical_sigma_per_sqrt_second=6.0e-5,
        empirical_drift_per_second=0.0,
        empirical_sample_count=60,
        observation_signal=_observation(Direction.UP, 0.5, 0.5),
    )
    assert tie.p_up_settlement > 0.5
    assert tie.p_down_settlement < 0.5
    assert tie.expected_settlement_direction == Direction.UP


def test_forecast_waits_for_round_lock_before_entry() -> None:
    config = BotConfig()
    config.require_round_direction_lock_for_entry = True
    forecaster = SettlementForecaster(config)
    forecast = forecaster.forecast(
        posterior=_posterior(0.82),
        multi_timescale=_multi(0.82),
        hazard=_hazard(0.82),
        kalman=_kalman(101.0, velocity=0.02),
        regime_estimate=_regime(),
        volatility_estimate=_volatility(),
        time_to_settlement=180.0,
        btc_price=101.0,
        price_to_beat=100.0,
        empirical_sigma_per_sqrt_second=6.0e-5,
        empirical_drift_per_second=0.0,
        empirical_sample_count=60,
        observation_signal=_observation(Direction.UP, 0.78, 0.80),
    )
    assert not forecast.round_direction_locked
    assert not forecast.execution_recommended
    assert forecast.execution_direction == Direction.UP
    assert forecast.execution_size_fraction == 0.0
    assert forecast.reject_reason == "round_direction_lock_pending"


def test_forecast_rejects_inside_settlement_buffer_even_with_lock() -> None:
    config = BotConfig()
    config.risk.settlement_time_buffer_seconds = 30.0
    forecaster = SettlementForecaster(config)
    forecast = forecaster.forecast(
        posterior=_posterior(0.90),
        multi_timescale=_multi(0.90),
        hazard=_hazard(0.90),
        kalman=_kalman(102.0, velocity=0.03),
        regime_estimate=_regime(),
        volatility_estimate=_volatility(),
        time_to_settlement=10.0,
        btc_price=102.0,
        price_to_beat=100.0,
        empirical_sigma_per_sqrt_second=6.0e-5,
        empirical_drift_per_second=0.0,
        empirical_sample_count=60,
        observation_signal=_observation(Direction.UP, 0.90, 0.90),
        round_direction=Direction.UP,
        round_direction_p_up=0.92,
        round_direction_confidence=0.94,
        round_direction_reason="test",
    )
    assert forecast.round_direction_locked
    assert forecast.expected_settlement_direction == Direction.UP
    assert forecast.execution_direction == Direction.UP
    assert not forecast.execution_recommended
    assert forecast.execution_size_fraction == 0.0
    assert "settlement_time_buffer" in forecast.reject_reason


def test_decision_rejects_unlocked_directional_signal() -> None:
    config = BotConfig()
    config.require_round_direction_lock_for_entry = True
    engine = DecisionEngine(config)
    forecast = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.70,
        p_down_settlement=0.30,
        time_to_settlement_seconds=200.0,
        settlement_confidence=0.80,
        settlement_uncertainty=0.10,
        settlement_probability_range=(0.60, 0.80),
        expected_settlement_direction=Direction.UP,
        edge_estimate=0.20,
        edge_confidence=0.70,
        forecast_quality=0.80,
        execution_recommended=True,
        execution_direction=Direction.UP,
        execution_size_fraction=0.10,
        execution_confidence=0.70,
        selected_probability=0.70,
        observation_ready=True,
        observation_valid=True,
        observation_seconds=60.0,
        observation_p_up=0.70,
        observation_confidence=0.80,
        observation_validation_score=0.90,
        observation_reason="test",
    )
    decision = engine.make_decision(
        settlement_forecast=forecast,
        execution_estimate=_execution_estimate(0.48),
        market_price_up=0.48,
        market_price_down=0.52,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug="btc-updown-5m-test",
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )
    assert not decision.should_trade
    assert decision.position_size == 0.0
    assert "round_direction_lock_pending" in decision.reason


def test_decision_keeps_valid_direction_before_lock() -> None:
    config = BotConfig()
    engine = DecisionEngine(config)
    forecast = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.70,
        p_down_settlement=0.30,
        time_to_settlement_seconds=200.0,
        settlement_confidence=0.80,
        settlement_uncertainty=0.10,
        settlement_probability_range=(0.60, 0.80),
        expected_settlement_direction=Direction.UP,
        edge_estimate=0.20,
        edge_confidence=0.70,
        forecast_quality=0.80,
        execution_recommended=True,
        execution_direction=Direction.UP,
        execution_size_fraction=0.10,
        execution_confidence=0.70,
        selected_probability=0.70,
        observation_ready=True,
        observation_valid=True,
        observation_seconds=60.0,
        observation_p_up=0.70,
        observation_confidence=0.80,
        observation_validation_score=0.90,
        observation_reason="test",
        reject_reason="round_direction_locked(test)",
    )
    direction, probability, _price, _edge, _edge_bps = engine._decision_values(
        forecast,
        market_price_up=0.50,
        market_price_down=0.50,
    )
    assert direction == Direction.UP
    assert probability == 0.70


def test_decision_rejects_negative_market_edge_before_observation_conflict() -> None:
    config = BotConfig()
    engine = DecisionEngine(config)
    forecast = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.732,
        p_down_settlement=0.268,
        time_to_settlement_seconds=167.0,
        settlement_confidence=0.64,
        settlement_uncertainty=0.12,
        settlement_probability_range=(0.63, 0.80),
        expected_settlement_direction=Direction.UP,
        edge_estimate=0.232,
        edge_confidence=0.62,
        forecast_quality=0.75,
        execution_recommended=True,
        execution_direction=Direction.UP,
        execution_size_fraction=0.08,
        execution_confidence=0.62,
        market_price_up=0.78,
        market_price_down=0.22,
        selected_market_price=0.78,
        selected_probability=0.732,
        observation_ready=True,
        observation_valid=False,
        observation_seconds=39.0,
        observation_p_up=0.583,
        observation_confidence=0.25,
        observation_validation_score=0.91,
        observation_reason="edge<0.657",
        reject_reason="observation_soft_gate(observation_invalid(edge<0.657))",
    )

    decision = engine.make_decision(
        settlement_forecast=forecast,
        execution_estimate=_execution_estimate(0.78),
        market_price_up=0.78,
        market_price_down=0.22,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug="btc-updown-5m-test",
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )

    assert not decision.should_trade
    assert decision.edge_bps > 0.0
    assert "OBS_INVALID" in decision.reason


def test_decision_rejects_execution_quality_reject() -> None:
    config = BotConfig()
    engine = DecisionEngine(config)
    forecast = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.923,
        p_down_settlement=0.077,
        time_to_settlement_seconds=180.0,
        settlement_confidence=0.90,
        settlement_uncertainty=0.05,
        settlement_probability_range=(0.88, 0.96),
        expected_settlement_direction=Direction.UP,
        edge_estimate=0.423,
        edge_confidence=0.90,
        forecast_quality=0.90,
        execution_recommended=True,
        execution_direction=Direction.UP,
        execution_size_fraction=0.10,
        execution_confidence=0.90,
        market_price_up=0.64,
        market_price_down=0.36,
        selected_market_price=0.64,
        selected_probability=0.923,
        observation_ready=True,
        observation_valid=True,
        observation_seconds=60.0,
        observation_p_up=0.923,
        observation_confidence=0.90,
        observation_validation_score=0.95,
        observation_reason="valid_test",
        round_direction_locked=True,
        round_direction=Direction.UP,
        round_direction_confidence=0.91,
        round_direction_p_up=0.923,
        round_direction_reason="test",
    )
    execution = _execution_estimate(0.64)
    execution.execution_quality = ExecutionQuality.REJECT
    execution.max_recommended_size = 0.0

    decision = engine.make_decision(
        settlement_forecast=forecast,
        execution_estimate=execution,
        market_price_up=0.64,
        market_price_down=0.36,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug="btc-updown-5m-test",
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )
    assert not decision.should_trade
    assert decision.position_size == 0.0
    assert "execution_quality_reject" in decision.reason
    assert "exec=reject" in decision.reason


def test_decision_rejects_inside_settlement_buffer() -> None:
    config = BotConfig()
    config.risk.settlement_time_buffer_seconds = 30.0
    engine = DecisionEngine(config)
    forecast = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.95,
        p_down_settlement=0.05,
        time_to_settlement_seconds=11.0,
        settlement_confidence=0.90,
        settlement_uncertainty=0.05,
        settlement_probability_range=(0.90, 0.98),
        expected_settlement_direction=Direction.UP,
        edge_estimate=0.45,
        edge_confidence=0.90,
        forecast_quality=0.92,
        execution_recommended=True,
        execution_direction=Direction.UP,
        execution_size_fraction=0.10,
        execution_confidence=0.90,
        market_price_up=0.55,
        market_price_down=0.45,
        selected_market_price=0.55,
        selected_probability=0.95,
        observation_ready=True,
        observation_valid=True,
        observation_seconds=60.0,
        observation_p_up=0.95,
        observation_confidence=0.90,
        observation_validation_score=0.95,
        observation_reason="valid_test",
        round_direction_locked=True,
        round_direction=Direction.UP,
        round_direction_confidence=0.92,
        round_direction_p_up=0.95,
        round_direction_reason="test",
    )
    decision = engine.make_decision(
        settlement_forecast=forecast,
        execution_estimate=_execution_estimate(0.55),
        market_price_up=0.55,
        market_price_down=0.45,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug="btc-updown-5m-test",
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )
    assert not decision.should_trade
    assert decision.position_size == 0.0
    assert "settlement_time_buffer" in decision.reason


def test_round_lock_invalidates_when_live_evidence_conflicts() -> None:
    config = BotConfig()
    bot = TradingBot(config)
    market_slug = "btc-updown-5m-test"
    bot._round_direction_locks[market_slug] = {
        "market_slug": market_slug,
        "timestamp": time.time(),
        "direction": Direction.DOWN,
        "p_up": 0.20,
        "confidence": 0.80,
        "log_odds": math.log(0.20 / 0.80),
        "reason": "seed",
    }
    conflict = bot._get_round_direction_lock(
        market_slug=market_slug,
        btc_price=102.0,
        price_to_beat=100.0,
        time_to_settlement=120.0,
        observation_signal=_observation(Direction.UP, 0.88, 0.90),
        technical_signal=_technical_signal(Direction.UP, 0.86, 0.88),
        empirical_sigma=6.0e-5,
        empirical_drift=0.0,
    )
    assert conflict is None
    assert market_slug not in bot._round_direction_locks


def test_entry_waits_for_live_price_pullback() -> None:
    config = BotConfig()
    engine = DecisionEngine(config)
    slug = "btc-updown-5m-test"

    forecast = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.81,
        p_down_settlement=0.19,
        time_to_settlement_seconds=180.0,
        settlement_confidence=0.88,
        settlement_uncertainty=0.08,
        settlement_probability_range=(0.74, 0.88),
        expected_settlement_direction=Direction.UP,
        edge_estimate=0.31,
        edge_confidence=0.82,
        forecast_quality=0.86,
        execution_recommended=True,
        execution_direction=Direction.UP,
        execution_size_fraction=0.10,
        execution_confidence=0.84,
        market_price_up=0.78,
        market_price_down=0.22,
        selected_market_price=0.78,
        selected_probability=0.81,
        observation_ready=True,
        observation_valid=True,
        observation_seconds=60.0,
        observation_p_up=0.81,
        observation_confidence=0.86,
        observation_validation_score=0.92,
        observation_reason="valid_test",
        round_direction_locked=True,
        round_direction=Direction.UP,
        round_direction_confidence=0.90,
        round_direction_p_up=0.81,
        round_direction_reason="test",
    )

    waiting = engine.make_decision(
        settlement_forecast=forecast,
        execution_estimate=_execution_estimate(0.78),
        market_price_up=0.78,
        market_price_down=0.22,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug=slug,
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )
    assert not waiting.should_trade
    assert "entry_price_wait_for_pullback" in waiting.reason

    pullback = engine.make_decision(
        settlement_forecast=forecast,
        execution_estimate=_execution_estimate(0.63),
        market_price_up=0.63,
        market_price_down=0.37,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug=slug,
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )
    assert pullback.should_trade
    assert pullback.direction == Direction.UP
    assert pullback.execution_estimate.limit_price <= config.market.max_entry_price


def test_entry_pullback_wait_allows_confirmed_flip() -> None:
    config = BotConfig()
    engine = DecisionEngine(config)
    slug = "btc-updown-5m-test"

    waiting_up = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.82,
        p_down_settlement=0.18,
        time_to_settlement_seconds=180.0,
        settlement_confidence=0.88,
        settlement_uncertainty=0.08,
        settlement_probability_range=(0.75, 0.89),
        expected_settlement_direction=Direction.UP,
        edge_estimate=0.32,
        edge_confidence=0.82,
        forecast_quality=0.86,
        execution_recommended=True,
        execution_direction=Direction.UP,
        execution_size_fraction=0.10,
        execution_confidence=0.84,
        market_price_up=0.78,
        market_price_down=0.22,
        selected_market_price=0.78,
        selected_probability=0.82,
        observation_ready=True,
        observation_valid=True,
        observation_seconds=60.0,
        observation_p_up=0.82,
        observation_confidence=0.86,
        observation_validation_score=0.92,
        observation_reason="valid_test",
        round_direction_locked=True,
        round_direction=Direction.UP,
        round_direction_confidence=0.90,
        round_direction_p_up=0.82,
        round_direction_reason="test",
    )
    decision = engine.make_decision(
        settlement_forecast=waiting_up,
        execution_estimate=_execution_estimate(0.78),
        market_price_up=0.78,
        market_price_down=0.22,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug=slug,
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )
    assert not decision.should_trade
    assert "entry_price_wait_for_pullback" in decision.reason

    flipped_down = SettlementForecast(
        timestamp=time.time(),
        p_up_settlement=0.18,
        p_down_settlement=0.82,
        time_to_settlement_seconds=160.0,
        settlement_confidence=0.89,
        settlement_uncertainty=0.08,
        settlement_probability_range=(0.11, 0.25),
        expected_settlement_direction=Direction.DOWN,
        edge_estimate=0.32,
        edge_confidence=0.83,
        forecast_quality=0.87,
        execution_recommended=True,
        execution_direction=Direction.DOWN,
        execution_size_fraction=0.10,
        execution_confidence=0.85,
        market_price_up=0.37,
        market_price_down=0.63,
        selected_market_price=0.63,
        selected_probability=0.82,
        observation_ready=True,
        observation_valid=True,
        observation_seconds=80.0,
        observation_p_up=0.18,
        observation_confidence=0.87,
        observation_validation_score=0.93,
        observation_reason="valid_flip",
        round_direction_locked=True,
        round_direction=Direction.DOWN,
        round_direction_confidence=0.91,
        round_direction_p_up=0.18,
        round_direction_reason="test_flip",
    )
    decision = engine.make_decision(
        settlement_forecast=flipped_down,
        execution_estimate=_execution_estimate(0.63),
        market_price_up=0.37,
        market_price_down=0.63,
        token_id_up="up-token",
        token_id_down="down-token",
        market_slug=slug,
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )
    assert decision.should_trade
    assert decision.direction == Direction.DOWN
    assert decision.execution_estimate.limit_price <= config.market.max_entry_price


def test_rsi50_macd50_confirmation_filter() -> None:
    config = BotConfig()
    strategy = TechnicalMomentumStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"

    up = TickBuffer()
    log_price = math.log(100.0)
    for i in range(180):
        if i > 0:
            log_price += 2.5e-5 + 4.0e-5 * math.sin(i * 0.61)
        up.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))
    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=up,
        time_to_settlement=120.0,
        now=start + 180.0,
    )
    assert signal.rsi_values["RSI50"] > 50.0
    assert signal.macd_histograms["MACD50"] > 0.0
    assert signal.direction == Direction.UP
    assert signal.is_valid

    conflict = TickBuffer()
    log_price = math.log(100.0)
    for i in range(180):
        if i > 0:
            drift = 4.5e-5 if i < 120 else -2.0e-4
            log_price += drift + 3.0e-5 * math.sin(i * 0.53)
        conflict.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))
    strategy = TechnicalMomentumStrategy(config)
    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=conflict,
        time_to_settlement=20.0,
        now=start + 180.0,
    )
    if signal.direction == Direction.DOWN:
        assert not signal.is_valid or signal.rsi_values["RSI50"] < 50.0
    else:
        assert not signal.is_valid or signal.macd_histograms["MACD50"] > 0.0


def test_invalid_ema_does_not_inflate_observation_confidence() -> None:
    config = BotConfig()
    config.observation_ema_slope_enabled = False
    strategy = ObservationWindowStrategy(config)
    start = 1_234_567_890.0
    slug = f"btc-updown-5m-{int(start)}"
    ticks = TickBuffer()
    log_price = math.log(100.0)
    for i in range(41):
        ticks.append(TickData(timestamp=start + i, price=math.exp(log_price), volume=1.0, side="buy"))

    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        btc_price=ticks.last_price() or 100.0,
        price_to_beat=100.0,
        time_to_settlement=220.0,
        volatility_estimate=_volatility(),
        now=start + 40.1,
    )

    assert signal.ema_slope_log_odds == 0.0
    assert not signal.is_valid
    assert signal.confidence < config.observation_min_direction_confidence
    assert "conf<" in signal.reason


def test_technical_hysteresis_blocks_weak_flip_without_amplifying() -> None:
    config = BotConfig()
    config.technical_momentum_rsi50_macd50_required = False
    config.technical_momentum_smoothing_alpha = 1.0
    config.technical_momentum_hysteresis_log_odds = 0.22
    strategy = TechnicalMomentumStrategy(config)
    slug = "btc-updown-5m-test"
    strategy._state_by_slug[slug] = {
        "last_log_odds": 0.0,
        "last_direction": Direction.UP,
    }

    p_up = strategy._sigmoid(-0.01)
    strategy._timeframe_signal = lambda prices, timeframe: (
        p_up,
        -0.004,
        49.8,
        -1.0e-7,
        1.0,
        0.02,
        -0.01,
        0.0,
    )
    strategy._rsi50_macd50_confirmation = lambda prices: (
        Direction.NEUTRAL,
        50.0,
        0.0,
        0.0,
        False,
        "disabled",
    )

    ticks = _trend_ticks(1_234_567_890.0, slope=0.0, count=180)
    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        time_to_settlement=120.0,
        now=1_234_568_069.0,
    )

    assert abs(signal.log_odds) < 1.0e-12
    assert signal.p_up == 0.5


def test_technical_invalid_signal_does_not_update_hysteresis_state() -> None:
    config = BotConfig()
    config.technical_momentum_rsi50_macd50_required = False
    config.technical_momentum_max_conflict = -0.01
    strategy = TechnicalMomentumStrategy(config)
    slug = "btc-updown-5m-invalid-state"
    strategy._state_by_slug[slug] = {
        "last_log_odds": 0.25,
        "last_direction": Direction.UP,
    }

    p_up = strategy._sigmoid(0.35)
    strategy._timeframe_signal = lambda prices, timeframe: (
        p_up,
        0.15,
        55.0,
        1.0e-7,
        1.0,
        0.25,
        0.25,
        0.10,
    )
    strategy._rsi50_macd50_confirmation = lambda prices: (
        Direction.UP,
        55.0,
        1.0e-7,
        1.0,
        True,
        "mock",
    )

    ticks = _trend_ticks(1_234_567_890.0, slope=0.0, count=180)
    signal = strategy.evaluate(
        market_slug=slug,
        tick_buffer=ticks,
        time_to_settlement=120.0,
        now=1_234_568_069.0,
    )

    assert not signal.is_valid
    assert signal.validation_score >= 0.42
    assert strategy._state_by_slug[slug]["last_log_odds"] == 0.25
    assert strategy._state_by_slug[slug]["last_direction"] == Direction.UP


def test_bayesian_reset_clears_round_smoothing() -> None:
    config = BotConfig()
    fusion = BayesianFusion(config.bayesian)
    calibrator = PosteriorCalibrator(config.bayesian)
    regime = _regime()

    strong_down = fusion.fuse(
        kalman_state=_kalman(100.0, velocity=-0.20),
        flow_metrics=_flow(-0.8, persistence=1.0),
        orderbook_pressure=_pressure(-0.8),
        regime_estimate=regime,
        volatility_estimate=_volatility(),
    )
    strong_down = calibrator.calibrate(strong_down, time.time())
    assert strong_down.p_down > 0.75

    fusion.reset()
    calibrator.reset()
    neutral = fusion.fuse(
        kalman_state=_kalman(100.0, velocity=0.0),
        flow_metrics=_flow(0.0, persistence=1.0),
        orderbook_pressure=_pressure(0.0),
        regime_estimate=regime,
        volatility_estimate=_volatility(),
    )
    neutral = calibrator.calibrate(neutral, time.time())
    assert 0.49 <= neutral.p_up <= 0.51
    assert 0.49 <= neutral.p_down <= 0.51


def test_tui_settlement_timer_uses_slug_before_forecast() -> None:
    start = 1_700_000_000.0
    now = start + 24.5
    state = MarketState(
        timestamp=now,
        btc_price=100_000.0,
        market_slug=f"btc-updown-5m-{int(start)}",
        price_to_beat=99_950.0,
    )

    tau = _time_to_settlement_from_state(state, None, now=now)
    assert tau is not None
    assert abs(tau - 275.5) < 1.0e-9


if __name__ == "__main__":
    tests = [
        test_observation_neutral_zone,
        test_observation_window_cuts_at_forty_seconds,
        test_observation_accepts_full_forty_second_return_count,
        test_observation_interpolates_cutoff_boundary,
        test_observation_signal_is_immutable_after_cutoff,
        test_multi_speed_ema_slope_tracks_direction_without_crossing,
        test_multi_speed_ema_slope_detects_turning_pressure,
        test_volatility_forecast_unit_conversion,
        test_lock_enforces_direction,
        test_terminal_probability_penalizes_down_tie_zone,
        test_forecast_waits_for_round_lock_before_entry,
        test_forecast_rejects_inside_settlement_buffer_even_with_lock,
        test_decision_rejects_unlocked_directional_signal,
        test_decision_keeps_valid_direction_before_lock,
        test_decision_rejects_negative_market_edge_before_observation_conflict,
        test_decision_rejects_execution_quality_reject,
        test_decision_rejects_inside_settlement_buffer,
        test_round_lock_invalidates_when_live_evidence_conflicts,
        test_entry_waits_for_live_price_pullback,
        test_entry_pullback_wait_allows_confirmed_flip,
        test_rsi50_macd50_confirmation_filter,
        test_invalid_ema_does_not_inflate_observation_confidence,
        test_technical_hysteresis_blocks_weak_flip_without_amplifying,
        test_technical_invalid_signal_does_not_update_hysteresis_state,
        test_bayesian_reset_clears_round_smoothing,
        test_tui_settlement_timer_uses_slug_before_forecast,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
