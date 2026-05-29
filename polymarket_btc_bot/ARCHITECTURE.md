# Polymarket BTC-UP/DOWN-5M Trading Bot - Architecture Document

## Executive Summary

This document describes a production-grade directional trading bot for Polymarket's BTC-UP/DOWN-5M binary options markets. The bot forecasts short-term BTC price direction using multi-source data fusion and executes trades when probabilistic edge exceeds defined thresholds.

**Core Objective**: Determine whether BTC-UP or BTC-DOWN has higher probability of resolving TRUE before expiration, and trade accordingly.

---

## 1. System Architecture

### 1.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRADING BOT CORE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   DNS-over-  │    │  WebSocket   │    │   Signal     │                  │
│  │   HTTPS      │───▶│  Manager     │───▶│   Engine     │                  │
│  │   Resolver   │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  External    │    │  Orderbook   │    │ Probability  │                  │
│  │  Price Feeds │    │  Sync        │    │   Model      │                  │
│  │  (Binance/   │    │              │    │              │                  │
│  │   Coinbase)  │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                           │
│         └───────────────────┼───────────────────┘                           │
│                             ▼                                               │
│                    ┌──────────────┐                                        │
│                    │   RTDS       │                                        │
│                    │   Feed       │                                        │
│                    └──────────────┘                                        │
│                             │                                               │
│                             ▼                                               │
│                    ┌──────────────┐                                        │
│                    │  Execution   │                                        │
│                    │  Engine      │                                        │
│                    └──────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Event Flow

1. **DNS Resolution Phase** (Pre-connection)
   - Resolve all WebSocket hostnames via DoH
   - Cache resolved IPs with TTL management
   - Establish TCP connections to resolved IPs

2. **WebSocket Connection Phase**
   - Connect to Polymarket orderbook WS
   - Connect to Polymarket events WS  
   - Connect to Binance BTC/USDT WS
   - Connect to Coinbase BTC/USD WS
   - Connect to RTDS feed (if available)

3. **Data Synchronization Phase**
   - Receive initial book snapshot from Polymarket
   - Subscribe to real-time updates
   - Initialize local orderbook state
   - Begin processing incremental updates

4. **Signal Generation Loop** (Continuous)
   ```
   while running:
       await new_market_data()
       update_orderbook()
       update_external_prices()
       process_rtds_update()
       
       if market_state_valid():
           signals = compute_all_signals()
           confidence = compute_confidence(signals)
           probability = compute_directional_probability(signals)
           
           if should_trade(probability, confidence):
               execute_trade(direction, size)
   ```

### 1.3 Data Flow

```
Market Data Sources → WebSocket Handlers → Normalized Events → Event Bus
                                                                    │
                                                                    ▼
                                    Orderbook State ←─ Update Queue
                                            │
                                            ▼
                              Signal Computation Pipeline
                                            │
                                            ▼
                              Probability Scoring Engine
                                            │
                                            ▼
                              Trade Decision Logic
                                            │
                                            ▼
                              Order Execution → Risk Checks
```

---

## 2. Market Microstructure Analysis

### 2.1 Polymarket Binary Options Structure

BTC-UP/DOWN-5M markets are binary options that resolve to:
- **UP**: TRUE if BTC price at expiration > price at market creation
- **DOWN**: TRUE if BTC price at expiration < price at market creation

Each outcome trades as a separate token priced between $0.00-$1.00, representing implied probability.

**Key Properties**:
- Tick size: Typically $0.01 (1 cent)
- Minimum order: Varies by market
- Expiration: 5 minutes from creation
- Settlement: Based on oracle price (typically Chainlink or similar)

### 2.2 Orderbook Dynamics

Binary option orderbooks exhibit unique characteristics:

1. **Complementary Pricing**: P(UP) + P(DOWN) ≈ 1.00 (minus spread)
2. **Mean Reversion**: Prices tend toward 0.50 as uncertainty increases near expiry
3. **Volatility Sensitivity**: Higher BTC volatility → prices move away from extremes

### 2.3 Liquidity Patterns

- **Early Life**: Wide spreads, low depth
- **Mid Life**: Tightest spreads, highest depth
- **Late Life**: Spreads widen, depth decreases as resolution approaches

---

## 3. RTDS Integration

### 3.1 What is RTDS?

RTDS (Real-Time Data Stream) provides:
- Reference pricing for market settlement
- Price-to-beat context
- Timing-sensitive market state
- Oracle feed previews

### 3.2 RTDS Data Usage

```python
class RTDSIntegration:
    """
    RTDS provides critical context for directional forecasting:
    
    1. Price-to-Beat: The reference price that determines UP/DOWN resolution
    2. Current Oracle Price: Real-time settlement price preview
    3. Time-to-Expiry: Precise countdown to resolution
    
    Integration Points:
    - Signal confirmation: External price momentum must align with orderbook signals
    - Probability scoring: RTDS price distance affects P(direction) calculation
    - Execution filtering: No trades if RTDS shows stale/orphaned data
    - No-trade filter: Skip if price-to-beat is ambiguous or missing
    """
```

### 3.3 Price-to-Beat Interpretation

```
Let:
  P_ref = Reference price (price at market creation)
  P_curr = Current BTC spot price (from RTDS/oracle)
  T_exp = Time to expiration
  
If P_curr > P_ref:
  - UP token intrinsic value increases
  - DOWN token intrinsic value decreases
  - Magnitude depends on (P_curr - P_ref) / P_ref and time remaining

Probability adjustment:
  distance_pct = (P_curr - P_ref) / P_ref * 100
  time_factor = sqrt(T_exp / 300)  # Normalize to 5-min window
  
  probability_boost = distance_pct * time_factor * sensitivity
```

---

## 4. Strategy Reasoning

### 4.1 Core Philosophy

The strategy exploits short-term directional predictability in BTC through:

1. **Orderflow Imbalance**: Aggressive buyers/sellers reveal informed views
2. **Spread Dynamics**: Spread compression/expansion signals conviction changes
3. **Liquidity Shifts**: Depth changes indicate institutional positioning
4. **Momentum Acceleration**: Rate-of-change in price signals continuation
5. **External Confirmation**: Spot market moves should lead prediction market

### 4.2 Why These Signals Matter

| Signal | What It Captures | Why It Predicts |
|--------|------------------|-----------------|
| Orderflow Imbalance | Net aggressive volume | Informed traders walk the book |
| Spread Compression | Decreasing uncertainty | Consensus forming |
| Spread Expansion | Increasing uncertainty | New information arriving |
| Bid/Ask Depth Ratio | Passive interest skew | Institutional positioning |
| Momentum | Price velocity | Trend continuation probability |
| External Price Divergence | Cross-market arbitrage | Prediction market lag |

### 4.3 Why Signals Fail

1. **Low Liquidity Regimes**: Signals become noisy when depth < threshold
2. **Near Expiry**: Mean-reversion dominates directional signals
3. **Oracle Uncertainty**: Settlement mechanism ambiguity
4. **Cross-Market Divergence**: External feeds stale or manipulated
5. **Regime Changes**: Volatility shifts invalidate historical patterns

---

## 5. Formula Explanations

### 5.1 Orderflow Imbalance (OFI)

```
OFI = (Volume_at_bid_removals - Volume_at_ask_additions) 
    + (Volume_at_ask_removals - Volume_at_bid_additions)

Normalized:
OFI_norm = OFI / (Total_bid_volume + Total_ask_volume)

Range: [-1, 1]
Interpretation: Positive = bullish pressure, Negative = bearish pressure
```

**Reasoning**: When bids are removed (hit) faster than added, and asks are added faster than removed, sellers are aggressive → bearish signal.

### 5.2 Spread Score

```
Spread_mid = (Best_ask - Best_bid) / Mid_price

Spread_score = 1.0 - min(Spread_mid / Max_expected_spread, 1.0)

Range: [0, 1]
Interpretation: Higher = tighter spread = more conviction
```

**Reasoning**: Tight spreads indicate market maker confidence and lower adverse selection risk.

### 5.3 Liquidity Imbalance

```
Bid_depth_5 = Sum(volume for bids within 5 ticks of mid)
Ask_depth_5 = Sum(volume for asks within 5 ticks of mid)

Liq_imbalance = (Bid_depth_5 - Ask_depth_5) / (Bid_depth_5 + Ask_depth_5)

Range: [-1, 1]
Interpretation: Positive = more bid support = bullish
```

### 5.4 Momentum Score

```
Returns = [log(P_t / P_{t-1}) for t in lookback_window]
Momentum = Sum(returns) / Std(returns)  # Risk-adjusted momentum

Momentum_score = tanh(Momentum)  # Bound to [-1, 1]
```

### 5.5 External Price Confirmation

```
P_poly = Polymarket implied probability (mid-price of UP token)
P_spot = Spot-implied probability from external feeds

Spot_distance = (P_spot_current - P_spot_reference) / P_spot_reference

If Spot_distance > 0 and increasing:
  External_confirmation = 1.0
Else if Spot_distance < 0 and decreasing:
  External_confirmation = -1.0
Else:
  External_confirmation = 0.0
```

### 5.6 Combined Probability Model

```
Weighted_Signal = w1*OFI_norm + w2*Spread_score + w3*Liq_imbalance 
                + w4*Momentum_score + w5*External_confirmation

Raw_Probability = 0.5 + 0.5 * tanh(Weighted_Signal * Sensitivity)

RTDS_Adjustment = f(RTDS_price_distance, time_to_expiry)

Final_Probability_UP = Raw_Probability + RTDS_Adjustment
Final_Probability_DOWN = 1.0 - Final_Probability_UP
```

### 5.7 Confidence Calculation

```
Confidence factors:
1. Signal Agreement: How many signals point same direction?
2. Signal Magnitude: Are signals strong or weak?
3. Data Quality: Is orderbook deep and fresh?
4. External Confirmation: Do spot markets agree?

Signal_agreement = (Count_positive_signals - Count_negative_signals) / Total_signals
Signal_magnitude = Mean(|signal_values|)
Data_quality = Min(1.0, Total_depth / Required_depth)
External_confirm = 1.0 if aligned else 0.5 if neutral else 0.0

Confidence = 0.5 + 0.25*Signal_agreement + 0.25*Signal_magnitude 
           + 0.25*Data_quality + 0.25*External_confirm

Range: [0, 1]
Threshold: Only trade if Confidence > confidence_threshold
```

---

## 6. Signal Weighting

### 6.1 Default Weights (Configurable)

```yaml
orderflow_weight: 0.30      # Highest - most predictive
spread_weight: 0.15         # Medium - confirms regime
liquidity_weight: 0.20      # Medium-high - shows commitment
momentum_weight: 0.20       # Medium-high - trend indicator
external_price_weight: 0.15 # Medium - cross-market check
```

### 6.2 Dynamic Weight Adjustment

Weights can be adjusted based on market regime:

```python
def adjust_weights_for_regime(regime: MarketRegime) -> Dict[str, float]:
    if regime == MarketRegime.HIGH_VOLATILITY:
        # Momentum becomes more important
        weights['momentum'] *= 1.5
        weights['liquidity'] *= 0.8  # Less reliable in chaos
    elif regime == MarketRegime.LOW_LIQUIDITY:
        # Reduce reliance on orderbook signals
        weights['orderflow'] *= 0.5
        weights['external'] *= 1.5  # External feeds more reliable
    return normalize(weights)
```

---

## 7. Trade Filtering & No-Trade Logic

### 7.1 Entry Conditions (ALL must be true)

1. `probability_UP > entry_threshold` OR `probability_DOWN > entry_threshold`
2. `confidence_score > confidence_threshold`
3. `time_to_expiry > min_time_to_expiry_seconds`
4. `spread_pct < halt_on_spread_pct`
5. `external_price_divergence < max_price_divergence_pct`
6. `daily_pnl > -daily_loss_limit_pct`
7. `concurrent_positions < max_concurrent_positions`
8. `loss_streak < max_loss_streak`

### 7.2 Rejection Conditions (ANY triggers rejection)

1. Orderbook stale (> heartbeat_timeout without update)
2. External feeds disconnected
3. RTDS data missing or flagged unreliable
4. Spread exceeds circuit breaker
5. Rate limit would be exceeded
6. Position size below minimum viable
7. Market in resolution/closed state

### 7.3 No-Trade Regimes

1. **First 30 seconds after market creation**: Insufficient data
2. **Last 30 seconds before expiry**: Mean-reversion dominates
3. **During major news events**: Unpredictable volatility
4. **After 3 consecutive losses**: Cooling-off period
5. **When daily loss limit hit**: Hard stop

---

## 8. Risk Controls

### 8.1 Position Sizing

```python
def calculate_position_size(probability: float, confidence: float) -> float:
    # Kelly criterion with fractional scaling
    kelly_fraction = (probability * (1 / (1 - probability)) - 1) if probability > 0.5 else 0
    
    # Apply confidence discount
    adjusted_kelly = kelly_fraction * confidence
    
    # Cap at maximum position size
    final_size = min(adjusted_kelly, config.max_position_size_pct)
    
    return final_size
```

### 8.2 Circuit Breakers

| Condition | Action |
|-----------|--------|
| Spread > 5% | Halt new entries |
| Price divergence > 2% | Close positions, halt |
| Heartbeat timeout | Disconnect, reconnect |
| Daily loss > 5% | Stop trading for day |
| Max drawdown > 10% | Emergency shutdown |

### 8.3 Rate Limiting

- Max 60 orders per minute (Polymarket limit)
- Max 10 API calls per second
- Exponential backoff on 429 responses

---

## 9. Exit Logic

### 9.1 Profit-Taking Exits

1. **Target reached**: Token price ≥ target_profit_pct above entry
2. **Probability reversal**: P(direction) falls below exit_threshold
3. **Time decay**: Holding into late life with insufficient edge

### 9.2 Stop-Loss Exits

1. **Hard stop**: Token price ≤ stop_loss_pct below entry
2. **Thesis broken**: External price moves against position > threshold
3. **Confidence collapse**: Confidence score drops below minimum

### 9.3 Emergency Exits

1. **Feed failure**: Any required data source disconnects
2. **System stress**: CPU/memory/resource limits breached
3. **Market anomaly**: Price gaps or halts detected

---

## 10. Latency Analysis

### 10.1 Critical Path Latencies

| Component | Target | Acceptable | Critical |
|-----------|--------|------------|----------|
| DNS lookup | <50ms | <200ms | >500ms |
| WS connect | <100ms | <500ms | >1000ms |
| Book sync | <200ms | <1000ms | >3000ms |
| Signal compute | <10ms | <50ms | >100ms |
| Trade decision | <5ms | <20ms | >50ms |
| Order submit | <100ms | <500ms | >1000ms |

### 10.2 Latency Bottlenecks

1. **DNS Resolution**: Mitigated by DoH caching
2. **WebSocket Handshake**: Mitigated by connection pooling
3. **Book Synchronization**: Mitigated by incremental updates
4. **Signal Computation**: Mitigated by async parallelization
5. **Network RTT**: Mitigated by geographic proximity to servers

### 10.3 Latency Mitigation Strategies

- Pre-resolve DNS before connections needed
- Maintain persistent WebSocket connections
- Process signals incrementally, not batched
- Use asyncio.gather() for parallel external fetches
- Co-locate near eu-west-2 (Polymarket primary region)

---

## 11. Failure Scenarios

### 11.1 Network Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| DNS resolution fails | Timeout after 5s | Try next DoH provider |
| WebSocket disconnects | Heartbeat timeout | Exponential backoff reconnect |
| External feed dies | No messages for N seconds | Continue with reduced signals |
| Complete network loss | All connections fail | Graceful shutdown, alert |

### 11.2 Data Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Stale orderbook | Hash unchanged > 5s | Request refresh, verify |
| Corrupted message | Parse exception | Log, skip, continue |
| Sequence gap | Missing sequence numbers | Resync from snapshot |
| RTDS unavailable | Connection refused | Trade without RTDS filter |

### 11.3 System Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Memory exhaustion | OS OOM killer | Restart with smaller buffers |
| CPU saturation | Processing lag | Reduce signal complexity |
| Disk full | Write errors | Rotate logs, alert |
| Clock drift | NTP comparison | Alert, potentially halt |

---

## 12. DNS Failure Handling

### 12.1 DoH Provider Failover

```python
async def resolve_with_fallback(hostname: str) -> Optional[str]:
    for provider in config.doh_providers:
        try:
            ip = await doh_lookup(hostname, provider.url)
            cache.set(hostname, ip, ttl=config.dns_cache_ttl)
            return ip
        except Exception as e:
            logger.warning(f"DoH provider {provider.name} failed: {e}")
            continue
    
    # All providers failed - try cached value even if expired
    cached = cache.get(hostname)
    if cached:
        logger.warning(f"Using expired cache for {hostname}")
        return cached
    
    raise DNSResolutionFailure(f"All DoH providers failed for {hostname}")
```

### 12.2 DNS Cache Management

- TTL-based expiration with jitter
- Background refresh before expiration
- Negative caching for failed lookups
- Per-provider health tracking

---

## 13. Reconnect Recovery

### 13.1 WebSocket Reconnect Strategy

```python
async def reconnect_with_backoff(ws: WebSocket, url: str):
    delay = config.reconnect_base_delay_ms
    attempt = 0
    
    while True:
        try:
            await ws.connect(url)
            await resync_state(ws)  # Restore subscriptions, request snapshot
            logger.info(f"Reconnected after {attempt} attempts")
            return
        except Exception as e:
            attempt += 1
            delay = min(delay * 2, config.reconnect_max_delay_ms)
            jitter = random.uniform(0, config.reconnect_jitter_ms)
            await asyncio.sleep((delay + jitter) / 1000)
```

### 13.2 State Recovery After Reconnect

1. **Orderbook**: Request full snapshot, replay missed updates
2. **Subscriptions**: Re-send subscription messages
3. **Sequence Numbers**: Validate continuity, detect gaps
4. **External Feeds**: Verify prices still within tolerance

---

## 14. Feed Desynchronization Handling

### 14.1 Detection

```python
def check_feed_synchronization() -> SyncStatus:
    timestamps = {
        'orderbook': orderbook.last_update_time,
        'binance': binance_feed.last_price_time,
        'coinbase': coinbase_feed.last_price_time,
        'rtds': rtds_feed.last_update_time,
    }
    
    max_drift = max(timestamps.values()) - min(timestamps.values())
    
    if max_drift > MAX_ALLOWED_DRIFT_MS:
        return SyncStatus.DESYNCHRONIZED
    return SyncStatus.SYNCHRONIZED
```

### 14.2 Recovery

1. **Minor drift (<1s)**: Continue trading, log warning
2. **Moderate drift (1-5s)**: Pause new entries, wait for sync
3. **Severe drift (>5s)**: Halt trading, investigate root cause

---

## 15. Step-by-Step Implementation

See the Python implementation files for complete code:

1. `doh_resolver.py` - DNS-over-HTTPS resolution
2. `websocket_manager.py` - Connection lifecycle management
3. `orderbook.py` - Deterministic orderbook synchronization
4. `signals.py` - Signal computation engine
5. `probability_model.py` - Directional probability scoring
6. `risk_manager.py` - Position sizing and circuit breakers
7. `execution_engine.py` - Order submission and management
8. `main.py` - Application entry point and orchestration

---

## 16. Conclusion

This architecture provides a robust, production-ready foundation for directional trading on Polymarket BTC binary options. Key design principles:

1. **Defense in Depth**: Multiple layers of validation and circuit breakers
2. **Graceful Degradation**: Continue operating with reduced capability during failures
3. **Observability**: Comprehensive logging and metrics for debugging
4. **Modularity**: Clean separation of concerns for maintainability
5. **Configurability**: All parameters exposed for tuning without code changes

The bot is designed to operate continuously with minimal human intervention while maintaining strict risk controls and rapid failure recovery.
