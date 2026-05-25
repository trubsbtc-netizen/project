"""
System-wide constants for Polymarket BTC UP/DOWN 5M bot.
All values are derived from official Polymarket documentation.
"""

# ─────────────────────────── API Endpoints ───────────────────────────

CLOB_HOST          = "https://clob.polymarket.com"
GAMMA_API_HOST     = "https://gamma-api.polymarket.com"
DATA_API_HOST      = "https://data-api.polymarket.com"
POLYGON_RPC_URL    = "https://api.zan.top/node/v1/polygon/mainnet/c0100d1f93234184bf1aaeb86f7f4537"

WS_MARKET_ENDPOINT = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_USER_ENDPOINT   = "wss://ws-subscriptions-clob.polymarket.com/ws/user"

# Polygon chain ID for Polymarket
POLYGON_CHAIN_ID = 137
CTF_CONTRACT_ADDRESS = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

# ─────────────────────────── WebSocket ───────────────────────────

WS_PING_INTERVAL_S    = 10          # Polymarket market/user WS heartbeat
WS_PONG_TIMEOUT_S     = 10          # Expect pong within 10s
WS_DEAD_TIMEOUT_S     = 45          # Mark dead if no message in 45s
WS_MAX_MESSAGE_SIZE   = 4 * 1024 * 1024  # 4MB max message
WS_QUEUE_MAXSIZE      = 10_000      # Backpressure: max queued messages

# Reconnect strategy: jittered exponential backoff
WS_RECONNECT_BASE_S   = 0.5        # Initial reconnect delay
WS_RECONNECT_MAX_S    = 30.0       # Max reconnect delay
WS_RECONNECT_FACTOR   = 2.0        # Exponential factor
WS_RECONNECT_JITTER   = 0.3        # 30% jitter applied to delay
WS_MAX_RECONNECT_ATTEMPTS = 0      # 0 = infinite

# ─────────────────────────── DNS over HTTPS ───────────────────────────

DOH_PROVIDERS = [
    "https://1.1.1.1/dns-query",          # Cloudflare primary
    "https://1.0.0.1/dns-query",          # Cloudflare secondary
    "https://8.8.8.8/resolve",            # Google primary
    "https://8.8.4.4/resolve",            # Google secondary
    "https://9.9.9.9/dns-query",          # Quad9
]
DOH_TIMEOUT_S     = 3.0
DOH_CACHE_TTL_S   = 300            # Cache DNS for 5 minutes

# Targets to resolve ahead of time
DNS_PREFETCH_HOSTS = [
    "clob.polymarket.com",
    "ws-subscriptions-clob.polymarket.com",
    "gamma-api.polymarket.com",
    "data-api.polymarket.com",
]

# ─────────────────────────── Heartbeat ───────────────────────────

HEARTBEAT_INTERVAL_S = 5           # Send every 5s (10s timeout, 5s buffer)
HEARTBEAT_ENDPOINT   = f"{CLOB_HOST}/heartbeat"

# ─────────────────────────── Market ───────────────────────────

# BTC UP/DOWN 5M market cycle duration
MARKET_CYCLE_S         = 300        # 5 minutes
MARKET_NEAR_EXPIRY_S   = 60         # Don't enter new positions < 60s to expiry
MARKET_COOLDOWN_S      = 3          # Wait N seconds after resolution before entering new market
MARKET_SEARCH_SLUG     = "btc-up-or-down"   # Gamma API search slug pattern
MARKET_SEARCH_TAG      = "BTC"

# GTD order expiry buffer (from Polymarket docs: must add 60s security threshold)
GTD_SECURITY_BUFFER_S  = 60

# ─────────────────────────── Fees ───────────────────────────

# Crypto markets taker fee: fee = 0.07 × C × p × (1-p)
CRYPTO_TAKER_FEE_RATE  = 0.07
# Fee exponent = 2 (quadratic: C × r × p × (1-p))
CRYPTO_FEE_EXPONENT    = 2
# Makers are never charged
MAKER_FEE_RATE         = 0.0

# ─────────────────────────── Order Constraints ───────────────────────────

MIN_ORDER_SIZE_USDC     = 5.0       # Minimum $5 per order (Polymarket minimum)
MIN_ORDER_SIZE_SHARES   = 1.0       # Minimum 1 share
MAX_PRICE_SLIPPAGE      = 0.03      # Max 3 cents slippage on market orders
DEFAULT_TICK_SIZE       = "0.01"    # Default tick size for BTC UP/DOWN markets

# ─────────────────────────── Signal Thresholds ───────────────────────────

# Minimum net EV (after fees) required to enter
MIN_EV_THRESHOLD        = 0.025     # 2.5% edge minimum

# Minimum confidence required for entry
MIN_CONFIDENCE          = 0.65      # 65% confidence minimum

# OFI (Order Flow Imbalance) threshold to register as signal
OFI_SIGNAL_THRESHOLD    = 0.25      # 25% imbalance

# Depth ratio threshold: how imbalanced must books be
DEPTH_RATIO_THRESHOLD   = 0.20      # 20% depth asymmetry

# Spread maximum to consider market liquid
MAX_TRADEABLE_SPREAD    = 0.06      # 6 cents max spread

# Microprice deviation from 0.5 to be considered signal
MICROPRICE_SIGNAL_DEVIATION = 0.04  # 4 cents from 0.5

# Minimum total depth (both sides) in UP book to consider liquid
MIN_BOOK_DEPTH_USDC     = 200.0

# Consecutive sweeps needed to confirm aggressive flow
SWEEP_CONFIRM_COUNT     = 2

# Velocity: min price change rate (per second) for velocity signal
MIN_PRICE_VELOCITY      = 0.002     # 0.2 cents/sec

# Spoof detection: order that appears and disappears within N ms
SPOOF_DISAPPEAR_MS      = 2000      # 2 seconds

# Large order threshold (as fraction of total depth) for spoof detection
SPOOF_SIZE_THRESHOLD    = 0.15      # Order > 15% of book depth is suspicious

# Absorption detection window
ABSORPTION_WINDOW_S     = 10.0

# ─────────────────────────── Risk Limits ───────────────────────────

MAX_POSITION_USDC       = 50.0      # Max $50 per single position
MAX_TOTAL_EXPOSURE_USDC = 150.0     # Max $150 total across all open positions
MAX_DAILY_LOSS_USDC     = 75.0      # Daily stop-loss
MAX_CONSECUTIVE_LOSSES  = 4         # Circuit breaker after 4 consecutive losses
MAX_DRAWDOWN_PCT        = 0.20      # 20% drawdown from peak triggers circuit breaker
MAX_TRADES_PER_MINUTE   = 3         # Frequency limiter
MAX_TRADES_PER_CYCLE    = 1         # Only 1 trade per 5-minute cycle
CIRCUIT_BREAKER_COOLDOWN_S = 300    # 5 minutes cooldown after circuit breaker trips
VOLATILITY_PAUSE_S      = 60        # Pause after detecting abnormal volatility

# ─────────────────────────── Logging ───────────────────────────

LOG_LEVEL               = "INFO"
LOG_FILE_PATH           = "logs/bot.jsonl"
LOG_ROTATION_MB         = 100
LOG_RETENTION_DAYS      = 7

# ─────────────────────────── Metrics ───────────────────────────

METRICS_PORT            = 9090      # Prometheus metrics HTTP port
METRICS_INTERVAL_S      = 10        # Emit metrics snapshot every 10s

# ─────────────────────────── State Recovery ───────────────────────────

STATE_FILE_PATH         = "state/bot_state.json"
STATE_SAVE_INTERVAL_S   = 5         # Save state every 5s
