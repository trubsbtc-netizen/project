#!/usr/bin/env python3
"""
Polymarket BTC-UP/DOWN-5M Trading Bot - Main Entry Point

Production-grade directional trading bot for Polymarket binary options.
"""

import asyncio
import signal
import sys
import yaml
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from doh_resolver import DNSOverHTTPSResolver
from websocket_manager import WebSocketManager, ConnectionConfig
from orderbook import OrderbookManager, parse_polymarket_book_message, OrderbookState
from signals import SignalEngine, MarketData


class TradingBot:
    """
    Main trading bot orchestrator.
    
    Coordinates all components:
    - DNS resolution
    - WebSocket connections
    - Orderbook management
    - Signal computation
    - Trade execution
    """
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Components
        self.doh_resolver: Optional[DNSOverHTTPSResolver] = None
        self.ws_manager: Optional[WebSocketManager] = None
        self.orderbook_manager: Optional[OrderbookManager] = None
        self.signal_engine: Optional[SignalEngine] = None
        
        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # External prices
        self._binance_price: Optional[float] = None
        self._coinbase_price: Optional[float] = None
        self._rtds_data: Dict[str, Any] = {}
        
        # Recent trades per market
        self._recent_trades: Dict[str, list] = {}
        
        logger.info("Trading bot initialized")
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configure logging."""
        log_config = self.config.get('logging', {})
        
        log_file = log_config.get('file', 'logs/polymarket_bot.log')
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=log_config.get('max_size_mb', 100) * 1024 * 1024,
                    backupCount=log_config.get('backup_count', 5),
                ),
            ],
        )
        
        from shared.http import suppress_noisy_loggers
        suppress_noisy_loggers()
        
        global logger
        logger = logging.getLogger(__name__)
    
    async def start(self):
        """Start the trading bot."""
        logger.info("Starting trading bot...")
        self._running = True
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Start WebSocket connections
            await self.ws_manager.start_all()
            
            logger.info("Trading bot started successfully")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the trading bot gracefully."""
        logger.info("Stopping trading bot...")
        self._running = False
        
        # Stop components in reverse order
        if self.ws_manager:
            await self.ws_manager.stop_all()
        
        if self.doh_resolver:
            await self.doh_resolver.stop()
        
        logger.info("Trading bot stopped")
    
    async def _initialize_components(self):
        """Initialize all bot components."""
        # DNS Resolver
        network_config = self.config.get('network', {})
        self.doh_resolver = DNSOverHTTPSResolver(
            providers=network_config.get('doh_providers', []),
            cache_ttl_seconds=network_config.get('dns_cache_ttl_seconds', 300),
            timeout_seconds=network_config.get('connection_timeout_seconds', 10),
        )
        await self.doh_resolver.start()
        logger.info("DNS resolver initialized")
        
        # Orderbook Manager
        risk_config = self.config.get('risk', {})
        self.orderbook_manager = OrderbookManager(
            heartbeat_timeout_seconds=risk_config.get('heartbeat_timeout_seconds', 30),
        )
        logger.info("Orderbook manager initialized")
        
        # Signal Engine
        signals_config = self.config.get('signals', {})
        self.signal_engine = SignalEngine(
            weights={
                'orderflow': signals_config.get('orderflow_weight', 0.30),
                'spread': signals_config.get('spread_weight', 0.15),
                'liquidity': signals_config.get('liquidity_weight', 0.20),
                'momentum': signals_config.get('momentum_weight', 0.20),
                'external': signals_config.get('external_price_weight', 0.15),
            },
            momentum_lookback=signals_config.get('momentum_lookback_periods', 5),
            volume_ma_periods=signals_config.get('volume_ma_periods', 20),
            volatility_lookback=signals_config.get('volatility_lookback_periods', 10),
            signal_decay_lambda=signals_config.get('signal_decay_lambda', 0.1),
        )
        logger.info("Signal engine initialized")
        
        # WebSocket Manager
        self.ws_manager = WebSocketManager(self.doh_resolver)
        
        # Add Polymarket orderbook connection
        poly_config = self.config.get('polymarket', {})
        self.ws_manager.add_connection(
            config=ConnectionConfig(
                name='polymarket_orderbook',
                url=poly_config.get('orderbook_ws_url', 'wss://bookfeed.polymarket.com'),
                subscriptions=self._get_polymarket_subscriptions(),
                heartbeat_interval_seconds=30,
                heartbeat_timeout_seconds=risk_config.get('heartbeat_timeout_seconds', 60),
            ),
            on_message=self._handle_polymarket_message,
            on_connected=self._on_polymarket_connected,
            on_disconnected=self._on_polymarket_disconnected,
        )
        
        # Add Binance connection
        ext_config = self.config.get('external_feeds', {})
        if ext_config.get('binance', {}).get('enabled', True):
            self.ws_manager.add_connection(
                config=ConnectionConfig(
                    name='binance',
                    url=ext_config['binance'].get('ws_url', 'wss://stream.binance.com:9443/ws/btcusdt@trade'),
                    subscriptions=[],
                    heartbeat_interval_seconds=30,
                ),
                on_message=self._handle_binance_message,
            )
            logger.info("Binance feed configured")
        
        # Add Coinbase connection
        if ext_config.get('coinbase', {}).get('enabled', True):
            self.ws_manager.add_connection(
                config=ConnectionConfig(
                    name='coinbase',
                    url=ext_config['coinbase'].get('ws_url', 'wss://ws-feed.exchange.coinbase.com'),
                    subscriptions=[{
                        'type': 'subscribe',
                        'product_ids': [ext_config['coinbase'].get('product_id', 'BTC-USD')],
                        'channels': ['ticker'],
                    }],
                    heartbeat_interval_seconds=30,
                ),
                on_message=self._handle_coinbase_message,
            )
            logger.info("Coinbase feed configured")
        
        logger.info("WebSocket manager initialized with all connections")
    
    def _get_polymarket_subscriptions(self) -> list:
        """Get Polymarket subscription messages."""
        poly_config = self.config.get('polymarket', {})
        
        # In production, would fetch active BTC 5M markets from API
        # For now, use placeholder asset IDs
        subscriptions = []
        
        if poly_config.get('subscribe_book_snapshots', True):
            subscriptions.append({
                'type': 'subscribe',
                'channel': 'orderbook',
                'asset_ids': ['btc-up-5m', 'btc-down-5m'],  # Placeholder
            })
        
        if poly_config.get('subscribe_trades', True):
            subscriptions.append({
                'type': 'subscribe',
                'channel': 'trades',
                'asset_ids': ['btc-up-5m', 'btc-down-5m'],
            })
        
        if poly_config.get('subscribe_price_changes', True):
            subscriptions.append({
                'type': 'subscribe',
                'channel': 'price_changes',
                'asset_ids': ['btc-up-5m', 'btc-down-5m'],
            })
        
        return subscriptions
    
    async def _handle_polymarket_message(self, message: Dict[str, Any]):
        """Handle incoming Polymarket message."""
        try:
            market_id, msg_type, data = parse_polymarket_book_message(message)
            
            if not market_id:
                return
            
            if msg_type == 'snapshot':
                self.orderbook_manager.process_snapshot(market_id, data)
                logger.debug(f"Processed snapshot for {market_id}")
                
            elif msg_type == 'update':
                self.orderbook_manager.process_update(market_id, data)
                
            elif msg_type == 'trade':
                # Track recent trades for orderflow signal
                if market_id not in self._recent_trades:
                    self._recent_trades[market_id] = []
                self._recent_trades[market_id].append(data)
                # Keep only last 100 trades
                self._recent_trades[market_id] = self._recent_trades[market_id][-100:]
            
            # Check if we should evaluate signals
            if self.orderbook_manager.get_book(market_id):
                await self._evaluate_signals(market_id)
                
        except Exception as e:
            logger.warning(f"Error processing Polymarket message: {e}")
    
    async def _handle_binance_message(self, message: Dict[str, Any]):
        """Handle incoming Binance message."""
        try:
            # Parse trade message
            if 'p' in message:  # Price field
                self._binance_price = float(message['p'])
        except Exception as e:
            logger.warning(f"Error processing Binance message: {e}")
    
    async def _handle_coinbase_message(self, message: Dict[str, Any]):
        """Handle incoming Coinbase message."""
        try:
            if message.get('type') == 'ticker' and 'price' in message:
                self._coinbase_price = float(message['price'])
        except Exception as e:
            logger.warning(f"Error processing Coinbase message: {e}")
    
    async def _on_polymarket_connected(self):
        """Called when Polymarket connection is established."""
        logger.info("Polymarket connection established")
    
    async def _on_polymarket_disconnected(self):
        """Called when Polymarket connection is lost."""
        logger.warning("Polymarket connection lost")
    
    async def _evaluate_signals(self, market_id: str):
        """Evaluate trading signals for a market."""
        if not self._running:
            return
        
        book = self.orderbook_manager.get_book(market_id)
        
        if not book or not book.is_valid():
            return
        
        # Check if enough time remains before expiry
        rtds_time = self._rtds_data.get(market_id, {}).get('time_to_expiry', 300)
        trading_config = self.config.get('trading', {})
        
        if rtds_time < trading_config.get('min_time_to_expiry_seconds', 30):
            return  # Too close to expiry
        
        # Build market data for signal engine
        from decimal import Decimal
        
        bid_levels = [(l.price, l.size) for l in book.bids.levels.values()]
        ask_levels = [(l.price, l.size) for l in book.asks.levels.values()]
        
        market_data = MarketData(
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            mid_price=book.mid_price,
            spread_pct=book.spread_pct or 0.0,
            bid_levels=bid_levels[:20],  # Top 20 levels
            ask_levels=ask_levels[:20],
            recent_trades=self._recent_trades.get(market_id, []),
            binance_price=Decimal(str(self._binance_price)) if self._binance_price else None,
            coinbase_price=Decimal(str(self._coinbase_price)) if self._coinbase_price else None,
            rtds_reference_price=Decimal(str(self._rtds_data.get(market_id, {}).get('reference_price', 0))) or None,
            rtds_time_to_expiry=rtds_time,
            last_update_time=time.time(),
        )
        
        # Compute signals
        signals = self.signal_engine.compute_all_signals(market_id, market_data)
        probability, confidence = self.signal_engine.compute_weighted_probability(signals)
        
        # Log signal summary
        logger.debug(
            f"Signals for {market_id}: P(UP)={probability:.3f}, confidence={confidence:.3f}"
        )
        
        # Check entry conditions
        if self._should_enter_trade(probability, confidence, book):
            direction = 'UP' if probability > 0.5 else 'DOWN'
            logger.info(
                f"ENTRY SIGNAL: {direction} on {market_id} "
                f"(P={probability:.3f}, conf={confidence:.3f})"
            )
            # In production: execute trade here
            # await self._execute_trade(market_id, direction, probability, confidence)
    
    def _should_enter_trade(
        self,
        probability: float,
        confidence: float,
        book: OrderbookState,
    ) -> bool:
        """Check if entry conditions are met."""
        trading_config = self.config.get('trading', {})
        risk_config = self.config.get('risk', {})
        
        # Probability threshold
        entry_threshold = trading_config.get('entry_threshold', 0.60)
        if probability < entry_threshold and probability > (1 - entry_threshold + 0.5):
            return False
        
        # Confidence threshold
        conf_threshold = trading_config.get('confidence_threshold', 0.50)
        if confidence < conf_threshold:
            return False
        
        # Spread check
        max_spread = risk_config.get('halt_on_spread_pct', 5.0)
        if book.spread_pct and book.spread_pct > max_spread:
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        return {
            'running': self._running,
            'connections': self.ws_manager.get_all_metrics() if self.ws_manager else {},
            'orderbooks': self.orderbook_manager.get_metrics() if self.orderbook_manager else {},
            'signals': self.signal_engine.get_metrics() if self.signal_engine else {},
            'dns': self.doh_resolver.get_metrics() if self.doh_resolver else {},
            'external_prices': {
                'binance': self._binance_price,
                'coinbase': self._coinbase_price,
            },
        }


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Polymarket BTC Trading Bot')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Create bot
    bot = TradingBot(args.config)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Shutdown signal received")
        bot._shutdown_event.set()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
    
    # Run bot
    try:
        await bot.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Bot crashed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
