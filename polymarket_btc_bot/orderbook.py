"""
Orderbook Manager for Polymarket Trading Bot

Deterministic orderbook synchronization with:
- Snapshot + incremental update model
- Sequence number validation
- Gap detection and recovery
- Efficient level storage
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from collections import defaultdict
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


@dataclass
class PriceLevel:
    """A single price level in the orderbook."""
    price: Decimal
    size: Decimal
    order_count: int = 1
    last_update_time: float = 0.0
    
    def __hash__(self):
        return hash((self.price, self.size))


@dataclass
class OrderbookSide:
    """One side of the orderbook (bids or asks)."""
    levels: Dict[Decimal, PriceLevel] = field(default_factory=dict)
    
    def best(self) -> Optional[PriceLevel]:
        """Get the best price level."""
        if not self.levels:
            return None
        return max(self.levels.values(), key=lambda l: l.price)
    
    def worst(self) -> Optional[PriceLevel]:
        """Get the worst price level."""
        if not self.levels:
            return None
        return min(self.levels.values(), key=lambda l: l.price)
    
    def get_level(self, price: Decimal) -> Optional[PriceLevel]:
        """Get a specific price level."""
        return self.levels.get(price)
    
    def add_or_update(self, price: Decimal, size: Decimal, order_count: int = 1):
        """Add or update a price level."""
        now = time.time()
        
        if size <= 0:
            # Remove level if size is zero or negative
            self.levels.pop(price, None)
        else:
            self.levels[price] = PriceLevel(
                price=price,
                size=size,
                order_count=order_count,
                last_update_time=now,
            )
    
    def depth_at_price(self, target_price: Decimal, side: str = 'bids') -> Decimal:
        """Calculate total depth up to target price."""
        total = Decimal('0')
        
        for level in self.levels.values():
            if side == 'bids' and level.price >= target_price:
                total += level.size
            elif side == 'asks' and level.price <= target_price:
                total += level.size
        
        return total
    
    def depth_within_ticks(self, ticks: int, tick_size: Decimal, mid_price: Decimal) -> Decimal:
        """Calculate total depth within N ticks of mid price."""
        total = Decimal('0')
        threshold = tick_size * ticks
        
        for level in self.levels.values():
            if abs(level.price - mid_price) <= threshold:
                total += level.size
        
        return total
    
    def clear(self):
        """Clear all levels."""
        self.levels.clear()
    
    def snapshot(self) -> List[Dict[str, Any]]:
        """Create a serializable snapshot."""
        sorted_levels = sorted(
            self.levels.values(),
            key=lambda l: l.price,
            reverse=True  # Highest first for bids
        )
        return [
            {'price': str(l.price), 'size': str(l.size), 'order_count': l.order_count}
            for l in sorted_levels
        ]


@dataclass
class OrderbookState:
    """Complete orderbook state."""
    market_id: str
    bids: OrderbookSide = field(default_factory=OrderbookSide)
    asks: OrderbookSide = field(default_factory=OrderbookSide)
    sequence_number: int = 0
    last_update_time: float = 0.0
    tick_size: Decimal = Decimal('0.01')
    is_synchronized: bool = False
    
    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get best bid price."""
        best = self.bids.best()
        return best.price if best else None
    
    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get best ask price."""
        best = self.asks.best()
        return best.price if best else None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """Get mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Get spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        """Get spread as percentage of mid."""
        if self.spread and self.mid_price and self.mid_price > 0:
            return float(self.spread / self.mid_price * 100)
        return None
    
    @property
    def total_bid_volume(self) -> Decimal:
        """Get total bid volume."""
        return sum(level.size for level in self.bids.levels.values())
    
    @property
    def total_ask_volume(self) -> Decimal:
        """Get total ask volume."""
        return sum(level.size for level in self.asks.levels.values())
    
    def is_valid(self) -> bool:
        """Check if orderbook is valid for trading."""
        return (
            self.is_synchronized and
            self.best_bid is not None and
            self.best_ask is not None and
            self.best_bid < self.best_ask and  # No crossed book
            self.total_bid_volume > 0 and
            self.total_ask_volume > 0
        )
    
    def time_since_update(self) -> float:
        """Get seconds since last update."""
        return time.time() - self.last_update_time


class OrderbookManager:
    """
    Manages orderbook state for multiple markets with:
    - Deterministic synchronization
    - Sequence gap detection
    - Automatic recovery
    - Efficient updates
    """
    
    def __init__(
        self,
        heartbeat_timeout_seconds: float = 30.0,
    ):
        self._books: Dict[str, OrderbookState] = {}
        self._heartbeat_timeout_seconds = heartbeat_timeout_seconds
        
        # Pending updates waiting for sequence
        self._pending_updates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Expected sequence numbers per market
        self._expected_sequences: Dict[str, int] = {}
        
        # Metrics
        self._update_count = 0
        self._gap_count = 0
        self._recovery_count = 0
    
    def get_book(self, market_id: str) -> Optional[OrderbookState]:
        """Get orderbook for a market."""
        return self._books.get(market_id)
    
    def create_or_get_book(self, market_id: str) -> OrderbookState:
        """Get existing book or create new one."""
        if market_id not in self._books:
            self._books[market_id] = OrderbookState(market_id=market_id)
        return self._books[market_id]
    
    def process_snapshot(self, market_id: str, snapshot: Dict[str, Any]) -> OrderbookState:
        """
        Process a full orderbook snapshot.
        
        Resets the book to the snapshot state.
        """
        book = self.create_or_get_book(market_id)
        
        # Clear existing state
        book.bids.clear()
        book.asks.clear()
        
        # Process bids
        for bid in snapshot.get('bids', []):
            price = Decimal(str(bid['price']))
            size = Decimal(str(bid['size']))
            order_count = bid.get('order_count', 1)
            book.bids.add_or_update(price, size, order_count)
        
        # Process asks
        for ask in snapshot.get('asks', []):
            price = Decimal(str(ask['price']))
            size = Decimal(str(ask['size']))
            order_count = bid.get('order_count', 1)
            book.asks.add_or_update(price, size, order_count)
        
        # Update metadata
        book.sequence_number = snapshot.get('sequence', 0)
        book.last_update_time = time.time()
        book.is_synchronized = True
        
        # Set expected next sequence
        self._expected_sequences[market_id] = book.sequence_number + 1
        
        # Clear pending updates
        self._pending_updates[market_id].clear()
        
        self._update_count += 1
        
        logger.debug(f"Snapshot processed for {market_id}: seq={book.sequence_number}")
        
        return book
    
    def process_update(self, market_id: str, update: Dict[str, Any]) -> Optional[OrderbookState]:
        """
        Process an incremental orderbook update.
        
        Handles out-of-order updates by buffering.
        """
        book = self.get_book(market_id)
        
        if not book or not book.is_synchronized:
            # Can't apply updates without a snapshot first
            logger.warning(f"Received update for unsynchronized book: {market_id}")
            return None
        
        update_seq = update.get('sequence', 0)
        expected_seq = self._expected_sequences.get(market_id, 0)
        
        # Check sequence
        if update_seq < expected_seq:
            # Old update - ignore
            logger.debug(f"Ignoring old update for {market_id}: seq={update_seq}, expected={expected_seq}")
            return book
        
        if update_seq > expected_seq:
            # Gap detected - buffer update
            logger.warning(f"Sequence gap for {market_id}: received={update_seq}, expected={expected_seq}")
            self._gap_count += 1
            
            # Buffer the update
            self._pending_updates[market_id].append(update)
            
            # Request resync if gap is too large
            if update_seq - expected_seq > 100:
                logger.error(f"Large sequence gap for {market_id}, requesting resync")
                self._request_resync(market_id)
            
            return book
        
        # Apply the update
        self._apply_update(book, update)
        
        # Update expected sequence
        self._expected_sequences[market_id] = update_seq + 1
        
        # Process any buffered updates
        self._process_buffered_updates(market_id)
        
        self._update_count += 1
        
        return book
    
    def _apply_update(self, book: OrderbookState, update: Dict[str, Any]):
        """Apply a single update to the book."""
        # Handle price changes
        for change in update.get('price_changes', []):
            side = change.get('side', '').upper()
            price = Decimal(str(change['price']))
            size = Decimal(str(change['size']))
            order_count = change.get('order_count', 1)
            
            if side == 'BUY' or side == 'BID':
                book.bids.add_or_update(price, size, order_count)
            elif side == 'SELL' or side == 'ASK':
                book.asks.add_or_update(price, size, order_count)
        
        # Handle trades
        for trade in update.get('trades', []):
            # Trades don't modify the book directly but we track them
            pass
        
        # Handle tick size changes
        if 'tick_size' in update:
            book.tick_size = Decimal(str(update['tick_size']))
            logger.info(f"Tick size changed for {book.market_id}: {book.tick_size}")
        
        book.last_update_time = time.time()
    
    def _process_buffered_updates(self, market_id: str):
        """Process buffered updates in sequence order."""
        buffered = self._pending_updates[market_id]
        expected_seq = self._expected_sequences.get(market_id, 0)
        
        # Sort by sequence
        buffered.sort(key=lambda u: u.get('sequence', 0))
        
        # Process contiguous updates
        while buffered:
            next_update = buffered[0]
            next_seq = next_update.get('sequence', 0)
            
            if next_seq == expected_seq:
                buffered.pop(0)
                book = self.get_book(market_id)
                if book:
                    self._apply_update(book, next_update)
                    self._expected_sequences[market_id] = next_seq + 1
                    self._update_count += 1
            else:
                break
    
    def _request_resync(self, market_id: str):
        """Request a full resync of the orderbook."""
        self._recovery_count += 1
        # In production, this would trigger a snapshot request
        # For now, just mark the book as needing resync
        book = self.get_book(market_id)
        if book:
            book.is_synchronized = False
    
    def check_health(self) -> Dict[str, Any]:
        """Check health of all orderbooks."""
        now = time.time()
        unhealthy = []
        
        for market_id, book in self._books.items():
            if not book.is_synchronized:
                unhealthy.append({
                    'market_id': market_id,
                    'reason': 'not_synchronized',
                })
            elif book.time_since_update() > self._heartbeat_timeout_seconds:
                unhealthy.append({
                    'market_id': market_id,
                    'reason': 'stale',
                    'seconds_since_update': book.time_since_update(),
                })
        
        return {
            'total_books': len(self._books),
            'synchronized_books': sum(1 for b in self._books.values() if b.is_synchronized),
            'unhealthy_books': unhealthy,
            'total_updates': self._update_count,
            'gaps_detected': self._gap_count,
            'recoveries': self._recovery_count,
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orderbook metrics."""
        return {
            'total_books': len(self._books),
            'update_count': self._update_count,
            'gap_count': self._gap_count,
            'recovery_count': self._recovery_count,
            'health': self.check_health(),
        }


def parse_polymarket_book_message(message: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Parse a Polymarket orderbook message.
    
    Returns: (market_id, message_type, data)
    """
    event_type = message.get('event_type', '')
    
    if event_type == 'book':
        # Full snapshot
        market_id = message.get('asset_id', message.get('market_id', ''))
        return market_id, 'snapshot', message
    
    elif event_type == 'price_change':
        # Incremental update
        market_id = message.get('asset_id', message.get('market_id', ''))
        return market_id, 'update', message
    
    elif event_type == 'last_trade_price':
        # Trade notification
        market_id = message.get('asset_id', message.get('market_id', ''))
        return market_id, 'trade', message
    
    elif event_type == 'tick_size_change':
        # Tick size update
        market_id = message.get('asset_id', message.get('market_id', ''))
        return market_id, 'tick_change', message
    
    return '', 'unknown', message
