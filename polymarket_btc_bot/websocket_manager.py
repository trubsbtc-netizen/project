"""
WebSocket Manager for Polymarket Trading Bot

Handles connection lifecycle, reconnection, heartbeat monitoring, and message routing
for multiple WebSocket feeds (Polymarket, Binance, Coinbase, RTDS).
"""

import asyncio
import time
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Awaitable
from enum import Enum
import aiohttp
import logging

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


@dataclass
class ConnectionConfig:
    """Configuration for a WebSocket connection."""
    name: str
    url: str
    subscriptions: List[Dict[str, Any]] = field(default_factory=list)
    heartbeat_interval_seconds: float = 30.0
    heartbeat_timeout_seconds: float = 60.0
    reconnect_base_delay_ms: float = 100.0
    reconnect_max_delay_ms: float = 30000.0
    reconnect_jitter_ms: float = 100.0
    max_message_size_bytes: int = 10 * 1024 * 1024  # 10MB


@dataclass
class ConnectionMetrics:
    """Metrics for a WebSocket connection."""
    connect_count: int = 0
    disconnect_count: int = 0
    message_count: int = 0
    error_count: int = 0
    reconnect_count: int = 0
    last_message_time: Optional[float] = None
    last_heartbeat_time: Optional[float] = None
    avg_latency_ms: float = 0.0


class WebSocketConnection:
    """
    Production-grade WebSocket connection with:
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring
    - Message routing
    - State recovery
    - Metrics collection
    """
    
    def __init__(
        self,
        config: ConnectionConfig,
        doh_resolver: Any,  # DNSOverHTTPSResolver
        on_message: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_connected: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnected: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.config = config
        self.doh_resolver = doh_resolver
        self.on_message = on_message
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        
        self._state = ConnectionState.DISCONNECTED
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        
        self._metrics = ConnectionMetrics()
        
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        # Subscription state for recovery
        self._subscriptions_active: List[Dict[str, Any]] = []
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED
    
    @property
    def metrics(self) -> ConnectionMetrics:
        return self._metrics
    
    async def start(self):
        """Start the connection and background tasks."""
        self._running = True
        self._shutdown_event.clear()
        
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
        )
        
        # Start connection loop
        asyncio.create_task(self._connection_loop())
    
    async def stop(self):
        """Stop the connection and cleanup resources."""
        self._running = False
        self._shutdown_event.set()
        
        # Cancel tasks
        for task in [self._receive_task, self._heartbeat_task, self._reconnect_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close WebSocket
        if self._ws:
            await self._ws.close()
        
        # Close session
        if self._session:
            await self._session.close()
        
        self._state = ConnectionState.DISCONNECTED
    
    async def _connection_loop(self):
        """Main connection loop with reconnection logic."""
        while self._running:
            try:
                await self._connect()
                
                if self._running:
                    # Wait until disconnected or shutdown
                    await self._shutdown_event.wait()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection loop error for {self.config.name}: {e}")
                self._metrics.error_count += 1
                
                if self._running:
                    await self._schedule_reconnect()
    
    async def _connect(self):
        """Establish WebSocket connection."""
        self._state = ConnectionState.CONNECTING
        self._metrics.connect_count += 1
        
        try:
            # Resolve hostname via DoH
            from urllib.parse import urlparse
            parsed = urlparse(self.config.url)
            hostname = parsed.hostname
            
            if hostname:
                ips = await self.doh_resolver.resolve(hostname)
                # Use resolved IP directly
                logger.debug(f"Resolved {hostname} to {ips[0]}")
            
            # Connect
            self._ws = await self._session.ws_connect(
                self.config.url,
                max_msg_size=self.config.max_message_size_bytes,
                heartbeat=self.config.heartbeat_interval_seconds,
            )
            
            self._state = ConnectionState.CONNECTED
            self._metrics.disconnect_count = 0  # Reset on successful connect
            logger.info(f"Connected to {self.config.name}")
            
            # Send subscriptions
            await self._send_subscriptions()
            
            # Notify connected
            if self.on_connected:
                await self.on_connected()
            
            # Start receive and heartbeat tasks
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Wait for disconnection
            await self._ws.wait_closed()
            
        except Exception as e:
            logger.warning(f"Connection failed for {self.config.name}: {e}")
            self._metrics.error_count += 1
            self._state = ConnectionState.FAILED
            
            if self._running:
                await self._schedule_reconnect()
        
        finally:
            self._state = ConnectionState.DISCONNECTED
            self._metrics.disconnect_count += 1
            
            # Notify disconnected
            if self.on_disconnected:
                await self.on_disconnected()
            
            self._shutdown_event.clear()
    
    async def _receive_loop(self):
        """Receive and route messages."""
        while self._running and self._ws and not self._ws.closed:
            try:
                msg = await self._ws.receive(timeout=self.config.heartbeat_timeout_seconds)
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    self._metrics.message_count += 1
                    self._metrics.last_message_time = time.time()
                    
                    try:
                        data = json.loads(msg.data)
                        
                        if self.on_message:
                            await self.on_message(data)
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON from {self.config.name}: {e}")
                        
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Handle binary messages if needed
                    pass
                    
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    logger.warning(f"WebSocket closed/error for {self.config.name}")
                    break
                    
            except asyncio.TimeoutError:
                # No message received within timeout - check heartbeat
                if self._metrics.last_heartbeat_time:
                    time_since_heartbeat = time.time() - self._metrics.last_heartbeat_time
                    if time_since_heartbeat > self.config.heartbeat_timeout_seconds:
                        logger.warning(f"Heartbeat timeout for {self.config.name}")
                        break
                        
            except Exception as e:
                logger.error(f"Receive error for {self.config.name}: {e}")
                self._metrics.error_count += 1
                break
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running and self._ws and not self._ws.closed:
            try:
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
                
                # Send ping
                if self._ws:
                    await self._ws.ping()
                    self._metrics.last_heartbeat_time = time.time()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat failed for {self.config.name}: {e}")
    
    async def _send_subscriptions(self):
        """Send subscription messages after connecting."""
        if not self._ws or self._ws.closed:
            return
        
        for subscription in self.config.subscriptions:
            try:
                await self._ws.send_json(subscription)
                self._subscriptions_active.append(subscription)
                logger.debug(f"Sent subscription to {self.config.name}")
            except Exception as e:
                logger.warning(f"Failed to send subscription: {e}")
    
    async def _schedule_reconnect(self):
        """Schedule reconnection with exponential backoff."""
        if not self._running or self._reconnect_task:
            return
        
        self._state = ConnectionState.RECONNECTING
        self._metrics.reconnect_count += 1
        
        # Exponential backoff with jitter
        base_delay = self.config.reconnect_base_delay_ms / 1000.0
        max_delay = self.config.reconnect_max_delay_ms / 1000.0
        jitter = random.uniform(0, self.config.reconnect_jitter_ms / 1000.0)
        
        delay = min(base_delay * (2 ** min(self._metrics.reconnect_count, 10)), max_delay) + jitter
        
        logger.info(f"Scheduling reconnect for {self.config.name} in {delay:.2f}s")
        
        self._reconnect_task = asyncio.create_task(self._reconnect_after(delay))
    
    async def _reconnect_after(self, delay: float):
        """Wait and then trigger reconnection."""
        try:
            await asyncio.sleep(delay)
            self._reconnect_task = None
            
            if self._running:
                logger.info(f"Reconnecting {self.config.name}")
                self._state = ConnectionState.CONNECTING
                await self._connect()
                
        except asyncio.CancelledError:
            pass
    
    def send_message(self, data: Dict[str, Any]):
        """Send a message through the WebSocket."""
        if self._ws and not self._ws.closed:
            asyncio.create_task(self._ws.send_json(data))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return connection metrics."""
        return {
            'name': self.config.name,
            'state': self._state.value,
            'is_connected': self.is_connected,
            'connect_count': self._metrics.connect_count,
            'disconnect_count': self._metrics.disconnect_count,
            'message_count': self._metrics.message_count,
            'error_count': self._metrics.error_count,
            'reconnect_count': self._metrics.reconnect_count,
            'last_message_time': self._metrics.last_message_time,
            'last_heartbeat_time': self._metrics.last_heartbeat_time,
            'avg_latency_ms': self._metrics.avg_latency_ms,
        }


class WebSocketManager:
    """
    Manages multiple WebSocket connections with unified interface.
    """
    
    def __init__(self, doh_resolver: Any):
        self.doh_resolver = doh_resolver
        self._connections: Dict[str, WebSocketConnection] = {}
        self._running = False
    
    def add_connection(
        self,
        config: ConnectionConfig,
        on_message: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_connected: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnected: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """Add a WebSocket connection."""
        conn = WebSocketConnection(
            config=config,
            doh_resolver=self.doh_resolver,
            on_message=on_message,
            on_connected=on_connected,
            on_disconnected=on_disconnected,
        )
        self._connections[config.name] = conn
    
    async def start_all(self):
        """Start all connections."""
        self._running = True
        
        for conn in self._connections.values():
            await conn.start()
    
    async def stop_all(self):
        """Stop all connections."""
        self._running = False
        
        for conn in self._connections.values():
            await conn.stop()
    
    def get_connection(self, name: str) -> Optional[WebSocketConnection]:
        """Get a connection by name."""
        return self._connections.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all connections."""
        return {
            name: conn.get_metrics()
            for name, conn in self._connections.items()
        }
    
    def are_all_connected(self) -> bool:
        """Check if all connections are active."""
        return all(conn.is_connected for conn in self._connections.values())
    
    def get_critical_connections(self) -> List[str]:
        """Get names of disconnected critical connections."""
        return [
            name for name, conn in self._connections.items()
            if not conn.is_connected
        ]
