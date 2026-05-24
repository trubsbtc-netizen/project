# BTC Poly Institutional Directional Engine

Websocket-first Polymarket BTC Up/Down 5-minute trading system.

Runtime hot path:

```text
websocket -> parsing -> normalization -> canonical state -> microstructure -> probability -> risk -> execution
```

Control-plane HTTP is disabled by default. PTB metadata is bootstrapped once at startup with short bounded JSON requests, then the runtime path stays websocket/cache-first. For zero cold HTTP metadata, use `PTB_MODE=static` and provide the current round condition/token/PTB values.

Optional startup tuning:

```powershell
PTB_STARTUP_BOOTSTRAP_TIMEOUT_S=3
```

Run:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Validation:

```powershell
python -B -m pytest -q
```
