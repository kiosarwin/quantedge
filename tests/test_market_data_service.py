import asyncio

from src.data.market_data import MarketDataService, MarketSnapshot


def _service(concurrency=2, timeout=0.05):
    svc = MarketDataService.__new__(MarketDataService)
    svc._snapshot_concurrency = concurrency
    svc._snapshot_timeout_s = timeout

    async def no_context():
        from src.models.market_context import unavailable_context

        return unavailable_context()

    svc.fetch_market_context = no_context
    return svc


def test_fetch_snapshots_limits_symbol_concurrency():
    svc = _service(concurrency=2, timeout=1.0)
    active = 0
    max_active = 0

    async def fake_fetch(symbol):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1
        return MarketSnapshot(symbol=symbol)

    svc.fetch_snapshot = fake_fetch

    out = asyncio.run(svc.fetch_snapshots(["A", "B", "C", "D", "E"]))

    assert set(out) == {"A", "B", "C", "D", "E"}
    assert max_active == 2


def test_fetch_snapshots_drops_slow_symbol_without_failing_batch():
    svc = _service(concurrency=2, timeout=0.03)

    async def fake_fetch(symbol):
        if symbol == "SLOW":
            await asyncio.sleep(1.0)
        else:
            await asyncio.sleep(0.01)
        return MarketSnapshot(symbol=symbol)

    svc.fetch_snapshot = fake_fetch

    out = asyncio.run(svc.fetch_snapshots(["FAST1", "SLOW", "FAST2"]))

    assert set(out) == {"FAST1", "FAST2"}


def _raw_ohlcv(values):
    base = 1_700_000_000_000
    return [[base + i * 3_600_000, v, v, v, v, 100.0] for i, v in enumerate(values)]


def test_fetch_market_context_caches_and_attaches_to_snapshots():
    class Client:
        def __init__(self):
            self.calls = []

        async def fetch_ohlcv(self, symbol, timeframe, limit=300):
            self.calls.append((symbol, timeframe, limit))
            if symbol.startswith("BTC"):
                return _raw_ohlcv([100, 103, 106])
            return _raw_ohlcv([10, 10.8, 11.8])

    cfg = {
        "timeframes": {"primary": "1h", "higher": "4h", "lower": "15m", "entry": "5m"},
        "order_book": {"depth_levels": 10},
        "trading": {"candle_refresh_seconds": 30},
        "safety": {"market_data_snapshot_concurrency": 2, "market_data_snapshot_timeout_seconds": 1},
        "smart_money": {"oi_baseline_seconds": 3600},
        "market_context": {"enabled": True, "ttl_seconds": 300, "lookback_bars": 3},
    }
    svc = MarketDataService(Client(), cfg)

    async def fake_fetch(symbol):
        return MarketSnapshot(symbol=symbol)

    svc.fetch_snapshot = fake_fetch

    out1 = asyncio.run(svc.fetch_snapshots(["ALT/USDT:USDT"]))
    out2 = asyncio.run(svc.fetch_snapshots(["ALT2/USDT:USDT"]))

    assert out1["ALT/USDT:USDT"].market_context.risk_on_state == "risk_on_alts"
    assert out1["ALT/USDT:USDT"].btc_state == "up"
    assert out2["ALT2/USDT:USDT"].market_context is out1["ALT/USDT:USDT"].market_context
    assert len(svc._client.calls) == 2


def test_fetch_market_context_disabled_returns_unavailable():
    cfg = {
        "timeframes": {"primary": "1h", "higher": "4h", "lower": "15m", "entry": "5m"},
        "order_book": {"depth_levels": 10},
        "trading": {"candle_refresh_seconds": 30},
        "safety": {},
        "market_context": {"enabled": False},
    }
    svc = MarketDataService(object(), cfg)

    context = asyncio.run(svc.fetch_market_context())

    assert context.confidence == 0.0
    assert context.risk_on_state == "unknown"


def test_fetch_snapshot_attaches_time_series_diagnostics(monkeypatch):
    cfg = {
        "timeframes": {"primary": "1h", "higher": "4h", "lower": "15m", "entry": "5m"},
        "order_book": {"depth_levels": 10},
        "trading": {"candle_refresh_seconds": 30},
        "safety": {},
        "smart_money": {"oi_baseline_seconds": 3600},
        "market_context": {"enabled": False},
        "time_series": {"enabled": True, "lookback_bars": 60, "min_returns": 30, "markov_window": 10, "markov_threshold_pct": 0.5},
    }
    svc = MarketDataService(object(), cfg)

    values = [100.0 + i * 0.2 for i in range(80)]

    async def fake_fetch_all(_symbol):
        candles = {"1h": __import__("pandas").DataFrame({"close": values})}
        return candles, {"last": values[-1], "quoteVolume": 1_000_000}, {}, {}, {}, 1.0, 0.5

    svc._fetch_all = fake_fetch_all

    snap = asyncio.run(svc.fetch_snapshot("ALT/USDT:USDT"))

    assert snap.time_series is not None
    assert snap.time_series.observations >= 30
    assert snap.time_series.markov_state in {"bull", "bear", "sideways", "unknown"}


def test_fetch_snapshots_registers_market_metadata_sectors():
    from src.models import strategy_passport as passport

    passport._DYNAMIC_SECTOR_MAP.pop("AUTO", None)
    svc = _service(concurrency=1, timeout=1.0)
    svc._client = SimpleClient = type(
        "SimpleClient",
        (),
        {
            "markets": {
                "AUTO/USDT:USDT": {
                    "base": "AUTO",
                    "info": {"underlyingType": "EQUITY", "underlyingSubType": ["TradFi"]},
                }
            }
        },
    )()

    async def fake_fetch(symbol):
        return MarketSnapshot(symbol=symbol)

    svc.fetch_snapshot = fake_fetch

    asyncio.run(svc.fetch_snapshots(["AUTO/USDT:USDT"]))

    assert passport.sector_for_asset("AUTO") == "tradfi_equity"
