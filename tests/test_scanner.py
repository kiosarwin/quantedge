import asyncio

from src.scanner.scanner import PairScanner


class _FailingTickerClient:
    markets = {}

    async def fetch_tickers(self):
        raise TimeoutError("ticker endpoint timed out")


def _cfg():
    return {
        "filters": {
            "min_24h_volume_usdt": 100_000_000,
            "min_price_usdt": 0.001,
            "blacklisted_pairs": [],
            "stablecoins": [],
        },
        "trading": {},
    }


def test_scanner_returns_empty_list_when_ticker_fetch_fails(caplog):
    scanner = PairScanner(_FailingTickerClient(), _cfg())

    result = asyncio.run(scanner.scan())

    assert result == []
    assert "Scanner ticker fetch failed" in caplog.text


class _TickerClient:
    markets = {
        "BTC/USDT:USDT": {"type": "swap", "active": True, "linear": True, "base": "BTC", "quote": "USDT", "settle": "USDT"},
        "EDGE/USDT:USDT": {"type": "swap", "active": True, "linear": True, "base": "EDGE", "quote": "USDT", "settle": "USDT"},
        "LOWVOL/USDT:USDT": {"type": "swap", "active": True, "linear": True, "base": "LOWVOL", "quote": "USDT", "settle": "USDT"},
    }

    async def fetch_tickers(self):
        return {
            "BTC/USDT:USDT": {"quoteVolume": 900_000_000, "last": 50000},
            "EDGE/USDT:USDT": {"quoteVolume": 500_000_000, "last": 1.23},
            "LOWVOL/USDT:USDT": {"quoteVolume": 10_000, "last": 0.42},
        }


def test_scanner_moves_priority_symbols_to_front_even_when_volume_eligible():
    scanner = PairScanner(_TickerClient(), _cfg())

    result = asyncio.run(scanner.scan(priority_symbols=["EDGE/USDT:USDT"]))

    assert result[:2] == ["EDGE/USDT:USDT", "BTC/USDT:USDT"]


def test_scanner_includes_valid_priority_symbol_below_volume_floor():
    scanner = PairScanner(_TickerClient(), _cfg())

    result = asyncio.run(scanner.scan(priority_symbols=["LOWVOL/USDT:USDT"]))

    assert result[0] == "LOWVOL/USDT:USDT"
    assert "BTC/USDT:USDT" in result
