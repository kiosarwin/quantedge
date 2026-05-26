"""Explicit setup passport for futures strategy routing."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import re

import yaml


@dataclass
class SetupPassport:
    symbol: str
    asset: str
    sector: str
    rotation_state: str
    direction: str
    setup_type: str
    sleeve: str
    regime: str
    sm_phase: str
    quality_score: float
    alignment: float
    participation_score: float
    crowding_state: str
    funding_state: str
    microstructure_state: str
    invalidation: str
    expected_path: str
    hold_profile: str
    decision: str
    reason: str

    def as_dict(self) -> dict:
        return asdict(self)


_SECTOR_MAP = {
    "BTC": "majors",
    "ETH": "majors",
    "BNB": "majors",
    "SOL": "l1",
    "AVAX": "l1",
    "NEAR": "l1",
    "SUI": "l1",
    "APT": "l1",
    "SEI": "l1",
    "INJ": "l1",
    "ADA": "l1",
    "TON": "l1",
    "TRX": "l1",
    "ARB": "l2",
    "OP": "l2",
    "STRK": "l2",
    "MANTA": "l2",
    "ERA": "l2",
    "DOGE": "meme",
    "SHIB": "meme",
    "PEPE": "meme",
    "1000PEPE": "meme",
    "WIF": "meme",
    "BONK": "meme",
    "FLOKI": "meme",
    "CHILLGUY": "meme",
    "AI": "ai",
    "TAO": "ai",
    "FET": "ai",
    "RENDER": "ai",
    "WLD": "ai",
    "ZEREBRO": "ai",
    "SKYAI": "ai",
    "GRASS": "depin",
    "FIL": "depin",
    "AR": "depin",
    "PHA": "depin",
    "DYM": "modular",
    "TIA": "modular",
    "ALT": "modular",
    "RIF": "btc_ecosystem",
    "BCH": "btc_ecosystem",
    "ORDI": "btc_ecosystem",
    "STX": "btc_ecosystem",
    "JCT": "microcap",
    "BEAT": "microcap",
    "TAG": "microcap",
    "IN": "microcap",
    "BOB": "microcap",
    "NAORIS": "microcap",
    "DEXE": "defi",
    "ENA": "defi",
    "UNI": "defi",
    "ASTER": "perp_dex",
    "HYPE": "perp_dex",
    "LINK": "oracle",
    "ONDO": "rwa",
    "EDEN": "rwa",
    "PLUME": "rwa",
    "ZEC": "privacy",
    "NIL": "privacy_ai",
    "LIT": "privacy_infra",
    "SAGA": "gaming",
    "FIDA": "solana_ecosystem",
    "XRP": "payments",
    "CL": "commodity",
    "MU": "tradfi_equity",
    "XAG": "commodity",
    "XAU": "commodity",
    "BZ": "new_listing",
    "GENIUS": "new_listing",
    "EIGEN": "restaking",
}


def asset_from_symbol(symbol: str) -> str:
    base = str(symbol or "").split("/")[0].split(":")[0].upper()
    return base or "UNKNOWN"


_BINANCE_ALPHA_ASSETS_PATH = Path(__file__).resolve().parents[2] / "config" / "binance_alpha_assets.yaml"


@lru_cache(maxsize=1)
def binance_alpha_assets() -> frozenset[str]:
    try:
        payload = yaml.safe_load(_BINANCE_ALPHA_ASSETS_PATH.read_text()) or {}
    except OSError:
        return frozenset()
    assets = payload.get("assets", []) if isinstance(payload, dict) else []
    return frozenset(str(asset or "").upper() for asset in assets if str(asset or "").strip())

_DYNAMIC_SECTOR_MAP: dict[str, str] = {}


def _market_info_values(market: dict) -> tuple[str, list[str]]:
    info = market.get("info", {}) if isinstance(market, dict) else {}
    underlying_type = str(info.get("underlyingType") or market.get("underlyingType") or "").strip().lower()
    raw_subtypes = info.get("underlyingSubType") or market.get("underlyingSubType") or []
    if isinstance(raw_subtypes, str):
        subtypes = [raw_subtypes]
    elif isinstance(raw_subtypes, (list, tuple, set)):
        subtypes = [str(item) for item in raw_subtypes]
    else:
        subtypes = []
    return underlying_type, [item.strip().lower() for item in subtypes if str(item).strip()]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sector_from_market_metadata(market: dict) -> str:
    underlying_type, subtypes = _market_info_values(market)
    labels = [underlying_type, *subtypes]
    label_text = " ".join(labels)

    if underlying_type == "equity" or "tradfi" in label_text or "stock" in label_text:
        return "tradfi_equity"
    if underlying_type in {"commodity", "commodities"} or any(
        key in label_text for key in ("commodity", "commodities", "metal", "metals", "energy")
    ):
        return "commodity"
    if underlying_type in {"index", "indices"} or "index" in label_text:
        return "tradfi_index"
    if "layer 1" in label_text or "layer-1" in label_text or "l1" in labels:
        return "l1"
    if "layer 2" in label_text or "layer-2" in label_text or "l2" in labels:
        return "l2"
    if "meme" in label_text:
        return "meme"
    if "ai" in labels or "artificial intelligence" in label_text:
        return "ai"
    if "depin" in label_text or "storage" in label_text:
        return "depin"
    if "defi" in label_text or "dex" in label_text:
        return "defi"
    if "rwa" in labels or "real world asset" in label_text or "real-world asset" in label_text:
        return "rwa"
    if "oracle" in label_text:
        return "oracle"
    if "privacy" in label_text:
        return "privacy"
    if "gaming" in label_text or "gamefi" in label_text:
        return "gaming"
    if "payment" in label_text or "payments" in label_text:
        return "payments"
    if underlying_type == "crypto" and subtypes:
        return f"crypto_{_slug(subtypes[0])}"
    return "other"


def register_market_sector(asset: str, market: dict) -> str:
    asset = str(asset or "").upper()
    if not asset:
        return "other"
    if asset in binance_alpha_assets():
        return "binance_alpha"
    if asset in _SECTOR_MAP:
        return _SECTOR_MAP[asset]
    sector = sector_from_market_metadata(market)
    if sector != "other":
        _DYNAMIC_SECTOR_MAP[asset] = sector
    return sector


def register_market_sectors(markets: dict) -> dict[str, str]:
    registered: dict[str, str] = {}
    for symbol, market in (markets or {}).items():
        if not isinstance(market, dict):
            continue
        asset = str(market.get("base") or asset_from_symbol(str(symbol))).upper()
        sector = register_market_sector(asset, market)
        if sector != "other":
            registered[asset] = sector
    return registered



def sector_for_asset(asset: str) -> str:
    asset = str(asset or "").upper()
    if asset in binance_alpha_assets():
        return "binance_alpha"
    if asset in _SECTOR_MAP:
        return _SECTOR_MAP[asset]
    return _DYNAMIC_SECTOR_MAP.get(asset, "other")


def classify_rotation_state(breakdown, sector: str) -> str:
    direction = str(getattr(breakdown, "direction", "") or "")
    fv = getattr(breakdown, "feature_vector", None)
    momentum = float(getattr(fv, "momentum_strength", 0.0) or 0.0) if fv else 0.0
    oi = float(getattr(fv, "oi_change_rate", 0.0) or 0.0) if fv else 0.0
    vol = float(getattr(fv, "volume_anomaly_score", 0.0) or 0.0) if fv else 0.0
    if sector in {"majors"} and abs(momentum) >= 25.0:
        return "major_beta_lead"
    if sector in {"meme", "microcap"} and vol >= 55.0 and oi >= 0.5:
        return "speculative_rotation"
    if direction == "long" and momentum >= 20.0 and oi >= 0.0:
        return "risk_on_rotation"
    if direction == "short" and momentum <= -20.0 and oi <= 0.0:
        return "risk_off_rotation"
    if vol >= 60.0:
        return "volume_rotation"
    return "unclassified_rotation"
