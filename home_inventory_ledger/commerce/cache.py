"""TTL cache honoring shop.dev caching rules.

Per shop.dev requirements:
- Price and availability MUST NOT be cached longer than 10 minutes (600 s).
- Product metadata may be cached up to 24 hours (86,400 s).
- Commissions are never cached.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from contracts import Listing

PRICE_TTL_SECONDS = 600  # 10 minutes strict limit
METADATA_TTL_SECONDS = 86_400  # 24 hours


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_listing_fresh(retrieved_at: str, max_age_seconds: int = PRICE_TTL_SECONDS) -> bool:
    """Return True if ISO-8601 `retrieved_at` is within `max_age_seconds` of current UTC time."""
    try:
        norm = retrieved_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(norm)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        return -5.0 <= age <= max_age_seconds
    except Exception:
        return False


class CatalogCache:
    """In-memory + optional JSON disk cache enforcing 10-minute price TTL."""

    def __init__(self, cache_file: str | Path | None = None, price_ttl: int = PRICE_TTL_SECONDS) -> None:
        self.price_ttl = price_ttl
        self.cache_file = Path(cache_file) if cache_file else None
        self._queries: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._products: dict[str, tuple[float, dict[str, Any]]] = {}
        if self.cache_file and self.cache_file.exists():
            self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self.cache_file.read_text()) if self.cache_file else {}
            self._queries = {
                k: (float(v[0]), v[1]) for k, v in raw.get("queries", {}).items()
            }
            self._products = {
                k: (float(v[0]), v[1]) for k, v in raw.get("products", {}).items()
            }
        except Exception:
            self._queries = {}
            self._products = {}

    def _save(self) -> None:
        if not self.cache_file:
            return
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"queries": self._queries, "products": self._products}
        self.cache_file.write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _norm_key(query: str) -> str:
        return " ".join(query.lower().strip().split())

    def get_search(self, query: str) -> list[Listing] | None:
        key = self._norm_key(query)
        entry = self._queries.get(key)
        if not entry:
            return None
        stored_ts, raw_list = entry
        if (time.time() - stored_ts) > self.price_ttl:
            del self._queries[key]
            return None
        listings = [Listing.model_validate(item) for item in raw_list]
        if any(not is_listing_fresh(item.retrieved_at, self.price_ttl) for item in listings):
            del self._queries[key]
            return None
        return listings

    def set_search(self, query: str, listings: list[Listing]) -> None:
        key = self._norm_key(query)
        self._queries[key] = (
            time.time(),
            [item.model_dump(mode="json") for item in listings],
        )
        self._save()

    def get_product(self, product_id: str) -> Listing | None:
        entry = self._products.get(product_id)
        if not entry:
            return None
        stored_ts, raw = entry
        if (time.time() - stored_ts) > self.price_ttl:
            del self._products[product_id]
            return None
        listing = Listing.model_validate(raw)
        if not is_listing_fresh(listing.retrieved_at, self.price_ttl):
            del self._products[product_id]
            return None
        return listing

    def set_product(self, listing: Listing) -> None:
        self._products[listing.listing_id] = (time.time(), listing.model_dump(mode="json"))
        self._save()
