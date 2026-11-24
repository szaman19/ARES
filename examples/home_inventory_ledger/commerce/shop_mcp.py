"""JSON-RPC 2.0 MCP client for Shopify Shop MCP (`https://mcp.shop.com`).

Supports:
- Dynamic MCP `tools/list` discovery
- `search_catalog` (UCP 2026 envelope and flat-argument fallback)
- `get_product` / `lookup_catalog`
- Replacement cart building (`update_cart` / `create_cart` + direct Shop checkout links)
- Explicit `--replay` deterministic catalog fixture for offline demo backup (disclosed in UI)
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote_plus

import httpx

from commerce.cache import CatalogCache, utc_now_iso
from contracts import LedgerItem, Listing


# Disclosed offline backup fixtures ONLY used when `replay_mode=True`
REPLAY_CATALOG_FIXTURES: list[dict[str, Any]] = [
    {
        "listing_id": "gid://shopify/Product/replay-mbp-a2992",
        "title": "Apple MacBook Pro 14-inch M3 Pro (Model A2992) - Space Black, 18GB RAM, 512GB SSD",
        "url": "https://shop.app/m/replay-apple-macbook-pro-a2992",
        "price": 1999.00,
        "merchant": "TechVault Authorized Reseller",
        "sku": "A2992-M3PRO",
        "variant_id": "gid://shopify/ProductVariant/9001",
        "specs": {"brand": "Apple", "model": "A2992", "mpn": "MRX33LL/A", "screen": "14 inch"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-mbp-a2338",
        "title": "Apple MacBook Pro 13-inch M2 (Model A2338) - 8GB Unified Memory, 256GB SSD",
        "url": "https://shop.app/m/replay-apple-macbook-pro-a2338",
        "price": 1299.00,
        "merchant": "TechVault Authorized Reseller",
        "sku": "A2338-M2",
        "variant_id": "gid://shopify/ProductVariant/9002",
        "specs": {"brand": "Apple", "model": "A2338", "mpn": "MNEH3LL/A", "screen": "13 inch"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-mbp-16",
        "title": "Apple MacBook Pro 16-inch M3 Max Laptop",
        "url": "https://shop.app/m/replay-apple-macbook-pro-16",
        "price": 2499.00,
        "merchant": "ProCompute Store",
        "sku": "MBP16-M3",
        "variant_id": "gid://shopify/ProductVariant/9003",
        "specs": {"brand": "Apple", "category": "laptop"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-mba-13",
        "title": "Apple MacBook Air 13-inch Laptop",
        "url": "https://shop.app/m/replay-apple-macbook-air-13",
        "price": 999.00,
        "merchant": "ProCompute Store",
        "sku": "MBA13",
        "variant_id": "gid://shopify/ProductVariant/9004",
        "specs": {"brand": "Apple", "category": "laptop"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-dell-u2723qe",
        "title": "Dell UltraSharp 27 4K USB-C Hub Monitor - U2723QE",
        "url": "https://shop.app/m/replay-dell-u2723qe",
        "price": 579.99,
        "merchant": "DisplayDirect",
        "sku": "U2723QE",
        "variant_id": "gid://shopify/ProductVariant/9005",
        "specs": {"brand": "Dell", "model": "U2723QE", "mpn": "210-BDPF"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-dell-27-generic",
        "title": "Dell 27-inch QHD IPS Monitor USB-C",
        "url": "https://shop.app/m/replay-dell-27-qhd",
        "price": 349.00,
        "merchant": "DisplayDirect",
        "sku": "DELL27QHD",
        "variant_id": "gid://shopify/ProductVariant/9006",
        "specs": {"brand": "Dell", "category": "monitor"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-sony-a7iv",
        "title": "Sony Alpha a7 IV Mirrorless Digital Camera Body (ILCE-7M4)",
        "url": "https://shop.app/m/replay-sony-ilce7m4",
        "price": 2498.00,
        "merchant": "Chelsea Camera & Optics",
        "sku": "ILCE-7M4",
        "variant_id": "gid://shopify/ProductVariant/9007",
        "specs": {"brand": "Sony", "model": "ILCE-7M4", "mpn": "ILCE7M4/B"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-sony-2470gm2",
        "title": "Sony FE 24-70mm f/2.8 GM II Lens (SEL2470GM2)",
        "url": "https://shop.app/m/replay-sony-sel2470gm2",
        "price": 2298.00,
        "merchant": "Chelsea Camera & Optics",
        "sku": "SEL2470GM2",
        "variant_id": "gid://shopify/ProductVariant/9008",
        "specs": {"brand": "Sony", "model": "SEL2470GM2"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-bose-qc-ultra",
        "title": "Bose QuietComfort Ultra Wireless Noise Cancelling Headphones (Model 880066-0100)",
        "url": "https://shop.app/m/replay-bose-qc-ultra",
        "price": 429.00,
        "merchant": "SoundStage NYC",
        "sku": "880066-0100",
        "variant_id": "gid://shopify/ProductVariant/9009",
        "specs": {"brand": "Bose", "model": "880066-0100"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-bose-qc",
        "title": "Bose QuietComfort Wireless Over-Ear Headphones",
        "url": "https://shop.app/m/replay-bose-qc",
        "price": 249.00,
        "merchant": "SoundStage NYC",
        "sku": "BOSE-QC-BLK",
        "variant_id": "gid://shopify/ProductVariant/9010",
        "specs": {"brand": "Bose", "category": "headphones"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-sony-wh1000xm5",
        "title": "Sony WH-1000XM5 Wireless Noise Canceling Headphones",
        "url": "https://shop.app/m/replay-sony-wh1000xm5",
        "price": 398.00,
        "merchant": "SoundStage NYC",
        "sku": "WH1000XM5",
        "variant_id": "gid://shopify/ProductVariant/9011",
        "specs": {"brand": "Sony", "model": "WH-1000XM5"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-keychron-q1",
        "title": "Keychron Q1 Pro QMK/VIA Wireless Custom Mechanical Keyboard",
        "url": "https://shop.app/m/replay-keychron-q1pro",
        "price": 199.00,
        "merchant": "KeyCraft Mechanical",
        "sku": "Q1P-M1",
        "variant_id": "gid://shopify/ProductVariant/9012",
        "specs": {"brand": "Keychron", "model": "Q1P-M1"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-mech-kb-generic",
        "title": "Compact 75% Hot-Swappable Mechanical Keyboard Aluminum",
        "url": "https://shop.app/m/replay-mech-kb-75",
        "price": 109.00,
        "merchant": "KeyCraft Mechanical",
        "sku": "KB75-ALU",
        "variant_id": "gid://shopify/ProductVariant/9013",
        "specs": {"category": "keyboard"},
    },
    {
        "listing_id": "gid://shopify/Product/replay-mbp-case-trap",
        "title": "Hard Shell Protective Case for MacBook Pro 14 A2992 Clear Cover",
        "url": "https://shop.app/m/replay-mbp-case",
        "price": 24.99,
        "merchant": "CaseWorld",
        "sku": "CASE-A2992",
        "variant_id": "gid://shopify/ProductVariant/9099",
        "specs": {"category": "accessory"},
    },
]


def _normalize_price_amount(raw_amount: Any, raw_obj: dict[str, Any] | None = None) -> float | None:
    """Parse UCP/Shopify price amounts safely.

    UCP 2026 `price_range.min.amount` uses integer minor units (cents, e.g. `8999` -> 89.99),
    whereas string amounts (`"89.99"`) or decimal floats (`89.99`) are already in major units.
    """
    if raw_amount is None:
        return None
    if isinstance(raw_amount, str):
        cleaned = raw_amount.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return None
        val = float(cleaned)
        # If decimal point present in string, definitely major units
        if "." in cleaned:
            return round(val, 2)
        # Pure integer string inside a UCP minor-unit envelope
        if raw_obj and ("ucp" in str(raw_obj) or val >= 10000):
            return round(val / 100.0, 2)
        return round(val, 2)
    if isinstance(raw_amount, int):
        # In UCP JSON-RPC responses, integer `amount` inside `{amount, currency}` is cents
        return round(raw_amount / 100.0, 2)
    if isinstance(raw_amount, float):
        return round(raw_amount, 2)
    return None


class ShopMCPClient:
    """Client for Shopify's Shop MCP server (`https://mcp.shop.com`)."""

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        cache: CatalogCache | None = None,
        replay_mode: bool = False,
        timeout: float = 12.0,
    ) -> None:
        self.endpoint = endpoint or os.getenv("SHOP_MCP_URL", "https://mcp.shop.com")
        self.api_key = api_key or os.getenv("SHOP_CATALOG_API_KEY", "")
        self.cache = cache or CatalogCache()
        self.replay_mode = replay_mode
        self.timeout = timeout
        self._req_id = 0
        self.call_count = 0
        self._discovered_tools: set[str] | None = None

    MCP_PROTOCOL_VERSION = "2026-07-28"

    def _headers(self, method: str, tool_name: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.MCP_PROTOCOL_VERSION,
            "Mcp-Method": method,
            "User-Agent": "HomeInventoryLedger/0.1 (Gemini4Hackathon)",
        }
        if tool_name:
            headers["Mcp-Name"] = tool_name
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._req_id += 1
        self.call_count += 1
        rpc_params: dict[str, Any] = dict(params or {})
        rpc_params.setdefault(
            "_meta",
            {
                "io.modelcontextprotocol/protocolVersion": self.MCP_PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientInfo": {
                    "name": "home-inventory-ledger",
                    "version": "0.1.0",
                },
                "io.modelcontextprotocol/clientCapabilities": {},
                "protocolVersion": self.MCP_PROTOCOL_VERSION,
            },
        )
        tool_name = str(rpc_params["name"]) if method == "tools/call" and "name" in rpc_params else None
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": rpc_params,
        }

        with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
            resp = client.post(
                self.endpoint,
                headers=self._headers(method, tool_name=tool_name),
                json=payload,
            )
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                for line in resp.text.splitlines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str and data_str != "[DONE]":
                            return json.loads(data_str)
            return resp.json()

    def list_tools(self) -> set[str]:
        if self._discovered_tools is not None:
            return self._discovered_tools
        if self.replay_mode:
            self._discovered_tools = {"search_catalog", "get_product", "update_cart", "get_cart"}
            return self._discovered_tools
        try:
            res = self._rpc("tools/list", {})
            tools = res.get("result", {}).get("tools", [])
            self._discovered_tools = {t.get("name", "") for t in tools if isinstance(t, dict)}
        except Exception:
            self._discovered_tools = {"search_catalog", "get_product", "lookup_catalog", "update_cart"}
        return self._discovered_tools

    def _search_replay(self, query: str, limit: int = 8) -> list[Listing]:
        q_tokens = [tok for tok in query.lower().replace("-", " ").split() if len(tok) >= 2]
        now = utc_now_iso()
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in REPLAY_CATALOG_FIXTURES:
            hay = " ".join(
                [
                    row["title"],
                    str(row.get("sku", "")),
                    json.dumps(row.get("specs", {})),
                ]
            ).lower()
            hay_norm = hay.replace("-", " ")
            score = sum(2 if tok in hay_norm else 0 for tok in q_tokens)
            # Exact SKU/model boost
            for tok in q_tokens:
                if tok == str(row.get("sku", "")).lower() or tok == str(row.get("specs", {}).get("model", "")).lower():
                    score += 6
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            Listing(
                listing_id=r["listing_id"],
                title=r["title"],
                url=r["url"],
                price=float(r["price"]),
                currency="USD",
                merchant=r.get("merchant"),
                sku=r.get("sku"),
                variant_id=r.get("variant_id"),
                available=True,
                specs=dict(r.get("specs", {})),
                retrieved_at=now,
            )
            for _, r in scored[:limit]
        ]

    def _parse_mcp_products(self, rpc_response: dict[str, Any]) -> list[Listing]:
        result = rpc_response.get("result") or {}
        structured = result.get("structuredContent")
        if not structured:
            for block in result.get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = block.get("text", "")
                    try:
                        structured = json.loads(txt)
                        break
                    except Exception:
                        continue
        if not isinstance(structured, dict):
            if isinstance(structured, list):
                raw_products = structured
            else:
                return []
        else:
            raw_products = (
                structured.get("products")
                or structured.get("items")
                or structured.get("offers")
                or structured.get("results")
                or []
            )

        now = utc_now_iso()
        listings: list[Listing] = []
        for idx, prod in enumerate(raw_products):
            if not isinstance(prod, dict):
                continue
            pid = str(prod.get("id") or prod.get("product_id") or prod.get("url") or f"shop-{idx}")
            title = str(prod.get("title") or prod.get("name") or "").strip()
            url = str(prod.get("url") or prod.get("onlineStoreUrl") or prod.get("link") or "").strip()
            if not title:
                continue

            # Extract price from variants or price_range or direct price
            price: float | None = None
            currency = "USD"
            sku: str | None = prod.get("sku")
            variant_id: str | None = None
            available = True

            variants = prod.get("variants") or []
            offer_merchant: str | None = None
            if isinstance(variants, list) and variants:
                v0 = variants[0] if isinstance(variants[0], dict) else {}
                variant_id = v0.get("id") or v0.get("variant_id")
                sku = sku or v0.get("sku")
                if not url and v0.get("url"):
                    url = str(v0["url"])
                offers = v0.get("offers") or []
                if isinstance(offers, list) and offers and isinstance(offers[0], dict):
                    off0 = offers[0]
                    if not url and off0.get("url"):
                        url = str(off0["url"])
                    seller = off0.get("seller")
                    if isinstance(seller, dict):
                        offer_merchant = seller.get("name") or seller.get("domain")
                    if price is None and off0.get("price") is not None:
                        op = off0["price"]
                        if isinstance(op, dict):
                            currency = str(op.get("currency") or currency)
                            price = _normalize_price_amount(op.get("amount"), structured)
                        else:
                            price = _normalize_price_amount(op, structured)
                v_avail = v0.get("availability")
                if isinstance(v_avail, dict) and "available" in v_avail:
                    available = bool(v_avail["available"])
                elif "availableForSale" in v0:
                    available = bool(v0["availableForSale"])
                v_price = v0.get("price")
                if price is None and isinstance(v_price, dict):
                    currency = str(v_price.get("currency") or v_price.get("currencyCode") or "USD")
                    price = _normalize_price_amount(v_price.get("amount"), structured)
                elif price is None and v_price is not None:
                    price = _normalize_price_amount(v_price, structured)

            if price is None:
                pr = prod.get("price_range") or prod.get("priceRange") or {}
                if isinstance(pr, dict):
                    min_p = pr.get("min") or pr.get("minVariantPrice") or {}
                    if isinstance(min_p, dict):
                        currency = str(min_p.get("currency") or min_p.get("currencyCode") or currency)
                        price = _normalize_price_amount(min_p.get("amount"), structured)
                    elif min_p:
                        price = _normalize_price_amount(min_p, structured)

            if price is None and "price" in prod:
                raw_p = prod["price"]
                if isinstance(raw_p, dict):
                    currency = str(raw_p.get("currency") or currency)
                    price = _normalize_price_amount(raw_p.get("amount"), structured)
                else:
                    price = _normalize_price_amount(raw_p, structured)

            if price is None or price <= 0:
                continue

            media = prod.get("media") or prod.get("images") or []
            image_url: str | None = prod.get("image_url") or prod.get("featuredImage")
            if not image_url and isinstance(media, list) and media:
                m0 = media[0]
                if isinstance(m0, dict):
                    image_url = m0.get("url") or m0.get("src")
                elif isinstance(m0, str):
                    image_url = m0

            merchant = (
                offer_merchant
                or prod.get("vendor")
                or prod.get("merchant")
                or (prod.get("shop", {}) or {}).get("name")
                if isinstance(prod.get("shop"), dict)
                else offer_merchant or prod.get("vendor") or prod.get("merchant")
            )

            specs: dict[str, Any] = {}
            for k in ("brand", "vendor", "mpn", "model", "sku", "barcode", "gtin", "productType", "category"):
                if prod.get(k):
                    specs[k] = prod[k]
            for opt in prod.get("options") or []:
                if isinstance(opt, dict) and str(opt.get("name", "")).lower() == "model":
                    labels = [
                        str(v.get("label"))
                        for v in (opt.get("values") or [])
                        if isinstance(v, dict) and v.get("label")
                    ]
                    if labels and "model" not in specs:
                        specs["model"] = " ".join(labels)
            if sku and "sku" not in specs:
                specs["sku"] = sku
            desc = prod.get("description")
            if isinstance(desc, dict) and desc.get("text"):
                specs["description"] = str(desc["text"])[:400]
            elif isinstance(desc, str) and desc:
                specs["description"] = desc[:400]

            if not url:
                url = f"https://shop.app/search?query={quote_plus(title)}"

            listings.append(
                Listing(
                    listing_id=pid,
                    title=title,
                    url=url,
                    price=price,
                    currency=currency,
                    merchant=str(merchant) if merchant else None,
                    sku=str(sku) if sku else None,
                    variant_id=str(variant_id) if variant_id else None,
                    available=available,
                    specs=specs,
                    image_url=str(image_url) if image_url else None,
                    retrieved_at=now,
                )
            )
        return listings

    def search_catalog(
        self,
        query: str,
        limit: int = 8,
        country: str = "US",
        currency: str = "USD",
        bypass_cache: bool = False,
    ) -> list[Listing]:
        """Search global Shopify catalog via Shop MCP with 10-min TTL caching."""
        clean_q = " ".join(query.strip().split())[:120]
        if not clean_q:
            return []

        if not bypass_cache:
            cached = self.cache.get_search(clean_q)
            if cached is not None:
                return cached[:limit]

        if self.replay_mode:
            results = self._search_replay(clean_q, limit=limit)
            self.cache.set_search(clean_q, results)
            return results

        # Attempt 1: UCP 2026 catalog envelope
        arg_attempts = [
            {
                "catalog": {
                    "query": clean_q,
                    "context": {"address_country": country, "currency": currency},
                    "pagination": {"limit": limit},
                }
            },
            {
                "query": clean_q,
                "context": "home inventory replacement cost lookup",
                "limit": limit,
            },
        ]

        listings: list[Listing] = []
        for args in arg_attempts:
            try:
                rpc_res = self._rpc("tools/call", {"name": "search_catalog", "arguments": args})
                if "error" in rpc_res:
                    continue
                listings = self._parse_mcp_products(rpc_res)
                if listings:
                    break
            except Exception:
                continue

        if listings:
            self.cache.set_search(clean_q, listings)
        return listings[:limit]

    def get_product(self, product_id: str) -> Listing | None:
        cached = self.cache.get_product(product_id)
        if cached is not None:
            return cached
        if self.replay_mode:
            for row in REPLAY_CATALOG_FIXTURES:
                if row["listing_id"] == product_id:
                    item = Listing(
                        listing_id=row["listing_id"],
                        title=row["title"],
                        url=row["url"],
                        price=float(row["price"]),
                        merchant=row.get("merchant"),
                        sku=row.get("sku"),
                        variant_id=row.get("variant_id"),
                        specs=dict(row.get("specs", {})),
                        retrieved_at=utc_now_iso(),
                    )
                    self.cache.set_product(item)
                    return item
            return None

        try:
            rpc_res = self._rpc(
                "tools/call",
                {"name": "get_product", "arguments": {"product_id": product_id, "id": product_id}},
            )
            parsed = self._parse_mcp_products(rpc_res)
            if parsed:
                self.cache.set_product(parsed[0])
                return parsed[0]
        except Exception:
            return None
        return None

    def create_replacement_cart(self, items: list[LedgerItem]) -> dict[str, Any]:
        """Build a 'Rebuild My Home' Shop replacement cart for the commerce bonus track.

        Attempts live MCP `update_cart` if variant IDs are available, and always returns
        verified line items with direct merchant/Shop checkout links (never auto-purchasing).
        """
        cart_lines: list[dict[str, Any]] = []
        mcp_line_inputs: list[dict[str, Any]] = []

        for item in items:
            listing = item.matched_listing or (
                item.candidate_listings[0] if item.candidate_listings else None
            )
            if not listing:
                continue
            line_total = round(listing.price * item.quantity, 2)
            cart_lines.append(
                {
                    "item_id": item.item_id,
                    "display_name": item.display_name,
                    "certification": item.price_status,
                    "quantity": item.quantity,
                    "unit_price": listing.price,
                    "line_total": line_total,
                    "listing_title": listing.title,
                    "merchant": listing.merchant or "Shop Merchant",
                    "url": listing.url,
                    "variant_id": listing.variant_id,
                    "retrieved_at": listing.retrieved_at,
                }
            )
            if listing.variant_id:
                mcp_line_inputs.append(
                    {"merchandise_id": listing.variant_id, "quantity": item.quantity}
                )

        checkout_url: str | None = None
        cart_id: str | None = None
        if not self.replay_mode and mcp_line_inputs:
            try:
                rpc_res = self._rpc(
                    "tools/call",
                    {"name": "update_cart", "arguments": {"add_items": mcp_line_inputs}},
                )
                structured = (rpc_res.get("result") or {}).get("structuredContent") or {}
                cart_obj = structured.get("cart") or structured
                if isinstance(cart_obj, dict):
                    checkout_url = cart_obj.get("checkout_url") or cart_obj.get("checkoutUrl")
                    cart_id = cart_obj.get("id")
            except Exception:
                checkout_url = None

        if not checkout_url and cart_lines:
            # Safe user-confirmed Shop search/bundle URL when anonymous MCP cart needs buyer OAuth
            top_query = quote_plus(cart_lines[0]["listing_title"][:60])
            checkout_url = cart_lines[0]["url"] or f"https://shop.app/search?query={top_query}"

        subtotal = round(sum(line["line_total"] for line in cart_lines), 2)
        return {
            "cart_id": cart_id or "shop-replacement-manifest",
            "line_count": len(cart_lines),
            "subtotal_usd": subtotal,
            "currency": "USD",
            "checkout_url": checkout_url,
            "requires_user_confirmation": True,
            "lines": cart_lines,
        }
