"""Schedule of Loss exporters (CSV and print-ready HTML/PDF report) with freshness re-checks."""

from __future__ import annotations

import base64
import csv
from datetime import datetime, timezone
import html
import io
from pathlib import Path

from commerce.shop_mcp import ShopMCPClient
from contracts import LedgerBoard
from ledger.board import compute_board_totals
from ledger.verifiers import listing_freshness, total_arithmetic


def refresh_stale_prices_for_export(
    board: LedgerBoard,
    shop_client: ShopMCPClient | None = None,
) -> LedgerBoard:
    """Re-verify `retrieved_at <= 10 min` per shop.dev rules before exporting a Schedule of Loss."""
    if shop_client is None:
        return board

    updated_items = dict(board.items)
    updated_listings = dict(board.listings_index)

    for iid, item in board.items.items():
        if item.matched_listing:
            fresh, _ = listing_freshness(item.matched_listing)
            if not fresh:
                refreshed = shop_client.get_product(item.matched_listing.listing_id)
                if refreshed is None and item.brand and item.model:
                    res = shop_client.search_catalog(
                        f"{item.brand} {item.model}", limit=3, bypass_cache=True
                    )
                    refreshed = res[0] if res else None
                if refreshed:
                    updated_listings[refreshed.listing_id] = refreshed
                    updated_items[iid] = item.model_copy(
                        update={
                            "matched_listing": refreshed,
                            "unit_price": refreshed.price if item.price_status == "CERTIFIED" else None,
                        }
                    )

    return board.model_copy(update={"items": updated_items, "listings_index": updated_listings})


def export_schedule_csv(board: LedgerBoard) -> str:
    """Generate an insurer-ready Schedule of Loss CSV with provenance & listing timestamps."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "item_id",
            "room",
            "category",
            "display_name",
            "brand",
            "model",
            "quantity",
            "identity_status",
            "price_status",
            "certified_unit_price_usd",
            "estimate_low_usd",
            "estimate_high_usd",
            "line_total_or_range_usd",
            "frame_timestamp_s",
            "listing_title",
            "listing_url",
            "listing_retrieved_at",
            "flags",
        ]
    )

    for item in board.items.values():
        q = item.quantity
        if item.price_status == "CERTIFIED" and item.unit_price is not None:
            line_val = f"{item.unit_price * q:.2f}"
        elif item.price_status == "UNCERTIFIED":
            lo = (item.price_low or 0.0) * q
            hi = (item.price_high or lo) * q
            line_val = f"{lo:.2f}-{hi:.2f}"
        else:
            line_val = "UNKNOWN"

        lst = item.matched_listing
        writer.writerow(
            [
                item.item_id,
                item.room,
                item.category,
                item.display_name,
                item.brand or "",
                item.model or "",
                q,
                item.identity_status,
                item.price_status,
                f"{item.unit_price:.2f}" if item.unit_price is not None else "",
                f"{item.price_low:.2f}" if item.price_low is not None else "",
                f"{item.price_high:.2f}" if item.price_high is not None else "",
                line_val,
                f"{item.first_seen_ms / 1000.0:.1f}",
                lst.title if lst else "",
                lst.url if lst else "",
                lst.retrieved_at if lst else "",
                "; ".join(item.flags),
            ]
        )

    totals = compute_board_totals(board)
    rows_check = [it.model_dump() for it in board.items.values()]
    arith_ok, _ = total_arithmetic(
        rows_check,
        totals["certified_total"],
        totals["estimate_low"],
        totals["estimate_high"],
    )
    writer.writerow([])
    writer.writerow(
        [
            "SUMMARY",
            f"Coverage C Limit: {board.policy.coverage_c_limit or 'N/A'}",
            f"Certified Total: ${totals['certified_total']:,.2f}",
            f"Estimated Range: ${totals['estimate_low']:,.2f}-${totals['estimate_high']:,.2f}",
            f"Verdict: {board.decision.verdict if board.decision else 'unknown'}",
            f"Arithmetic Verified: {arith_ok}",
        ]
    )
    return buf.getvalue()


def export_schedule_html(board: LedgerBoard) -> str:
    """Generate a standalone print-to-PDF Schedule of Loss report with embedded frame proof."""
    totals = compute_board_totals(board)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = board.decision.verdict.replace("_", " ").upper() if board.decision else "PENDING"
    conf = f"{board.decision.confidence:.2f}" if board.decision else "—"
    limit_str = f"${board.policy.coverage_c_limit:,.2f}" if board.policy.coverage_c_limit else "Not provided"

    row_html: list[str] = []
    for it in board.items.values():
        thumb = ""
        if it.best_frame_path and Path(it.best_frame_path).exists():
            b64 = base64.b64encode(Path(it.best_frame_path).read_bytes()).decode("ascii")
            thumb = f'<img src="data:image/jpeg;base64,{b64}" style="width:88px;height:56px;object-fit:cover;border-radius:6px;" />'

        if it.price_status == "CERTIFIED" and it.unit_price is not None:
            price_str = f"<strong>${it.unit_price * it.quantity:,.2f}</strong>"
            badge_cls = "cert"
        elif it.price_status == "UNCERTIFIED":
            lo = (it.price_low or 0.0) * it.quantity
            hi = (it.price_high or lo) * it.quantity
            price_str = f"${lo:,.0f} – ${hi:,.0f}"
            badge_cls = "uncert"
        else:
            price_str = "Needs Appraisal / Unknown"
            badge_cls = "unk"

        src_link = ""
        if it.matched_listing:
            src_link = (
                f'<a href="{html.escape(it.matched_listing.url)}" target="_blank">'
                f'{html.escape(it.matched_listing.title[:48])}…</a><br/>'
                f'<small>Retrieved: {html.escape(it.matched_listing.retrieved_at)}</small>'
            )
        elif it.candidate_listings:
            src_link = f"<small>{len(it.candidate_listings)} comparable Shop listings (25th–75th pct)</small>"

        flags_html = ""
        if it.conflict_detail:
            flags_html = f'<div class="conflict">⚠ {html.escape(it.conflict_detail)}</div>'

        row_html.append(
            f"""
            <tr>
              <td>{thumb}<br/><small>@{it.first_seen_ms/1000:.1f}s</small></td>
              <td><strong>{html.escape(it.display_name)}</strong><br/><small>{html.escape(it.room)} · {html.escape(it.description)}</small>{flags_html}</td>
              <td>{it.quantity}</td>
              <td><span class="badge {badge_cls}">{it.price_status}</span></td>
              <td>{price_str}</td>
              <td>{src_link}</td>
            </tr>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Schedule of Loss — Home Inventory Ledger</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #141824; margin: 32px; }}
  h1 {{ margin-bottom: 4px; }}
  .meta {{ color: #5b6478; margin-bottom: 20px; font-size: 13px; }}
  .summary {{ display: flex; gap: 20px; background: #f5f7fb; padding: 16px 20px; border-radius: 12px; margin-bottom: 24px; }}
  .summary div {{ flex: 1; }}
  .summary .lbl {{ font-size: 11px; text-transform: uppercase; color: #5b6478; font-weight: 700; }}
  .summary .val {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 10px 8px; border-bottom: 1px solid #e2e6ef; vertical-align: top; }}
  th {{ font-size: 11px; text-transform: uppercase; color: #5b6478; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; color: #fff; }}
  .cert {{ background: #16a34a; }}
  .uncert {{ background: #d98c14; }}
  .unk {{ background: #64748b; }}
  .conflict {{ margin-top: 4px; color: #b91c1c; font-size: 11px; font-weight: 600; }}
  .disclaimer {{ margin-top: 24px; font-size: 11px; color: #64748b; }}
  @media print {{ .no-print {{ display: none; }} }}
</style>
</head>
<body>
  <button class="no-print" onclick="window.print()" style="float:right;padding:8px 14px;cursor:pointer;">Print / Save PDF</button>
  <h1>Pre-Loss Schedule of Contents &amp; Coverage Report</h1>
  <div class="meta">Generated {now_str} · Session {html.escape(board.session_id)} · Board v{board.board_version}</div>
  <div class="summary">
    <div><div class="lbl">Coverage Verdict</div><div class="val">{html.escape(verdict)} <small style="font-size:13px;font-weight:400;">(conf {conf})</small></div></div>
    <div><div class="lbl">Policy Limit (Coverage C)</div><div class="val">{limit_str}</div></div>
    <div><div class="lbl">Certified Floor</div><div class="val" style="color:#16a34a;">${totals['certified_total']:,.2f}</div></div>
    <div><div class="lbl">Estimated Replacement Total</div><div class="val">${totals['estimate_low']:,.0f} – ${totals['estimate_high']:,.0f}</div></div>
  </div>
  <table>
    <thead>
      <tr><th>Visual Proof</th><th>Item &amp; Provenance</th><th>Qty</th><th>Status</th><th>Replacement (New)</th><th>Catalog Citation</th></tr>
    </thead>
    <tbody>
      {''.join(row_html)}
    </tbody>
  </table>
  <div class="disclaimer">
    Informational pre-loss inventory only — not a legal coverage determination or financial advice.
    CERTIFIED prices require both a legible hardware model identifier in video and an exact live catalog match.
  </div>
</body>
</html>"""
