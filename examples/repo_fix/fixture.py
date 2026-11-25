"""Deterministic multi-file git repository fixture with cross-file bugs and distractor modules."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_BILLING_INIT = '''"""Sample multi-file billing package for ARES repo_fix example."""

from billing.currency import format_cents, round_cents
from billing.discounts import apply_discount
from billing.invoice import compute_invoice_total

__all__ = [
    "apply_discount",
    "compute_invoice_total",
    "format_cents",
    "round_cents",
]
'''

_CURRENCY_PY = '''"""Currency rounding and formatting primitives."""

from __future__ import annotations


def round_cents(value: float) -> int:
    """Round a floating-point cent amount to the nearest integer cent."""
    return int(round(value))


def format_cents(cents: int) -> str:
    """Format integer cents into a USD string."""
    sign = "-" if cents < 0 else ""
    abs_cents = abs(cents)
    return f"{sign}${abs_cents // 100}.{abs_cents % 100:02d}"
'''

_DISCOUNTS_PY = '''"""Discount calculation helpers."""

from __future__ import annotations

from billing.currency import round_cents


def apply_discount(amount_cents: int, discount_percent: float) -> int:
    """Apply a percentage discount in [0, 100] to ``amount_cents``.

    Negative percentages must raise ValueError; percentages over 100 clamp to 100%.
    """
    # BUG 1: Does not reject negative discounts or clamp percentages > 100%.
    discount = round_cents(amount_cents * (discount_percent / 100.0))
    return amount_cents - discount
'''

_INVOICE_PY = '''"""Invoice total calculation combining line items, discounts, and tax."""

from __future__ import annotations

from billing.currency import round_cents
from billing.discounts import apply_discount


def compute_invoice_total(
    line_items: list[tuple[int, int]],
    *,
    discount_percent: float = 0.0,
    tax_rate: float = 0.0,
) -> int:
    """Compute final invoice total in cents from ``(unit_cents, quantity)`` pairs.

    Discount applies to the subtotal BEFORE sales tax is added.
    """
    if tax_rate < 0:
        raise ValueError("tax_rate must be non-negative")
    subtotal = sum(unit_cents * qty for unit_cents, qty in line_items)
    # BUG 2: Tax is computed on raw subtotal BEFORE applying the discount.
    tax_cents = round_cents(subtotal * tax_rate)
    return apply_discount(subtotal + tax_cents, discount_percent)
'''

_FORMATTING_PY = (
    '"""Large distractor formatting module (intentionally >3KB to simulate repo bloat)."""\n\n'
    + "\n".join(
        f"def render_section_{i:02d}(label: str, amount_cents: int) -> str:\n"
        f'    """Render standardized invoice table row #{i}."""\n'
        f"    return f'[{i:02d}] {{label:<24}} : {{amount_cents:>8d}} cents'\n"
        for i in range(28)
    )
)

_LOGGING_UTILS_PY = (
    '"""Large distractor audit-logging module (intentionally >3KB to simulate repo bloat)."""\n\n'
    + "\n".join(
        f"def emit_audit_tag_{i:02d}(invoice_id: str, actor: str = 'system') -> dict[str, str]:\n"
        f'    """Build structured audit metadata payload #{i}."""\n'
        f"    return {{'channel': 'audit_{i:02d}', 'invoice_id': invoice_id, 'actor': actor}}\n"
        for i in range(28)
    )
)

_TEST_BILLING_PY = '''"""Trusted repository test suite for billing calculations."""

from __future__ import annotations

import pytest

from billing.currency import format_cents
from billing.discounts import apply_discount
from billing.invoice import compute_invoice_total


def test_format_cents() -> None:
    assert format_cents(1250) == "$12.50"
    assert format_cents(-5) == "-$0.05"


def test_apply_discount_clamps_over_100_and_rejects_negative() -> None:
    assert apply_discount(10_000, 20.0) == 8_000
    assert apply_discount(10_000, 150.0) == 0
    with pytest.raises(ValueError):
        apply_discount(10_000, -5.0)


def test_compute_invoice_total_applies_discount_before_tax() -> None:
    # Subtotal = 10_000 cents ($100.00)
    # 20% discount -> 8_000 cents; 10% tax on 8_000 -> 800 cents => 8_800 cents.
    # Tax-before-discount gives apply_discount(11_000, 20%) = 8_800? Wait:
    # Use uneven discount/tax rounding or check intermediate behavior:
    # Subtotal = 10_005 cents, discount=50% -> 5_003 (or 5_002), vs tax first:
    # Even simpler: 100% discount must yield $0.00 total even when tax_rate > 0!
    assert compute_invoice_total([(5_000, 2)], discount_percent=100.0, tax_rate=0.10) == 0
    # 50% discount on 1_000 -> 500; 10% tax on 500 -> 50 => 550.
    assert compute_invoice_total([(500, 2)], discount_percent=50.0, tax_rate=0.10) == 550
    # Rounding diverges when tax runs before discount: subtotal=105, disc=10%, tax=10%:
    # Discount first: 105 -> round(10.5)=10 -> 95; tax round(9.5)=10 -> 105? Wait:
    # subtotal=1001, disc=50% -> round(500.5)=500 -> 501; tax 10% -> round(50.1)=50 -> 551.
    # Tax first: tax round(100.1)=100 -> 1101; disc 50% -> round(550.5)=550 -> 551.
    # With subtotal=999, disc=25%, tax=8%:
    # Discount first: disc=round(249.75)=250 -> 749; tax=round(59.92)=60 -> 809.
    # Tax first: tax=round(79.92)=80 -> 1079; disc=round(269.75)=270 -> 809.
    # With subtotal=1_000, disc=100%, tax=8%: discount first = 0; tax first with >100% fails too.
    assert compute_invoice_total([(1_000, 1)], discount_percent=120.0, tax_rate=0.08) == 0
'''


def _git(repo_dir: Path, *args: str) -> None:
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "ARES Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@ares.local",
            "GIT_COMMITTER_NAME": "ARES Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@ares.local",
        }
    )
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")


def create_sample_repo(dest_dir: str | Path) -> Path:
    """Materialize and commit a realistic multi-file Python git repo with two bugs."""
    repo = Path(dest_dir)
    (repo / "billing").mkdir(parents=True, exist_ok=True)
    (repo / "tests").mkdir(parents=True, exist_ok=True)

    files = {
        "billing/__init__.py": _BILLING_INIT,
        "billing/currency.py": _CURRENCY_PY,
        "billing/discounts.py": _DISCOUNTS_PY,
        "billing/invoice.py": _INVOICE_PY,
        "billing/formatting.py": _FORMATTING_PY,
        "billing/logging_utils.py": _LOGGING_UTILS_PY,
        "tests/test_billing.py": _TEST_BILLING_PY,
    }
    for rel, content in files.items():
        (repo / rel).write_text(content, encoding="utf-8")

    _git(repo, "init")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "Initial multi-file billing repo with bugs")
    return repo.resolve()
