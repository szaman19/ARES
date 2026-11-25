"""Optional isolated workspace sandboxes for ARES v2 (strictly outside core)."""

from ares.sandbox.git import CommandResult, GitWorktreeSandbox

__all__ = [
    "CommandResult",
    "GitWorktreeSandbox",
]
