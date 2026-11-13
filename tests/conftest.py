"""Shared pytest configuration.

PyTorch and TensorFlow can crash when loaded into the same process (their bundled
LLVM builds register duplicate symbols). The two suites therefore run in separate
processes — use ``make test`` or:

    pytest tests/test_pinn.py
    pytest tests/test_surrogate.py

If a single ``pytest`` session would import both frameworks, the second suite is
skipped at collection time with a note in the summary instead of crashing.
"""

from __future__ import annotations

_collected: set[str] = set()
_skipped: list[str] = []


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    """Collect at most one of the two framework suites per session.

    Collection imports test modules (and hence the frameworks) up front, so the check
    is on which suite was already accepted, not on ``sys.modules``.
    """
    name = collection_path.name
    if name not in {"test_pinn.py", "test_surrogate.py"}:
        return None
    other = "test_surrogate.py" if name == "test_pinn.py" else "test_pinn.py"
    if other in _collected:
        _skipped.append(name)
        return True
    _collected.add(name)
    return None


def pytest_terminal_summary(terminalreporter):
    for name in _skipped:
        terminalreporter.write_line(
            f"NOTE: {name} was not collected: PyTorch and TensorFlow cannot share one "
            f"process. Run `pytest tests/{name}` separately (or `make test`).",
            yellow=True,
        )
