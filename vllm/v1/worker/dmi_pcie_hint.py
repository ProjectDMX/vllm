"""Optional DMI PCIe hint bridge for vendored vLLM workers."""

try:
    from monitoring.governor import emit_hint as _emit_hint
except Exception:
    _emit_hint = None


def emit_dmi_pcie_hint(
    *,
    direction: str,
    source: str,
    est_bytes: int = 0,
    valid_until_ns: int = 0,
) -> None:
    """Emit a best-effort hint without coupling upstream vLLM to DMI."""

    if _emit_hint is None:
        return
    try:
        _emit_hint(
            direction=direction,
            source=source,
            est_bytes=est_bytes,
            valid_until_ns=valid_until_ns,
        )
    except Exception:
        pass
