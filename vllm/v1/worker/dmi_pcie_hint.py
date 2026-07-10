# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility import for the DMI PCIe hint bridge."""

from vllm.distributed.kv_transfer.dmi_pcie_hint import (
    D2HHintLease,
    connector_manages_dmi_pcie_hints,
    emit_dmi_pcie_hint,
)

__all__ = [
    "D2HHintLease",
    "connector_manages_dmi_pcie_hints",
    "emit_dmi_pcie_hint",
]
