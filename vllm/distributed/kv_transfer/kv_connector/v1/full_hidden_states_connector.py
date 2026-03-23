# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Full hidden states connector: extracts hidden states during both prefill
# and decode phases.  Based on Yibo's modified ExampleHiddenStatesConnector.
#
# Prefill: extracts all prompt token hidden states at once via KV cache.
# Decode: extracts one token's hidden states per step, accumulates in CPU
#         memory, writes to disk when the request finishes.
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import safetensors
import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def extract_from_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Extract data from KV cache.
    Assume shape: (num_pages, page_size, num_heads, head_size)
    """
    padded_kv = kv_cache.flatten(0, 1)[slot_mapping]
    return padded_kv[:num_tokens]


@dataclass
class ReqMeta:
    req_id: str
    filename: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    new_req: bool

    @staticmethod
    def make_meta(
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        new_req: bool,
    ) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()
        return ReqMeta(
            req_id=req_id,
            filename=filename,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            new_req=new_req,
        )


@dataclass
class DecodeReqMeta:
    """Metadata for a single decode step extraction."""
    req_id: str
    slot: int


@dataclass
class FullHiddenStatesConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    decode_requests: list[DecodeReqMeta] = field(default_factory=list)

    def add_request(
        self,
        req_id: str,
        filename: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        new_req: bool = True,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(
                req_id, filename, token_ids, block_ids, block_size, new_req
            )
        )

    def add_decode_request(self, req_id: str, slot: int) -> None:
        self.decode_requests.append(DecodeReqMeta(req_id=req_id, slot=slot))


class FullHiddenStatesConnector(KVConnectorBase_V1):
    """Extracts hidden states during both prefill and decode.

    Prefill: extracts all prompt token hidden states at once.
    Decode: extracts one token per step, accumulates in CPU memory,
            writes to disk when the request finishes.
    """

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return False

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp"
        )
        self.cache_layers: list[str] = []
        logger.info("FullHiddenStatesConnector storage: %s", self._storage_path)

        assert vllm_config.speculative_config is not None
        spec_config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(spec_config, "eagle_aux_hidden_state_layer_ids", [])
        )

        self._request_filenames: dict[str, str] = {}
        self._active_requests: dict[str, NewRequestData] = {}
        self._req_blocks: dict[str, list[int]] = {}
        self._req_num_tokens: dict[str, int] = {}
        self._decode_buffers: dict[str, list[torch.Tensor]] = {}

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, *args, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self):
        pass

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionLayer,
        )
        layers = get_layers_from_vllm_config(
            self._vllm_config, CacheOnlyAttentionLayer, list(kv_caches.keys())
        )
        self.cache_layers = list(layers.keys())
        assert len(self.cache_layers) == 1, (
            f"Expected 1 CacheOnlyAttentionLayer, got {len(self.cache_layers)}"
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        if layer_name not in self.cache_layers:
            return

        from vllm.model_executor.models.extract_hidden_states import (
            CacheOnlyAttentionMetadata,
        )
        assert isinstance(attn_metadata, CacheOnlyAttentionMetadata)

        connector_metadata = self._get_connector_metadata()
        assert isinstance(connector_metadata, FullHiddenStatesConnectorMetadata)

        flat_kv = kv_layer.flatten(0, 1)

        # Prefill: extract all prompt tokens at once
        os.makedirs(self._storage_path, exist_ok=True)
        for request in connector_metadata.requests:
            hidden_states = extract_from_kv_cache(
                kv_layer, request.slot_mapping, request.token_ids.shape[0]
            )
            tensors = {
                "hidden_states": hidden_states.detach().cpu(),
                "token_ids": request.token_ids.detach().cpu(),
            }
            safetensors.torch.save_file(tensors, request.filename)

        # Decode: extract 1 token per request, accumulate in CPU buffer
        for dreq in connector_metadata.decode_requests:
            hs = flat_kv[dreq.slot]
            self._decode_buffers.setdefault(dreq.req_id, []).append(
                hs.detach().cpu()
            )

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks",
        num_external_tokens: int
    ):
        assert num_external_tokens == 0

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = FullHiddenStatesConnectorMetadata()

        # New requests (prefill)
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            filename = os.path.join(
                self._storage_path, f"{new_req.req_id}.safetensors")
            meta.add_request(
                new_req.req_id,
                filename=filename,
                token_ids=token_ids,
                block_ids=new_req.block_ids[0],
                block_size=self._block_size,
            )
            self._request_filenames[new_req.req_id] = filename
            self._active_requests[new_req.req_id] = new_req
            self._req_blocks[new_req.req_id] = list(new_req.block_ids[0])
            self._req_num_tokens[new_req.req_id] = len(token_ids)

        # Cached requests (decode): extract 1 new token per step
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            if req_id not in self._active_requests:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]
            if new_block_ids is not None:
                self._req_blocks[req_id].extend(new_block_ids[0])

            # Compute slot for the latest token
            num_tokens = self._req_num_tokens[req_id]
            block_idx = num_tokens // self._block_size
            offset = num_tokens % self._block_size
            all_blocks = self._req_blocks[req_id]
            slot = all_blocks[block_idx] * self._block_size + offset

            meta.add_decode_request(req_id=req_id, slot=slot)
            self._req_num_tokens[req_id] = num_tokens + 1

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        req_filename = self._request_filenames.pop(req_id, None)
        _ = self._active_requests.pop(req_id, None)
        _ = self._req_blocks.pop(req_id, None)
        _ = self._req_num_tokens.pop(req_id, None)

        # Save decode hidden states alongside prefill
        decode_hs_list = self._decode_buffers.pop(req_id, [])
        decode_filename = None
        if decode_hs_list:
            decode_hs = torch.stack(decode_hs_list, dim=0)
            decode_filename = os.path.join(
                self._storage_path, f"{req_id}_decode.safetensors"
            )
            safetensors.torch.save_file(
                {"hidden_states": decode_hs}, decode_filename
            )

        return False, {
            "hidden_states_path": req_filename,
            "decode_hidden_states_path": decode_filename,
        }

    @classmethod
    def get_required_kvcache_layout(
            cls, vllm_config: "VllmConfig") -> str | None:
        return "NHD"
