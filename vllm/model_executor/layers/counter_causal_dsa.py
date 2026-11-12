# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental full Counter scoring against completed DeepSeek MLA KV.

This is a slow, eager correctness implementation, not a fused DSA kernel.
The original sparse attention, allocator and top-k kernels are unchanged.
Only the score source and the explicitly configured retention quotas change.
"""

import logging
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)
_input_batch: ContextVar[Any] = ContextVar("counter_dsa_input_batch", default=None)
_runtime: ContextVar[Any] = ContextVar("counter_dsa_runtime", default=None)
_dummy_run: ContextVar[bool] = ContextVar("counter_dsa_dummy_run", default=False)


@dataclass(frozen=True)
class CounterDSAConfig:
    """Selection quotas count visible positions, not allocated KV slots."""

    min_gap: int = 1
    frozen_tokens: int = 4
    local_window: int = 128
    history_tokens: int = 380
    query_chunk_size: int = 128
    lm_head_chunk_size: int = 128
    warmup: str = "native"

    def __post_init__(self):
        for key in (
            "min_gap",
            "frozen_tokens",
            "local_window",
            "history_tokens",
            "query_chunk_size",
            "lm_head_chunk_size",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"counter_dsa.{key} must be a non-negative int")
        if self.local_window < self.min_gap + 1:
            raise ValueError("local_window must cover the unscorable suffix")
        if not self.history_tokens or not self.query_chunk_size:
            raise ValueError("history_tokens and query_chunk_size must be positive")
        if not self.lm_head_chunk_size:
            raise ValueError("lm_head_chunk_size must be positive")
        if self.topk_tokens not in (512, 1024, 2048):
            raise ValueError("Total must match a native persistent-top-k budget")
        if self.warmup not in ("native", "error"):
            raise ValueError("warmup must be 'native' or 'error'")

    @property
    def topk_tokens(self):
        return self.frozen_tokens + self.local_window + self.history_tokens

    @classmethod
    def from_vllm(cls, vllm_config):
        additional = vllm_config.additional_config
        value = additional.get("counter_dsa") if isinstance(additional, dict) else None
        if value is None or value is False:
            return None
        if value is True:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("counter_dsa must be a JSON object or boolean")
        return cls(**value)


@contextmanager
def counter_input_batch(batch):
    """Expose actual request order without changing the worker's scheduling."""
    token = _input_batch.set(batch)
    try:
        yield
    finally:
        _input_batch.reset(token)


def active_counter_runtime():
    return _runtime.get()


@contextmanager
def counter_dummy_run():
    """Keep native dummy/profile execution distinct from a real request batch."""
    token = _dummy_run.set(True)
    try:
        yield
    finally:
        _dummy_run.reset(token)


def counter_dsa_enabled(vllm_config):
    additional = vllm_config.additional_config
    value = additional.get("counter_dsa") if isinstance(additional, dict) else None
    return value is not None and value is not False


def gather_mla_prefix(cache, block_table, request_index, length):
    """Gather a request's logical prefix using *this layer's* block table."""
    if cache.ndim != 3 or cache.shape[1] <= 0:
        raise RuntimeError("Counter requires a [blocks, block_size, width] MLA cache")
    block_size = cache.shape[1]
    positions = torch.arange(length, device=cache.device, dtype=torch.long)
    block_offsets = positions // block_size
    if length and int(block_offsets[-1]) >= block_table.shape[1]:
        raise RuntimeError("Counter prefix extends beyond its block table")
    blocks = block_table[request_index, block_offsets].long()
    if bool(((blocks < 0) | (blocks >= cache.shape[0])).any()):
        raise RuntimeError("Counter found an invalid historical KV block")
    return cache[blocks, positions % block_size]


def unpack_mla_prefix(rows, kv_rank, rope_dim, dtype):
    """Decode BF16 MLA or the upstream V3.2 656-byte fp8_ds_mla format."""
    if rows.dtype in (torch.bfloat16, torch.float16, torch.float32):
        if rows.shape[-1] != kv_rank + rope_dim:
            raise RuntimeError("Unexpected non-quantized MLA cache width")
        return rows[:, :kv_rank].to(dtype), rows[:, kv_rank:].to(dtype)
    if (
        rows.dtype == torch.uint8
        and rows.shape[-1] == 656
        and kv_rank == 512
        and rope_dim == 64
    ):
        nope = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = rows[:, 512:528].contiguous().view(torch.float32)
        nope = (nope.reshape(-1, 4, 128) * scales.unsqueeze(-1)).reshape(-1, 512)
        rope = rows[:, 528:656].contiguous().view(torch.bfloat16)
        return nope.to(dtype), rope.to(dtype)
    raise RuntimeError("Unsupported MLA cache layout; no approximate fallback")


def reverse_attention(q, k, v, positions, min_gap, chunk_size, scale):
    """Full counter-causal attention, tiled only over independent Q rows."""
    pieces = []
    for start in range(0, len(positions), chunk_size):
        end = min(start + chunk_size, len(positions))
        mask = positions[None, :] > positions[start:end, None] + min_gap
        pieces.append(
            F.scaled_dot_product_attention(
                q[:, start:end],
                k,
                v,
                attn_mask=mask[None],
                dropout_p=0.0,
                scale=scale,
            )
        )
    return torch.cat(pieces, dim=1)


def fill_selection_scores(row, start, end, old_scores, config, native=False):
    """Fill one causal prefix and pin sinks/local without changing top-k.

    The first/last ranges form a union for short contexts; they are never
    counted twice. When the prefix exceeds the budget, exactly history_tokens
    unprotected positions compete after the protected quotas are reserved.
    """
    length = end - start
    if length < 0 or start < 0 or end > row.numel():
        raise RuntimeError("Invalid Counter score row bounds")
    if length <= config.topk_tokens:
        row[start:end] = 0.0  # Every visible position fits; no score is needed.
    elif not native:
        if old_scores is None or len(old_scores) < length - config.local_window:
            raise RuntimeError("Uncompleted history outside the protected local window")
        row[start:end] = float("-inf")
        count = min(length, len(old_scores))
        row[start : start + count] = old_scores[:count]
    row[start : min(end, start + config.frozen_tokens)] = float("inf")
    row[max(start, end - config.local_window) : end] = float("inf")


def full_counter_scores(owner, token_ids, layer_kv, config):
    """Replay every decoder layer with cached K/V and a reverse mask.

    Args:
        owner: DeepseekV2ForCausalLM, including its real loaded modules.
        token_ids: Actual token IDs corresponding to the completed prefix.
        layer_kv: Callable(layer_index) returning normalized latent K/V and
            already-RoPE-rotated cached K positional components.
        config: Counter configuration. This never calls the main attention
            module, indexer, or KV update functions.
    """
    hidden = owner.model.embed_input_ids(token_ids)
    positions = torch.arange(len(token_ids), device=token_ids.device)
    for index, layer in enumerate(owner.model.layers):
        wrapper = layer.self_attn.mla_attn
        normed = layer.input_layernorm(hidden)
        if wrapper.q_lora_rank is not None:
            # The fused projection also emits a NEW KV component. It is ignored:
            # Counter reuses the cached KV instead of replacing it.
            projected = wrapper.fused_qkv_a_proj(normed)[0]
            q_c = wrapper.q_a_layernorm(projected[:, : wrapper.q_lora_rank])
            q = wrapper.q_b_proj(q_c)[0]
        else:
            q = wrapper.q_proj(normed)[0]
        q = q.reshape(-1, wrapper.num_heads, wrapper.qk_head_dim)
        if wrapper.rotary_emb is not None:
            q_pe = q[..., wrapper.qk_nope_head_dim :].clone()
            dummy_k = torch.zeros(
                (len(positions), 1, wrapper.qk_rope_head_dim),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            q_pe, _ = wrapper.rotary_emb(positions, q_pe, dummy_k)
            q = torch.cat((q[..., : wrapper.qk_nope_head_dim], q_pe), dim=-1)
        latent, k_pe = layer_kv(index)
        kv = wrapper.kv_b_proj(latent.to(hidden.dtype))[0].reshape(
            -1,
            wrapper.num_heads,
            wrapper.qk_nope_head_dim + wrapper.v_head_dim,
        )
        k_nope, v = kv.split(
            (wrapper.qk_nope_head_dim, wrapper.v_head_dim),
            dim=-1,
        )
        # Cached K positional components already carry RoPE. Do not rotate again.
        k = torch.cat((k_nope, k_pe[:, None].expand(-1, wrapper.num_heads, -1)), -1)
        out = reverse_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            positions,
            config.min_gap,
            config.query_chunk_size,
            wrapper.mla_attn.scale,
        )
        out = out.transpose(0, 1).reshape(len(token_ids), -1)
        hidden = hidden + wrapper.o_proj(out)[0]
        normed = layer.post_attention_layernorm(hidden)
        # MoE's token-local reference replay must not exceed the serving
        # kernel's configured per-call token capacity. This is private replay
        # tiling; it does not change the scheduler's batching or attention.
        hidden = hidden + torch.cat(
            [
                layer.mlp(normed[start : start + config.query_chunk_size])
                for start in range(0, len(token_ids), config.query_chunk_size)
            ]
        )
    hidden = owner.model.norm(hidden)
    return -_true_token_logits(owner, hidden, token_ids, config.lm_head_chunk_size)


def _true_token_logits(owner, hidden, token_ids, chunk_size):
    """Extract raw target-token logits without softmax or changing generation."""
    head = owner.lm_head
    shard = head.shard_indices
    if shard.num_added_elements:
        raise RuntimeError("Counter reference scorer does not support added vocabulary")
    result = []
    for start in range(0, len(token_ids), chunk_size):
        end = min(start + chunk_size, len(token_ids))
        logits = head.quant_method.apply(head, hidden[start:end])
        target = token_ids[start:end]
        valid = (target >= shard.org_vocab_start_index) & (
            target < shard.org_vocab_end_index
        )
        local = (target - shard.org_vocab_start_index).clamp(0, logits.shape[-1] - 1)
        values = logits.float().gather(1, local[:, None]).squeeze(1)
        values = torch.where(valid, values, 0.0)
        if owner._counter_tp_size > 1:
            from vllm.distributed import tensor_model_parallel_all_reduce

            values = tensor_model_parallel_all_reduce(values)
        result.append(values)
    values = torch.cat(result)
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError("Counter produced non-finite real-token logits")
    return values


@dataclass
class _Request:
    req_id: str
    index: int
    old_len: int
    query_start: int
    query_end: int
    token_ids: torch.Tensor


class CounterDSARuntime:
    """Eager bridge between request bookkeeping, completed MLA KV and scores."""

    def __init__(self, owner, vllm_config, config):
        self.owner_ref = weakref.ref(owner)
        self.vllm_config = vllm_config
        self.config = config
        self.histories = {}
        self.rows = []
        self.metadata = None
        self.score_cache = {}
        self.indexer_prefix = None
        self.stats = {
            "indexer_calls": 0,
            "full_counter_passes": 0,
            "counter_rows": 0,
            "native_warmup_rows": 0,
            "all_fit_rows": 0,
        }
        self.warned_warmup = False
        self.validate_support(vllm_config, config)
        first = owner.model.layers[0].self_attn.mla_attn.mla_attn
        if first.get_attn_backend().get_name() != "FLASHMLA_SPARSE":
            raise ValueError(
                "Counter reference bridge requires FLASHMLA_SPARSE metadata"
            )

    @staticmethod
    def validate_support(vllm_config, config=None):
        """Reject unimplemented execution modes instead of changing their settings."""
        parallel = vllm_config.parallel_config
        checks = {
            "pipeline parallelism": parallel.pipeline_parallel_size != 1,
            "context parallelism": (
                parallel.prefill_context_parallel_size != 1
                or parallel.decode_context_parallel_size != 1
            ),
            "data/expert parallelism": (
                parallel.data_parallel_size != 1 or parallel.enable_expert_parallel
            ),
            "sequence-parallel MoE": parallel.use_sequence_parallel_moe,
            "microbatching/DBO": parallel.use_ubatching,
            "speculative decoding": vllm_config.speculative_config is not None,
            "LoRA": vllm_config.lora_config is not None,
            "KV transfer/offload": (
                vllm_config.kv_transfer_config is not None
                or vllm_config.offload_config.uva.cpu_offload_gb > 0
                or vllm_config.offload_config.prefetch.offload_group_size > 0
            ),
            "CUDA graph/compiled execution (use --enforce-eager)": (
                not vllm_config.model_config.enforce_eager
            ),
            "non-BF16 model execution": (
                vllm_config.model_config.dtype != torch.bfloat16
            ),
        }
        bad = [name for name, enabled in checks.items() if enabled]
        if bad:
            raise ValueError(
                "Counter DSA reference mode does not support: " + ", ".join(bad)
            )
        if config is not None and (
            config.query_chunk_size
            > vllm_config.scheduler_config.max_num_batched_tokens
        ):
            raise ValueError(
                "Counter query_chunk_size exceeds the existing MoE token capacity; "
                "lower the private replay tile rather than changing batch scheduling"
            )

    @contextmanager
    def step(self, input_ids, positions):
        from vllm.forward_context import get_forward_context

        metadata = get_forward_context().attn_metadata
        if _dummy_run.get() or not isinstance(metadata, dict):
            yield
            return
        batch = _input_batch.get()
        if batch is None or input_ids is None or positions.ndim != 1:
            raise RuntimeError(
                "Counter requires text token IDs and the GPU runner batch"
            )
        owner = self.owner_ref()
        first = owner.model.layers[0].self_attn.mla_attn
        main_meta = metadata[first.mla_attn.layer_name]
        index_meta = metadata[first.indexer.k_cache.prefix]
        starts = main_meta.query_start_loc.cpu().tolist()
        lengths = index_meta.seq_lens.cpu().tolist()
        self.rows = []
        self.metadata = metadata
        self.score_cache = {}
        live = set(batch.req_ids)
        self.histories = {k: v for k, v in self.histories.items() if k in live}
        for i in range(main_meta.num_reqs):
            start, end = starts[i : i + 2]
            old_len = int(lengths[i]) - (end - start)
            if old_len < 0:
                raise RuntimeError("Negative completed-prefix length")
            req_id = batch.req_ids[i]
            known = int(batch.num_tokens_no_spec[i])
            needed = max(known, int(lengths[i]))
            ids = self.histories.get(req_id)
            if ids is None:
                ids = torch.full(
                    (needed,), -1, device=input_ids.device, dtype=torch.long
                )
                # Includes all prompt IDs, including prefixes reused from another
                # request. Physical cache slots are never used as token identity.
                ids[:known] = batch.token_ids_cpu_tensor[i, :known].to(input_ids.device)
            elif len(ids) < needed:
                grown = torch.full(
                    (max(needed, len(ids) * 2),),
                    -1,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                grown[: len(ids)] = ids
                ids = grown
            expected = torch.arange(old_len, int(lengths[i]), device=positions.device)
            if not torch.equal(positions[start:end], expected):
                raise RuntimeError("Counter request/token/absolute-position mismatch")
            # Actual consumed device tokens override possibly asynchronous CPU IDs.
            ids[old_len : int(lengths[i])] = input_ids[start:end]
            self.histories[req_id] = ids
            request = _Request(req_id, i, old_len, start, end, ids)
            self.rows.extend((request, old_len + j) for j in range(end - start))
        token = _runtime.set(self)
        try:
            yield
        finally:
            _runtime.reset(token)
            self.metadata = None
            self.rows = []
            self.score_cache = {}

    def start_indexer(self, prefix):
        self.indexer_prefix = prefix
        self.stats["indexer_calls"] += 1
        # Reuse only within this original indexer invocation (prefill chunks).
        # Another layer/indexer invocation recomputes the full Counter score.
        self.score_cache = {}

    def _needs_native(self, row):
        request, position = self.rows[row]
        length = position + 1
        return length > self.config.topk_tokens and (
            length - self.config.local_window > request.old_len
        )

    def needs_native_prefill(self, chunk):
        return any(
            self._needs_native(i) for i in range(chunk.token_start, chunk.token_end)
        )

    def needs_native_decode(self, num_tokens):
        return any(self._needs_native(i) for i in range(num_tokens))

    @torch.inference_mode()
    def _scores(self, request):
        if request.req_id in self.score_cache:
            return self.score_cache[request.req_id]
        from vllm.forward_context import set_forward_context

        owner = self.owner_ref()
        token_ids = request.token_ids[: request.old_len]
        if bool(((token_ids < 0) | (token_ids >= owner.config.vocab_size)).any()):
            raise RuntimeError("Missing or invalid historical token ID")

        def layer_kv(index):
            wrapper = owner.model.layers[index].self_attn.mla_attn
            attn = wrapper.mla_attn
            meta = self.metadata[attn.layer_name]
            rows = gather_mla_prefix(
                attn.kv_cache,
                meta.block_table,
                request.index,
                request.old_len,
            )
            return unpack_mla_prefix(
                rows,
                wrapper.kv_lora_rank,
                wrapper.qk_rope_head_dim,
                self.vllm_config.model_config.dtype,
            )

        # MoE/linear kernels see the replay's token count, not the outer decode
        # count. No main attention module or cache update runs in this context.
        with set_forward_context(
            attn_metadata=None,
            vllm_config=self.vllm_config,
            num_tokens=len(token_ids),
        ):
            scores = full_counter_scores(owner, token_ids, layer_kv, self.config)
        self.stats["full_counter_passes"] += 1
        # Same safety priority as the original Counter scorer. These entries
        # have no allowed future context under the reverse mask.
        scores[-(self.config.min_gap + 1) :] = float("inf")
        self.score_cache[request.req_id] = scores
        return scores

    def _apply_row(self, logits, row, flat_row, start, end):
        request, position = self.rows[flat_row]
        if end - start != position + 1:
            raise RuntimeError("Indexer score columns do not match the causal prefix")
        native = self._needs_native(flat_row)
        if native:
            if self.config.warmup == "error":
                raise RuntimeError(
                    "No completed snapshot for this prefill; warmup=error"
                )
            if not self.warned_warmup:
                logger.warning(
                    "Counter DSA: bootstrap rows use native DSA scores because "
                    "unfinished history extends outside local%d. Counts are in "
                    "model.counter_runtime.stats; this is not full-Counter prefill.",
                    self.config.local_window,
                )
                self.warned_warmup = True
            self.stats["native_warmup_rows"] += 1
            scores = None
        elif end - start <= self.config.topk_tokens:
            self.stats["all_fit_rows"] += 1
            scores = None
        else:
            self.stats["counter_rows"] += 1
            scores = self._scores(request)
        fill_selection_scores(logits[row], start, end, scores, self.config, native)

    def apply_prefill(self, logits, chunk):
        starts = chunk.cu_seqlen_ks.cpu().tolist()
        ends = chunk.cu_seqlen_ke.cpu().tolist()
        for row, (start, end) in enumerate(zip(starts, ends)):
            self._apply_row(logits, row, row + chunk.token_start, start, end)
        return logits

    def apply_decode(self, logits, lengths):
        lens = lengths.flatten().cpu().tolist()
        if len(lens) != logits.shape[0]:
            raise RuntimeError("Unsupported padded/speculative decode row layout")
        for row, end in enumerate(lens):
            self._apply_row(logits, row, row, 0, end)
        return logits
