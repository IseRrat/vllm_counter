# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU mathematical/reference tests, independent of vLLM GPU extensions.

These tests do not claim that a DS-V3.2 checkpoint or native CUDA kernels ran.
"""

import ast
import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "_counter_dsa_unit_module",
    ROOT / "vllm/model_executor/layers/counter_causal_dsa.py",
)
C = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = C
SPEC.loader.exec_module(C)


class Box(SimpleNamespace):
    pass


def vllm_config():
    return Box(
        additional_config={},
        parallel_config=Box(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            data_parallel_size=1,
            enable_expert_parallel=False,
            use_sequence_parallel_moe=False,
            use_ubatching=False,
        ),
        model_config=Box(enforce_eager=True, dtype=torch.bfloat16),
        scheduler_config=Box(max_num_batched_tokens=1024),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        offload_config=Box(
            uva=Box(cpu_offload_gb=0),
            prefetch=Box(offload_group_size=0),
        ),
    )


def test_default_is_opt_in_and_budget_is_512():
    cfg = vllm_config()
    assert C.CounterDSAConfig.from_vllm(cfg) is None
    assert not C.counter_dsa_enabled(cfg)
    cfg.additional_config = {"counter_dsa": {}}
    result = C.CounterDSAConfig.from_vllm(cfg)
    assert C.counter_dsa_enabled(cfg)
    assert result.topk_tokens == 512
    assert (result.min_gap, result.frozen_tokens, result.local_window) == (1, 4, 128)
    assert result.history_tokens == 380


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_gap": -1},
        {"min_gap": True},
        {"local_window": 1},
        {"history_tokens": 0},
        {"query_chunk_size": 0},
        {"lm_head_chunk_size": 0},
        {"warmup": "silent"},
        {"history_tokens": 381},
    ],
)
def test_invalid_counter_configuration(kwargs):
    with pytest.raises(ValueError):
        C.CounterDSAConfig(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("prefill_context_parallel_size", 2),
        ("enable_expert_parallel", True),
        ("use_sequence_parallel_moe", True),
        ("use_ubatching", True),
    ],
)
def test_unsupported_parallel_modes_are_rejected_not_reconfigured(field, value):
    cfg = vllm_config()
    setattr(cfg.parallel_config, field, value)
    with pytest.raises(ValueError):
        C.CounterDSARuntime.validate_support(cfg)
    assert getattr(cfg.parallel_config, field) == value


def test_graphs_and_speculation_require_explicit_supported_execution():
    cfg = vllm_config()
    cfg.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="enforce-eager"):
        C.CounterDSARuntime.validate_support(cfg)
    cfg.model_config.enforce_eager = True
    cfg.speculative_config = Box()
    with pytest.raises(ValueError, match="speculative"):
        C.CounterDSARuntime.validate_support(cfg)


@pytest.mark.parametrize(
    "length", [1, 3, 4, 127, 128, 131, 132, 511, 512, 513, 640, 2049]
)
def test_sink_local_and_history_budget(length):
    cfg = C.CounterDSAConfig()
    old = torch.randperm(max(length - 1, 1), generator=torch.Generator().manual_seed(7))
    old = old.float()
    row = torch.full((length + 19,), -999999.0)
    start, end = 7, 7 + length
    C.fill_selection_scores(row, start, end, old, cfg)
    k = min(cfg.topk_tokens, length)
    chosen = row[start:end].topk(k).indices.tolist()
    protected = set(range(min(4, length))) | set(range(max(0, length - 128), length))
    assert protected <= set(chosen)
    assert len(chosen) == len(set(chosen)) == k
    assert (row[:start] == -999999).all() and (row[end:] == -999999).all()
    if length > cfg.topk_tokens:
        middle = list(range(4, length - 128))
        expected = sorted(middle, key=lambda i: old[i].item(), reverse=True)[:380]
        assert set(chosen) - protected == set(expected)


def test_unscored_history_cannot_silently_be_selected():
    cfg = C.CounterDSAConfig()
    with pytest.raises(RuntimeError, match="Uncompleted history"):
        C.fill_selection_scores(torch.zeros(800), 0, 800, torch.zeros(600), cfg)


def test_native_bootstrap_keeps_scores_and_enforces_quotas():
    cfg = C.CounterDSAConfig()
    native = torch.arange(800).float()
    C.fill_selection_scores(native, 0, 800, None, cfg, native=True)
    assert torch.equal(native[4:672], torch.arange(4, 672).float())
    assert torch.isposinf(native[:4]).all() and torch.isposinf(native[-128:]).all()


@pytest.mark.parametrize("gap", [0, 1, 4])
@pytest.mark.parametrize("chunk", [1, 3, 16])
def test_reverse_attention_against_independent_dense_reference(gap, chunk):
    g = torch.Generator().manual_seed(9)
    q = torch.randn(2, 5, 4, generator=g, dtype=torch.float64)
    k = torch.randn(2, 5, 4, generator=g, dtype=torch.float64)
    v = torch.randn(2, 5, 3, generator=g, dtype=torch.float64)
    positions = torch.tensor([0, 2, 5, 9, 12])
    actual = C.reverse_attention(q, k, v, positions, gap, chunk, 0.5)
    reference = torch.zeros(2, 5, 3, dtype=torch.float64)
    for i in range(5):
        valid = [j for j in range(5) if positions[j] > positions[i] + gap]
        if valid:
            for head in range(2):
                scores = torch.stack(
                    [torch.dot(q[head, i], k[head, j]) * 0.5 for j in valid]
                )
                reference[head, i] = torch.softmax(scores, 0) @ v[head, valid]
    torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_paged_gather_has_no_cross_request_or_padding_reads():
    cache = torch.full((8, 4, 3), float("nan"))
    a = torch.arange(21).reshape(7, 3).float()
    b = torch.arange(15).reshape(5, 3).float() + 1000
    table = torch.tensor([[5, 1, -1], [7, 3, -1]])
    for values, blocks in [(a, [5, 1]), (b, [7, 3])]:
        for i, value in enumerate(values):
            cache[blocks[i // 4], i % 4] = value
    torch.testing.assert_close(C.gather_mla_prefix(cache, table, 0, 7), a)
    torch.testing.assert_close(C.gather_mla_prefix(cache, table, 1, 5), b)
    with pytest.raises(RuntimeError, match="invalid historical"):
        C.gather_mla_prefix(cache, table, 0, 9)


def test_unpack_bf16_and_upstream_fp8_mla_656_bytes():
    bf16 = torch.arange(24).reshape(3, 8).bfloat16()
    latent, rope = C.unpack_mla_prefix(bf16, 6, 2, torch.float32)
    torch.testing.assert_close(latent, bf16[:, :6].float())
    torch.testing.assert_close(rope, bf16[:, 6:].float())
    values = (torch.arange(1024).reshape(2, 512) % 13 - 6).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1, 2, 4, 8], [0.5, 0.25, 0.125, 2]], dtype=torch.float32)
    rope = torch.arange(128).reshape(2, 64).bfloat16()
    packed = torch.cat(
        (
            values.view(torch.uint8),
            scales.view(torch.uint8),
            rope.view(torch.uint8),
        ),
        -1,
    )
    latent, actual_rope = C.unpack_mla_prefix(packed, 512, 64, torch.float32)
    expected = values.float() * scales.repeat_interleave(128, -1)
    torch.testing.assert_close(latent, expected)
    torch.testing.assert_close(actual_rope, rope.float())


class TupleLinear(torch.nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class ToyRoPE:
    def __call__(self, positions, q, k):
        angle = positions[:, None, None].to(q.dtype) / 10

        def rotate(x):
            return torch.stack(
                (
                    x[..., 0] * angle[..., 0].cos() - x[..., 1] * angle[..., 0].sin(),
                    x[..., 0] * angle[..., 0].sin() + x[..., 1] * angle[..., 0].cos(),
                ),
                -1,
            )

        return rotate(q), rotate(k)


def toy_model():
    torch.manual_seed(25)
    hidden, vocab = 8, 19
    embedding = torch.nn.Embedding(vocab, hidden, dtype=torch.float64)
    layers, kvs = [], []
    for _ in range(3):
        wrapper = Box(
            num_heads=2,
            qk_head_dim=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            q_lora_rank=3,
            kv_lora_rank=3,
            v_head_dim=2,
            fused_qkv_a_proj=TupleLinear(hidden, 8, bias=False, dtype=torch.float64),
            q_a_layernorm=torch.nn.LayerNorm(3, dtype=torch.float64),
            q_b_proj=TupleLinear(3, 8, bias=False, dtype=torch.float64),
            kv_b_proj=TupleLinear(3, 8, bias=False, dtype=torch.float64),
            o_proj=TupleLinear(4, hidden, bias=False, dtype=torch.float64),
            rotary_emb=ToyRoPE(),
            mla_attn=Box(scale=0.5),
        )
        layers.append(
            Box(
                self_attn=Box(mla_attn=wrapper),
                input_layernorm=torch.nn.LayerNorm(hidden, dtype=torch.float64),
                post_attention_layernorm=torch.nn.LayerNorm(
                    hidden, dtype=torch.float64
                ),
                mlp=torch.nn.Sequential(
                    torch.nn.Linear(hidden, 11, dtype=torch.float64),
                    torch.nn.SiLU(),
                    torch.nn.Linear(11, hidden, dtype=torch.float64),
                ),
            )
        )
        kvs.append(
            (
                torch.randn(9, 3, dtype=torch.float64),
                torch.randn(9, 2, dtype=torch.float64),
            )
        )
    head = Box(
        weight=torch.randn(vocab, hidden, dtype=torch.float64),
        shard_indices=Box(
            org_vocab_start_index=0, org_vocab_end_index=vocab, num_added_elements=0
        ),
        quant_method=Box(apply=lambda head, h: F.linear(h, head.weight)),
    )
    return Box(
        model=Box(
            layers=layers,
            embed_input_ids=embedding,
            norm=torch.nn.LayerNorm(hidden, dtype=torch.float64),
        ),
        lm_head=head,
        _counter_tp_size=1,
    ), kvs


@pytest.mark.parametrize("gap", [0, 1, 4])
def test_full_three_layer_counter_matches_independent_dense_replay(gap):
    owner, kvs = toy_model()
    ids = torch.tensor([2, 7, 5, 11, 3, 1, 9, 4, 6])
    cfg = C.CounterDSAConfig(min_gap=gap, query_chunk_size=2, lm_head_chunk_size=3)
    actual = C.full_counter_scores(owner, ids, kvs.__getitem__, cfg)
    hidden = owner.model.embed_input_ids(ids)
    pos = torch.arange(len(ids))
    for layer, (latent, cached_rope) in zip(owner.model.layers, kvs):
        w = layer.self_attn.mla_attn
        normed = layer.input_layernorm(hidden)
        q_c = w.q_a_layernorm(w.fused_qkv_a_proj(normed)[0][:, :3])
        q = w.q_b_proj(q_c)[0].reshape(9, 2, 4)
        rotated, _ = w.rotary_emb(pos, q[..., 2:], torch.zeros(9, 1, 2, dtype=q.dtype))
        q = torch.cat((q[..., :2], rotated), -1)
        projected = w.kv_b_proj(latent)[0].reshape(9, 2, 4)
        k = torch.cat((projected[..., :2], cached_rope[:, None].expand(-1, 2, -1)), -1)
        values = projected[..., 2:]
        outputs = torch.zeros(9, 2, 2, dtype=torch.float64)
        for i in range(9):
            valid = list(range(i + gap + 1, 9))
            for head in range(2):
                if valid:
                    logits = (k[valid, head] @ q[i, head]) / math.sqrt(4)
                    outputs[i, head] = logits.softmax(0) @ values[valid, head]
        hidden = hidden + w.o_proj(outputs.reshape(9, 4))[0]
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    logits = F.linear(owner.model.norm(hidden), owner.lm_head.weight)
    # The indexer interface uses FP32 scores, even when the independent oracle
    # computes the replay in FP64. Compare at the interface's precision.
    reference = -logits[torch.arange(9), ids].float()
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    # Removing any layer changes scores: this is not the single-layer Fast path.
    one_layer = owner.model.layers[:1]
    owner.model.layers = one_layer
    assert not torch.allclose(
        actual, C.full_counter_scores(owner, ids, kvs.__getitem__, cfg)
    )


def runtime_fixture(monkeypatch, lengths, query_lens, req_ids):
    cfg = vllm_config()
    main_name, index_name = "main", "index"
    owner = Box(
        model=Box(
            layers=[
                Box(
                    self_attn=Box(
                        mla_attn=Box(
                            mla_attn=Box(
                                layer_name=main_name,
                                get_attn_backend=lambda: Box(
                                    get_name=lambda: "FLASHMLA_SPARSE"
                                ),
                            ),
                            indexer=Box(k_cache=Box(prefix=index_name)),
                        )
                    )
                )
            ]
        )
    )
    runtime = C.CounterDSARuntime(owner, cfg, C.CounterDSAConfig())
    starts = [0]
    positions = []
    for length, query in zip(lengths, query_lens):
        starts.append(starts[-1] + query)
        positions.extend(range(length - query, length))
    main = Box(num_reqs=len(req_ids), query_start_loc=torch.tensor(starts))
    metadata = {main_name: main, index_name: Box(seq_lens=torch.tensor(lengths))}
    context = Box(attn_metadata=metadata)
    monkeypatch.setitem(
        sys.modules, "vllm.forward_context", Box(get_forward_context=lambda: context)
    )
    batch = Box(
        req_ids=req_ids,
        num_tokens_no_spec=lengths,
        token_ids_cpu_tensor=torch.arange(len(req_ids) * max(lengths)).reshape(
            len(req_ids), -1
        )
        % 17,
    )
    input_ids = torch.arange(starts[-1]) % 17
    return owner, runtime, batch, input_ids, torch.tensor(positions)


def test_runtime_uses_consumed_tokens_and_restores_context(monkeypatch):
    owner, runtime, batch, input_ids, positions = runtime_fixture(
        monkeypatch,
        [640, 700],
        [1, 1],
        ["a", "b"],
    )
    input_ids[:] = torch.tensor([12, 14])
    with C.counter_input_batch(batch), runtime.step(input_ids, positions):
        assert C.active_counter_runtime() is runtime
        assert runtime.rows[0][0].old_len == 639
        assert runtime.rows[1][0].old_len == 699
        assert runtime.histories["a"][639] == 12
        assert runtime.histories["b"][699] == 14
        assert not runtime.needs_native_decode(2)
    assert C.active_counter_runtime() is None
    assert runtime.metadata is None and not runtime.rows
    assert owner is not None  # Keep weak owner alive through the context.


def test_dummy_warmup_with_real_metadata_does_not_require_live_requests(monkeypatch):
    owner, runtime, _, input_ids, positions = runtime_fixture(
        monkeypatch,
        [640],
        [1],
        ["a"],
    )
    with C.counter_dummy_run(), runtime.step(input_ids, positions):
        assert C.active_counter_runtime() is None
        assert not runtime.histories
    # A real request without the worker's binding must not silently use DSA.
    with (
        pytest.raises(RuntimeError, match="GPU runner batch"),
        runtime.step(input_ids, positions),
    ):
        pass
    assert owner is not None


def test_mixed_prefill_offsets_do_not_leak_future_or_other_requests(monkeypatch):
    owner, runtime, batch, ids, positions = runtime_fixture(
        monkeypatch,
        [640, 640],
        [1, 640],
        ["completed", "new"],
    )
    calls = []

    def scores(request):
        calls.append(request.req_id)
        assert request.old_len == 639
        # The current input is NOT part of the completed-prefix score.
        return torch.arange(request.old_len).float().neg()

    monkeypatch.setattr(runtime, "_scores", scores)
    starts = torch.tensor([0] + [640] * 640)
    ends = torch.tensor([640] + [640 + i for i in range(1, 641)])
    chunk = Box(token_start=0, token_end=641, cu_seqlen_ks=starts, cu_seqlen_ke=ends)
    with C.counter_input_batch(batch), runtime.step(ids, positions):
        assert runtime.needs_native_prefill(chunk)
        native = torch.rand(641, 1280)
        native_before = native.clone()
        runtime.apply_prefill(native, chunk)
        assert calls == ["completed"]
        assert torch.equal(native[0, 4:512], -torch.arange(4, 512).float())
        # New request row 512 sees only its first 512 tokens: all fit.
        assert (native[512, 644:1024] == 0).all()
        # Row 513 is a native bootstrap row, excluding its pinned ranges.
        assert torch.equal(native[513, 644:1025], native_before[513, 644:1025])
        assert runtime.stats["counter_rows"] == 1
        assert runtime.stats["native_warmup_rows"] == 128
        # Poisoning future columns cannot alter the causal candidate range.
        assert torch.equal(native[513, 1153:], native_before[513, 1153:])
    assert owner is not None


def test_strict_warmup_mode_does_not_silently_fall_back(monkeypatch):
    owner, runtime, batch, ids, positions = runtime_fixture(
        monkeypatch,
        [640],
        [640],
        ["new"],
    )
    runtime.config = C.CounterDSAConfig(warmup="error")
    with (
        C.counter_input_batch(batch),
        runtime.step(ids, positions),
        pytest.raises(RuntimeError, match="warmup=error"),
    ):
        runtime._apply_row(torch.zeros(1, 640), 0, 639, 0, 640)
    assert owner is not None


def test_completed_snapshot_prefix_and_per_indexer_score_lifetime(monkeypatch):
    owner, runtime, batch, ids, positions = runtime_fixture(
        monkeypatch,
        [640],
        [1],
        ["a"],
    )
    owner.config = Box(vocab_size=19)
    wrapper = owner.model.layers[0].self_attn.mla_attn
    wrapper.kv_lora_rank = 3
    wrapper.qk_rope_head_dim = 2
    wrapper.mla_attn.kv_cache = torch.arange(1024 * 5).reshape(1, 1024, 5).bfloat16()
    context = sys.modules["vllm.forward_context"].get_forward_context()
    context.attn_metadata["main"].block_table = torch.tensor([[0]])
    calls = []

    @contextmanager
    def replay_context(**kwargs):
        assert kwargs["num_tokens"] == 639 and kwargs["attn_metadata"] is None
        yield

    sys.modules["vllm.forward_context"].set_forward_context = replay_context

    def score(owner, token_ids, layer_kv, config):
        assert len(token_ids) == 639  # Current device input at position639 excluded.
        latent, rope = layer_kv(0)
        assert latent.shape == (639, 3) and rope.shape == (639, 2)
        calls.append(token_ids.clone())
        return -token_ids.float()

    monkeypatch.setattr(C, "full_counter_scores", score)
    with C.counter_input_batch(batch), runtime.step(ids, positions):
        request = runtime.rows[0][0]
        runtime.start_indexer("layer0.indexer")
        first = runtime._scores(request)
        assert torch.isposinf(first[-2:]).all()
        assert runtime._scores(request) is first
        # Future prompt IDs / the current input cannot enter the completed score.
        request.token_ids[639:] = 18
        runtime.start_indexer("layer1.indexer")
        second = runtime._scores(request)
        torch.testing.assert_close(first, second)
        assert runtime.stats["indexer_calls"] == 2
        assert runtime.stats["full_counter_passes"] == 2
        assert len(calls) == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_indexer_control_flow_with_mixed_batch_and_mocked_kernels(
    monkeypatch,
    enabled,
):
    """Execute the real Python indexer body; mock ONLY native kernel boundaries."""
    owner, runtime, batch, ids, positions = runtime_fixture(
        monkeypatch,
        [640, 640],
        [1, 640],
        ["decode", "prefill"],
    )
    monkeypatch.setattr(
        runtime, "_scores", lambda req: -torch.arange(req.old_len).float()
    )
    chunk = Box(
        token_start=1,
        token_end=641,
        total_seq_lens=640,
        skip_kv_gather=False,
        block_table=torch.tensor([[0]]),
        cu_seq_lens=torch.tensor([0, 640]),
        cu_seqlen_ks=torch.zeros(640, dtype=torch.int32),
        cu_seqlen_ke=torch.arange(1, 641, dtype=torch.int32),
    )

    class Metadata(Box):
        pass

    meta = Metadata(
        slot_mapping=torch.arange(641),
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        max_seq_len=640,
        prefill=Box(chunks=[chunk]),
        decode=Box(
            decode_lens=torch.tensor([1]),
            requires_padding=False,
            seq_lens=torch.tensor([[640]], dtype=torch.int32),
            block_table=torch.tensor([[0]]),
            schedule_metadata=None,
        ),
    )
    events = []

    def choose(logits, starts, ends, output):
        output.fill_(-1)
        for r, (start, end) in enumerate(zip(starts, ends)):
            n = min(end - start, output.shape[1])
            output[r, :n] = logits[r, start:end].topk(n).indices.int()

    def prefill_topk(logits, ks, ke, output, rows, stride0, stride1, k):
        assert (rows, stride0, stride1, k) == (640, logits.stride(0), 1, 512)
        events.append("prefill_topk")
        choose(logits, ks.tolist(), ke.tolist(), output)

    def persistent(logits, lengths, output, workspace, k, max_len):
        assert k == 512 and max_len == 640
        events.append("decode_topk")
        lens = lengths.flatten().tolist()
        choose(logits, [0] * len(lens), lens, output)

    def prefill_logits(q, k, weights, ks, ke, clean_logits):
        events.append("native_prefill_score")
        return torch.arange(640).float().expand(640, -1).clone()

    def decode_logits(*args, **kwargs):
        events.append("native_decode_score")
        return torch.arange(640).float()[None]

    def gather(cache, values, scales, block_table, lengths):
        events.append("gather_index_keys")
        values.zero_()
        scales.zero_()

    def insert(*args):
        events.append("insert_index_keys")

    class Workspace:
        def get_simultaneous(self, *specs):
            return tuple(torch.empty(shape, dtype=dtype) for shape, dtype in specs)

    metadata_dict = {"index": meta}
    ns = dict(
        torch=torch,
        LayerNameType=str,
        get_forward_context=lambda: Box(attn_metadata=metadata_dict),
        current_platform=Box(
            fp8_dtype=lambda: torch.float32, is_xpu=lambda: False, is_cuda=lambda: True
        ),
        _resolve_layer_name=lambda x: x,
        DeepseekV32IndexerMetadata=Metadata,
        active_counter_runtime=lambda: C.active_counter_runtime() if enabled else None,
        current_workspace_manager=lambda: Workspace(),
        RADIX_TOPK_WORKSPACE_SIZE=1024 * 1024,
        ops=Box(
            indexer_k_quant_and_cache=insert,
            cp_gather_indexer_k_quant_cache=gather,
            top_k_per_row_prefill=prefill_topk,
        ),
        fp8_fp4_mqa_logits=prefill_logits,
        fp8_fp4_paged_mqa_logits=decode_logits,
    )
    source = ast.parse(
        (ROOT / "vllm/model_executor/layers/sparse_attn_indexer.py").read_text(
            encoding="utf-8"
        )
    )
    functions = []
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_gather_workspace_shapes",
            "kv_cache_as_quant_view",
            "sparse_attn_indexer",
        }:
            node.decorator_list = []
            functions.append(node)
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "actual_indexer", "exec"),
        ns,
    )
    monkeypatch.setattr(torch.ops._C, "persistent_topk", persistent, raising=False)
    result = torch.full((641, 512), -1, dtype=torch.int32)
    with C.counter_input_batch(batch), runtime.step(ids, positions):
        ns["sparse_attn_indexer"](
            torch.zeros(641, 8),
            "index",
            torch.zeros(16, 64, 6),
            torch.zeros(641, 1, 2),
            None,
            torch.zeros(641, 2),
            torch.ones(641, 1),
            128,
            "ue8m0",
            512,
            2,
            640,
            640,
            result,
            False,
        )
    assert events.count("insert_index_keys") == 1
    assert events.count("prefill_topk") == events.count("decode_topk") == 1
    assert events.count("native_prefill_score") == 1
    assert events.count("native_decode_score") == (0 if enabled else 1)
    chosen = set(result[0].tolist())
    expected = (
        (set(range(384)) | set(range(512, 640))) if enabled else set(range(128, 640))
    )
    assert chosen == expected
    assert runtime.stats["indexer_calls"] == int(enabled)
    assert owner is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native CUDA kernels required"
)
def test_native_prefill_and_persistent_topk_keep_sink_local_quotas():
    """Run on a built CUDA vLLM installation; never claim CPU torch.topk suffices."""
    import vllm._custom_ops as ops

    config = C.CounterDSAConfig()
    lengths = [100, 640, 700]
    starts = [0, 100, 740]
    total = sum(lengths)
    prefill = torch.full((3, total), -1e20, device="cuda")
    decode = torch.full((3, 704), -1e20, device="cuda")
    expected = []
    for row, (start, length) in enumerate(zip(starts, lengths)):
        scores = torch.randperm(length - 1, device="cuda").float()
        C.fill_selection_scores(prefill[row], start, start + length, scores, config)
        C.fill_selection_scores(decode[row], 0, length, scores, config)
        expected.append(
            set(decode[row, :length].topk(min(length, 512)).indices.cpu().tolist())
        )
    output_prefill = torch.full((3, 512), -1, device="cuda", dtype=torch.int32)
    output_decode = torch.full_like(output_prefill, -1)
    ks = torch.tensor(starts, device="cuda", dtype=torch.int32)
    ke = ks + torch.tensor(lengths, device="cuda", dtype=torch.int32)
    ops.top_k_per_row_prefill(
        prefill,
        ks,
        ke,
        output_prefill,
        3,
        prefill.stride(0),
        prefill.stride(1),
        512,
    )
    workspace = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    torch.ops._C.persistent_topk(
        decode,
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        output_decode,
        workspace,
        512,
        max(lengths),
    )
    torch.cuda.synchronize()
    for output in [output_prefill, output_decode]:
        for row, length in enumerate(lengths):
            valid = output[row][output[row] >= 0].cpu().tolist()
            assert len(valid) == len(set(valid)) == min(length, 512)
            assert set(valid) == expected[row]
