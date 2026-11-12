# Experimental completed-prefix Counter scoring for DeepSeek V3.2

This research fork is based on upstream vLLM `v0.23.0`, commit
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`. It adds an opt-in, eager
reference implementation. It has **not** been validated with a real DeepSeek
V3.2 checkpoint or the native GPU attention/top-k kernels. It is not a
production-ready performance implementation.

## Algorithm

The requested selection budget is 512 visible positions per query:

| Component | Positions |
| --- | ---: |
| Leading sink tokens | 4 |
| Historical tokens selected by full Counter | 380 |
| Most recent local window | 128 |

Ranges overlap for short contexts; they are treated as a union. Contexts of
512 tokens or fewer fit without pruning. This changes the checkpoint's native
`index_topk=2048` to 512 **only when Counter mode is explicitly enabled**.

Full Counter replays **every** decoder layer with cached main-model K/V, then
extracts each actual historical token's raw LM-head logit. Importance is
`U_i = -logit_i`, not negative DSA similarity and not normalized NLL. The
reverse mask permits `j > i + 1` for the default `min_gap=1`. The otherwise
unscorable last `min_gap+1` snapshot tokens receive the original Counter
safety priority. Sink/local positions are pinned before native top-k.

The indexer runs before current-step main MLA attention. Therefore scoring
uses the **completed prefix before this forward**, not incomplete KV from
later layers. For ordinary decode this excludes the current input token,
which is protected by the local window. For a multi-token forward the
snapshot may lag by the whole scheduled chunk. It is not same-step exact
Counter as used in the earlier Hugging Face eviction experiment.

Each layer's own main-MLA block table gathers that request's historical KV.
Token IDs are tracked by logical request ID and actual consumed device
tokens, not by reusable physical cache slots. Cached RoPE keys are not
rotated a second time. BF16 MLA and V3.2's 656-byte `fp8_ds_mla` layout are
handled; the learned FP8 **indexer** cache is not mistaken for main-model KV.

The existing indexer call sites, IndexCache skip rules, prefill chunking,
batch scheduler, top-k kernels, main sparse attention and KV allocator remain.
Scores are recomputed at each original indexer invocation; a request's score
is reused only across chunks within that invocation. Private Q/MoE/LM-head
tiling bounds reference-workspace allocation without changing serving batch
size. This deliberately expensive full replay is not the single-layer Fast
approximation.

## Initialization boundary

At the first prefill there is no completed snapshot. A large later prefill
can also introduce unfinished history outside local128. Such rows use
**native DSA scores with the same sink/local quotas** (`warmup="native"`).
This is explicit bootstrap behavior, not full-Counter prefill; the runtime
logs a warning on its first occurrence and counts all such rows. Set
`warmup="error"` to reject them instead. It will not fabricate Counter
scores for tokens whose all-layer KV does not exist.

## Execution support

The first reference bridge requires:

- `FLASHMLA_SPARSE` and BF16 model execution; FP8 checkpoint weights are used
  through their existing loaded projection/MoE methods.
- `--enforce-eager`. CUDA graph/compiled replay is not implemented. Use eager
  mode for the baseline too when comparing this reference implementation.
- Pipeline/context/data parallel sizes1; no expert parallel, sequence-parallel
  MoE, DBO/microbatching, speculative decoding, LoRA, KV transfer or CPU weight
  offload. Unsupported modes raise an error rather than being reconfigured.
- Tensor-parallel projection/vocabulary handling is implemented, including
  reduction of the true-token logit across vocabulary shards, but multi-GPU
  runtime correctness remains unvalidated.

The native scheduler's batch limits are not changed. The private
`query_chunk_size` must fit its existing MoE token capacity. Main KV remains
allocated: selecting 512 positions does **not** limit physical cache occupancy
to 512 tokens. Native dummy/profile runs stay native; their memory estimate
does not account for the full long-prefix Counter replay workspace. Available
headroom must be checked on real hardware before any serving/performance claim.

## Enable

Use a Linux vLLM build from this branch with compatible FlashMLA/DeepGEMM and
enough model/working memory. `TP` is the tensor-parallel size appropriate to
the available hardware; this document does not assert a model fits any
particular card count.

```bash
MODEL=/path/to/DeepSeek-V3.2
TP=8  # Example only: choose for the actual model capacity and hardware.
vllm serve "$MODEL" --tensor-parallel-size "$TP" \
  --enforce-eager --attention-backend FLASHMLA_SPARSE \
  --additional-config '{"counter_dsa":{"min_gap":1,"frozen_tokens":4,"local_window":128,"history_tokens":380}}'
```

For a same-512-budget native DSA baseline, use the same model, hardware,
eager mode and batch settings, omit `counter_dsa`, and pass
`--hf-overrides '{"index_topk":512}'`. Native2048 is a separate budget control;
it must not be confused with a same-budget scorer comparison. No experiment
has been launched automatically by this change.

With the Python `LLM` API, counters can be retrieved after generation:

```python
stats = llm.apply_model(lambda model: dict(model.counter_runtime.stats))
```

Each worker reports `indexer_calls`, `full_counter_passes`, `counter_rows`,
`native_warmup_rows` and `all_fit_rows`. A result using native bootstrap must
report that count rather than claiming every prefill row used Counter.

## Validation

The CPU tests load the scoring module independently of GPU extensions:

```bash
.venv/bin/python -m pytest \
  tests/model_executor/layers/test_counter_causal_dsa.py --noconftest -q
```

They cover an independent three-layer dense mathematical oracle, absolute
reverse masks, score sign, target-token alignment, completed-prefix lifetime,
mixed-request/chunk offsets, noncontiguous KV pages, FP8 cache unpacking,
sink/local quota unions and explicit unsupported-mode rejection. The actual
Python indexer body is also executed with **mocked** native kernel boundaries
to check both original and Counter control flow.

One additional test runs native prefill and persistent CUDA top-k and checks
the exact selected set and padding. It is explicitly skipped without CUDA.
That test, distributed execution and a real full-checkpoint end-to-end smoke
must pass on a supported GPU before trusting runtime or performance results.
The initial CPU checks used Windows, Python3.14 and torch2.15+cpu, not the
target Linux/GPU environment. Python3.12 syntax is checked separately.
