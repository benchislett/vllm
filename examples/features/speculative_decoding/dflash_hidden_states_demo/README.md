# DFlash hidden-state client/server demo

This demo launches
`nvidia/smart-panda-mtp-graft-NVFP4-20260701` with
`nvidia/nano3.5_smart_panda_dflash`, sends concurrent OpenAI-compatible chat
requests, and writes one speculative-decoding training trace per request. The
recorder also supports EAGLE3 and DSpark speculators that consume auxiliary
target hidden states; DFlash provides the concrete configuration used here.

The server and client must use the same target and draft models and must have
access to the same dump directory. The client also needs Hugging Face access
to the model checkpoints so it can load the target output norm and LM head and
read the draft model configuration.

## Start the server

Run the server in the first terminal, adjusting tensor parallelism and memory
settings for the available GPUs:

```bash
.venv/bin/python \
  examples/features/speculative_decoding/dflash_hidden_states_demo/server.py \
  --tensor-parallel-size 1 \
  --dump-dir /tmp/dflash-traces
```

The launcher defaults to 16 speculative tokens, a maximum model length of
32768, 32 concurrent sequences, and GPU memory utilization of 0.85. It forces
the V2 model runner, disables prefix caching so all prompt positions are
recorded, enables request-ID headers, and passes `--trust-remote-code`.
Unknown arguments are forwarded to `vllm serve`; for example, append
`--enforce-eager` to disable CUDA graphs while debugging. Use `--dry-run` to
print the full `vllm serve` command without starting it.

## Run the client

After the server is ready, run the client in another terminal:

```bash
.venv/bin/python \
  examples/features/speculative_decoding/dflash_hidden_states_demo/client.py \
  --base-url http://127.0.0.1:8000/v1 \
  --dump-dir /tmp/dflash-traces \
  --num-requests 12 \
  --concurrency 8 \
  --max-tokens 64
```

The client sends the built-in single-turn and multi-turn conversations in
parallel. Requests use temperature 1.0 sampling and set
`chat_template_kwargs.enable_thinking` to `true`. If the number of requests
exceeds the number of conversations, the client reuses them cyclically.

If the server was launched with model overrides, pass the corresponding
`--model` and `--draft-model` values to the client. Use `--revision` and
`--draft-revision` when the server uses non-default Hugging Face revisions.

After every response completes, the client:

- discovers its dump by reading the `request_id` safetensors metadata;
- validates the trace and reconstructs the prompt and completion token IDs;
- computes target-output metrics over the primary completion trajectory;
- randomly selects one completed request using `--seed`;
- detokenizes its prompt and completion separately through the server, with a
  visible separator between them.

Dump filenames are the SHA-256 digest of vLLM's internal request ID. The
internal ID can include a suffix not present in the public chat-completion ID,
so the client discovers files from metadata instead of predicting filenames
from response IDs.

## Trace format

Each completed request produces one safetensors file and no manifest or
sidecar. Its metadata contains:

- `format`: `vllm-spec-decode-training-hidden-states-v1`
- `request_id`: vLLM's internal request ID

The file contains these tensors:

| Tensor | Meaning |
| --- | --- |
| `prompt_len` | Original prompt length as a one-element tensor. |
| `prefill_token_ids` | Token IDs computed during recorded prefills. |
| `prefill_positions` | Absolute positions for the prefill rows. |
| `prefill_hidden_states` | Auxiliary target states with shape `[prefill rows, auxiliary layers, hidden size]`. |
| `verification_input_token_ids` | Anchors followed by every proposed token, including rejected proposals. |
| `verification_positions` | Absolute positions for all verification rows. |
| `verification_hidden_states` | Target states for every verification row with shape `[verification rows, auxiliary layers, hidden size]`. |
| `verification_block_offsets` | Offsets delimiting verification blocks. |
| `output_token_ids` | Accepted proposals plus each block's recovery or bonus token. |
| `output_block_offsets` | Offsets delimiting the committed outputs from each block. |

Auxiliary layer IDs are not duplicated in each trace. Their order comes from
the target and draft model configurations supplied to the server and client.
The client verifies that the final configured auxiliary layer is the target's
final pre-output-norm hidden state before computing logits.

The first verification block's anchor is the initial generated token. A later
block's anchor repeats the preceding block's final committed token and is
checked for continuity rather than appended twice. The concatenated committed
outputs are trimmed to the API's completion-token count because a final
speculative step can compute tokens beyond a stop condition.

## Validation and metrics

Both prefill and verification states must be nonempty and finite, with no
NaNs, infinities, or all-zero token/layer vectors. Exact zero scalar components
are counted and reported but permitted.

For target-output metrics, the client loads the target output RMSNorm and LM
head from the Hugging Face checkpoint. The smart-panda NVFP4 head is
dequantized in vocabulary chunks, so the client does not materialize a full
BF16 copy or a full token-by-vocabulary logits matrix. The default
`--target-metrics-device auto` uses CUDA when available; use
`--target-metrics-device cpu` if the serving GPU lacks spare memory. Control
temporary projection memory with `--logits-vocab-chunk-size`.

The final prompt state predicts the first completion token. For each
verification block, the anchor and accepted-prefix states predict its committed
outputs. Rejected proposal rows remain in the trace but are excluded from the
primary-trajectory metrics.

For every primary prediction position, the client computes:

- target entropy over the full vocabulary, in bits;
- `q = max softmax(logits)`, the probability of the target's top-1 token;
- whether that top-1 token equals the sampled next token.

It reports token-weighted global and per-request mean and median entropy, mean
and median `q`, and top-1 hit rate. With temperature 1.0 and no additional
sampling truncation, the expected hit rate is the arithmetic mean of `q`, not
its median. The per-request hit-rate histogram therefore should not be read as
a histogram of per-token top-1 probabilities.

The terminal output also includes histograms for:

- per-token entropy;
- global top-1 hit and miss counts;
- per-request top-1 hit rates;
- committed output tokens per verification step;
- API completion lengths.

Differences between consecutive `output_block_offsets` are the number of
tokens generated by each speculative verification step. Their global mean is
printed as the mean acceptance length and includes the recovery or bonus token.

## Resource behavior

The recorder holds active traces in CPU memory and writes them asynchronously
when requests finish. The client writes no files. Each request therefore has
O(1) persistent filesystem artifacts: one safetensors dump.

Increase concurrency and output length gradually when testing large models,
since recorded states consume CPU memory proportional to prompt length,
verification work, auxiliary layer count, and hidden size.
