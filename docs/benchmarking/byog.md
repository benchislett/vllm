# BYOG: build an image and run a suite

BYOG combines a vLLM CI image with a runner, EvalScope, a pinned GSM8K dataset
and reference specs. Image creation runs on CPU builders; contributors run inference on their
own GPUs. This initial implementation supports one local server per run and
NVIDIA CUDA images for amd64 or arm64.

## Preview and run

From a checkout containing the BYOG benchmark and eval clients:

```bash
python -m tools.byog.run \
    --spec tools/byog/specs/nemotron-nano-nvfp4.yaml \
    --output preview --dry-run
```

A dry run resolves configuration, validates server/client flags using vLLM's
parsers, and writes the spec and command plan. It does not download weights,
start a server, or require GPUs. Use a different, empty output directory for the
actual run; omit `--dry-run` to execute it.

The included reference is a **smoke configuration** for the pinned NVIDIA
Nemotron Nano NVFP4 checkpoint: GSM8K on 32 questions and 8k1k workloads at concurrency 1/16/64
(20/160/640 measured requests). It is an initial configuration to validate on
your hardware, not a calibrated model accuracy or performance reference.

For an image published by the build below:

```bash
IMAGE='public.ecr.aws/REGISTRY/BYOG-REPOSITORY@sha256:DIGEST'
docker run --rm --gpus all --ipc host \
    -v "$PWD/out:/out" \
    -v "$HF_HUB_CACHE:/root/.cache/huggingface/hub" \
    -e HF_TOKEN -e BYOG_IMAGE_REFERENCE="$IMAGE" \
    "$IMAGE" --spec tools/byog/specs/nemotron-nano-nvfp4.yaml --output /out/run
```

Model weights are downloaded on the machine running the suite, using the pinned
model revision. The benchmark tokenizer is resolved at that same revision.
The image contains the clients and pinned GSM8K data, not model weights. Other
evaluation datasets are loaded by EvalScope at run time (or from a mounted local
snapshot). Install `vllm[eval]` when running from a local checkout.

The runner starts one server with a unique served-model name, waits for that
identity, and probes generation before running the suite. Evaluations and
benchmark concurrency points run sequentially against that server. Startup and
client timeouts are configurable. The runner stops its process groups on normal
completion, errors and interruption, and preserves completed results when later
steps fail. Exit zero means every selected job completed, not that an accuracy
threshold was met. Prefix caching is disabled for the suite and each benchmark
verifies the server configuration and hit counters.

## Reference recipes and overrides

A spec selects a pinned model, `serve.config` (native vLLM server settings),
`serve.env`, evaluations, and benchmark workloads/concurrency points. Use
`--override custom.yaml` to merge user settings; mapping keys merge recursively
and lists replace the original list. For example:

```yaml
serve:
  config:
    tensor-parallel-size: 2
  startup_timeout_seconds: 1800
benchmarks:
  - workload: 8k1k-v1
    concurrency: [1, 16, 64, 256]
```

Evaluations accept any registered EvalScope dataset name or the MRCR presets
from [Accuracy Evaluation](evaluation.md). For example, an override can replace
the smoke evaluation list:

```yaml
evals:
  - name: gsm8k
    args:
      generation-config: {max_tokens: 32768, temperature: 0.7}
  - name: gpqa_diamond
    args:
      generation-config: {max_tokens: 32768, temperature: 0.7}
  - name: aime25
    args:
      generation-config: {max_tokens: 32768, temperature: 0.7}
  - name: mrcr-v2-8needle-128k
    args:
      generation-config: {max_tokens: 4096, temperature: 0}
  - name: mrcr-v2-8needle-256k
    args:
      generation-config: {max_tokens: 4096, temperature: 0}
```

Select a model and server context large enough for the MRCR inputs **and output**;
the included 16k smoke server configuration cannot run this expanded suite.
These are example sampling settings, not a calibrated reference for every model.
Use `limit` for a smoke subset and `eval-batch-size` for request concurrency.
In the image, GSM8K defaults to its bundled local snapshot; explicit
`dataset-args` or `dataset-hub` disables that automatic selection. For other
datasets, use those options to pin a revision or mount a local dataset snapshot.
The runner owns `api-url`, `work-dir`, model and report paths, and accepts eval
settings directly in the spec rather than external `eval-config` files.

An optional recipe points to a **local rendered single-node Recipes JSON**:

```yaml
recipe:
  path: ./recipe.json
  sha256: FULL_SHA256_OF_RECIPE_JSON
```

Export the desired deployment from [vLLM Recipes](https://recipes.vllm.ai/), save
its JSON and pin its hash. BYOG reuses the existing
[`tools/recipes` converter](../../tools/recipes/README.md); it does not fetch a
moving website configuration at run time. The recipe must select the same model
as the spec. Explicit `serve.config`/`serve.env` settings override the recipe,
and the bundle retains a copy of the pinned recipe. Multi-process recipe
renderings are rejected instead of launching only part of a deployment.

The runner owns model revision, host, port, model alias, and client report
locations. These cannot be overridden inside server/client arguments. Keep
credentials in the process environment rather than the recorded spec. A null
environment override removes a variable; the default removes the CI image's
`VLLM_GPU_SYNC_CHECK=error` and selects the Python benchmark client.

## Result bundle

The output directory contains:

- `spec.resolved.yaml`, the exact resolved configuration, plus recipe JSON when used.
- `plan.json`, `server.yaml`, `server.json`, server/client logs and a generation probe.
- Individual `byog.measurement/v1` reports and benchmark request records.
- Per-eval directories containing EvalScope configs, predictions, reviews and native reports.
- `result.json` (`byog.result/v0`), job statuses, build metadata and basic hardware snapshots.
- `manifest.json`, file sizes and SHA256 hashes; `bundle.tar.gz` containing those files.
- `repro.sh`, using the reported image digest when supplied, or the local Python runner.

`BYOG_IMAGE_REFERENCE` is supplied by the caller and recorded as such; a process
inside a container cannot independently attest its own image digest. Build info
records the source/harness commit, base digest and Buildkite URL. This MVP uses
the harness from the requested source commit, not an independently trusted
harness release. Neither a complete report nor BuildKit provenance is a signed
correctness certificate. Upload/authentication, signing, PR comments, dashboard
ingestion, multi-node orchestration and resuming partial suites are follow-ups.

Reports include raw text and paths from the run. Review them before sharing.
GPU inventory includes model, memory and driver, without UUID or serial number;
it is inventory, not a claim about how many GPUs each server operation used.

## Manual image publication

Repository maintainers can invoke the **Build BYOG image** workflow with a source
branch/commit and architecture. The workflow resolves the branch to a SHA before
submitting an **image-only** Buildkite build. No GPU test suite is submitted.
Source branches must include the benchmark/eval clients and BYOG build scripts;
older branches fail rather than producing an image with missing entrypoints.

One-time infrastructure configuration is required:

1. Create a dedicated public ECR BYOG repository. Keep it separate from CI and
   release repositories. Grant its publication permission to the existing
   premerge image-builder role.
2. Configure a separate Buildkite pipeline for this GitHub repository, with
   `buildkite-agent pipeline upload .buildkite/byog-pipeline.yaml` as its upload
   command. Use the existing CPU image builders and their Docker/AWS tooling.
3. Set repository variables `BUILDKITE_ORGANIZATION`, `BUILDKITE_BYOG_PIPELINE`,
   `BYOG_IMAGE_REPO`, `BYOG_AMD64_QUEUE` and `BYOG_ARM64_QUEUE`; configure the
   existing `BUILDKITE_API_TOKEN` secret with access to create builds there.
   Queues must match the requested architecture. `BYOG_REGISTRY` on the builders
   can override the default public registry.

The build invokes the existing x86 or ARM CI image builder, reusing an existing
commit image where available and its normal build caches otherwise. It resolves
the resulting manifest by digest, verifies pinned GSM8K content, then builds and
pushes `docker/Dockerfile.byog`. The amd64 Bake overlay is pinned to a ci-infra
revision. The final image has a separate per-build tag, BuildKit provenance and a
`byog-image.json` artifact containing its digest. The Buildkite page also links
the image through the existing annotation utility.

**Retention is manual in this MVP.** The dedicated repository retains the image
and its copied layers independently of the CI tag's lifetime. This code neither
provisions that repository nor installs cleanup rules; configure retention with
the infrastructure owners before promising an expiry to users. No cloud builds
or GPU runs are triggered by checking out or previewing this code.
