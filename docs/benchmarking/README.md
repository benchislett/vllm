# Benchmark Suites

vLLM provides comprehensive benchmarking tools for performance testing and evaluation.

- **[Accuracy Evaluation](./evaluation.md)**: Run `vllm eval gsm8k` against an existing server and save reproducible result reports.
- **[Benchmark CLI](./cli.md)**: `vllm bench` CLI tools and specialized benchmark scripts for interactive performance testing.
- **[Parameter Sweeps](./sweeps.md)**: Automate `vllm bench` runs for multiple configurations, useful for [optimization and tuning](../configuration/optimization.md).
- **[Performance Dashboard](./dashboard.md)**: Automated CI that publishes benchmarks on each commit.
- **[Benchmark using vLLM Recipes](../../tools/recipes/README.md)**: Use the Recipe conversion tool to generate `config.yaml` and `env.sh` from [vLLM Recipes](https://recipes.vllm.ai/) for Benchmarking.

- **[BYOG](./byog.md)**: Manually build an image and run pinned benchmark/evaluation suites on your GPUs.
