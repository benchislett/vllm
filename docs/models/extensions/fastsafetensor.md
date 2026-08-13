Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

The default `--load-format auto` prefers fastsafetensors on supported NVIDIA and AMD GPU configurations. If fastsafetensors fails before yielding any weights, vLLM falls back to the standard Safetensors loader. If Safetensors weights are unavailable, it falls back to PyTorch `.bin` weights.

Pass `--load-format fastsafetensors` to require fastsafetensors without these fallbacks.
