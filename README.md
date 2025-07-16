# 📼 FastForward

FastForward is a Python package built on top of PyTorch for neural network
quantization. It aims to serve as a foundation for research and prototyping in
quantization. By leveraging PyTorch's eager mode, you can experiment with
neural network quantization as easily as with any other PyTorch module. This
means you can use breakpoints, print statements, and other introspection
methods with quantized neural networks, just as you would with standard ones.

## Status

FastForward is currently under active development. While it has been
successfully used in various projects, core parts of the library may still
change.

## Main Features

- **Quantized Tensor**: A versatile container for quantized data that supports
  multiple quantization formats while retaining metadata.
- **Range Estimation**: General methods for range estimation that can be easily
  extended to new quantization methods.
- **Quantized Operator Dispatching**: A dispatcher built on top of the PyTorch
  dispatcher, specialized for different quantization schemes and methods.
- **Quantization Setup and Initialization**: A step-by-step process for
  converting a non-quantized model into a quantized one, customizable at each
  stage.
- **Quantization Safety**: A default mode ensuring correctly quantized models
  that can be deployed to efficient hardware, with opt-out options if needed.
  This helps to catch common quantization mistakes early.
- **mpath**: A utility to describe, access, and update multiple layers in
  a module hierarchy at a higher level of abstraction.

## Roadmap

- **More Quantization Methods**: Implementations of quantization methods such
  as Omniquant, GPTQ, SpinQuant, and others.
- **Autoquant**: Automatic conversions any non-quantized PyTorch model into an
  eager-mode quantized model.
- **Export**: Generation of deployment artifacts for functional quantized neural
  networks.

## Getting Started

To get started, explore these tutorials:

- [Getting Started: Quantizing a LLM from Scratch](https://qualcomm-ai-research.github.io/fastforward/latest/examples/quantizing_networks.nb/)
- [Quick Start: Quantization of Llama-v3](https://qualcomm-ai-research.github.io/fastforward/latest/examples/quick_start_quantize_llms.nb/)
- [Save and load quantization state](examples/save_load_quantization_state.nb.py)

For more tutorials and the API reference, visit the [general documentation](https://qualcomm-ai-research.github.io/fastforward).

### How to Get It

1. Ensure you have a working installation of PyTorch.

2. Install FastForward:

```bash
pip install git+https://github.com/Qualcomm-AI-research/fastforward@main
```
