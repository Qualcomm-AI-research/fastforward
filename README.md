<div align="center">
  <img src="docs/assets/ff-logo-purple.png" alt="FastForward logo" width="140">

  <h1>FastForward</h1>

  <p><b>Eager-mode neural network quantization for PyTorch.</b><br>
  Shrink and speed up your models — with the debugger, prints, and pdb still working.</p>

  <p>
    <a href="https://qualcomm-ai-research.github.io/fastforward"><img src="https://img.shields.io/badge/docs-latest-6a4cbf" alt="docs"></a>
    <img src="https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue" alt="python">
    <img src="https://img.shields.io/badge/PyTorch-%E2%89%A5%202.4-ee4c2c" alt="pytorch">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause--Clear-green" alt="license"></a>
  </p>
</div>

---

FastForward is a Python package built on top of PyTorch for neural-network
quantization. It is designed for research and prototyping: because it runs in
PyTorch's eager mode, quantized models behave like any other `torch.nn.Module`.
You can drop in `breakpoint()`, add `print` statements, and step through with
`pdb` — nothing new to learn.

## Why FastForward

- **Eager-mode by design** — `pdb`, `print`, and IDE debuggers work through
  quantized ops. No graph capture, no tracing indirection.
- **PyTorch-native dispatcher** — extends the PyTorch dispatcher rather than
  replacing it, so your model stays a `nn.Module`.
- **Safe by default** — a strict mode catches common quantization mistakes
  (e.g. calling a quantized op with un-quantized tensors) early; opt out per
  call when you need to.
- **Extensible** — quantizers, range estimators, and operators are all plug-in
  points, ideal for research on new quantization methods.



## Quick look

```python
import fastforward as ff
import torch

# 1. Convert any PyTorch model to a quantization-ready one.
model = MyModel()
ff.quantize_model(model)

# 2. Attach 8-bit per-channel weight quantizers to every linear layer.
weight_quantizers = ff.find_quantizers(model, "**/[quantizer:parameter/weight]")
weight_quantizers.initialize(ff.nn.LinearQuantizer, num_bits=8, granularity=ff.PerChannel())

# 3. Calibrate on real data.
with ff.estimate_ranges(model, ff.range_setting.RunningMinMaxRangeEstimator):
    for batch in calibration_loader:
        model(**batch)

# 4. Run the quantized model like any PyTorch model — pdb still works.
output = model(**input_batch)
```

See the [Quick Start on Llama-v3](https://qualcomm-ai-research.github.io/fastforward/latest/examples/quick_start_quantize_llms.nb/)
for the full walkthrough.

## Features

- **Quantized Tensor** — a versatile container for quantized data that supports
  multiple quantization formats while retaining metadata.
- **Range Estimation** — general methods for range estimation, easy to extend
  to new quantization schemes.
- **Quantized Operator Dispatch** — a dispatcher built on top of PyTorch's,
  specialized for different quantization schemes and methods.
- **Quantization Setup** — a step-by-step process for converting a
  non-quantized model into a quantized one, customizable at each stage.
- **mpath** — a utility to search, access, and update layers deep in a module
  hierarchy at a higher level of abstraction.
- **Autoquant** *(experimental)* — automatic conversion of any PyTorch model
  into an eager-mode quantized-ready model.
- **Export** — generation of deployment artifacts from quantized networks.

## Install

Requires a working PyTorch install (≥ 2.4).

```bash
pip install git+https://github.com/Qualcomm-AI-research/fastforward@main
```

## Tutorials

- [Getting Started — Quantizing an LLM from scratch](https://qualcomm-ai-research.github.io/fastforward/latest/examples/quantizing_networks.nb/)
- [Quick Start — Quantization of Llama-v3](https://qualcomm-ai-research.github.io/fastforward/latest/examples/quick_start_quantize_llms.nb/)
- [Save and load quantization state](https://qualcomm-ai-research.github.io/fastforward/latest/examples/save_load_quantization_state.nb/)
- [Autoquantizing PyTorch modules](https://qualcomm-ai-research.github.io/fastforward/latest/examples/autoquant.nb/)
- [mpath — selecting submodules and quantizers](https://qualcomm-ai-research.github.io/fastforward/latest/examples/mpath.nb/)
- [Exporting a quantized model](https://qualcomm-ai-research.github.io/fastforward/latest/examples/export_llama.nb/)

Full docs and API reference: <https://qualcomm-ai-research.github.io/fastforward>.

## Status

FastForward is under active development. It is already used in research and
production projects at Qualcomm AI Research, but core APIs may still evolve.
Roadmap items include additional post-training methods (Omniquant, SpinQuant)
and richer export targets.


## Citation

If you use FastForward in your research, please cite it as:

```bibtex
@software{fastforward,
    title        = {FastForward: A PyTorch-based Library for Neural Network Quantization},
    author       = {Peters, Jorn and Behrends, S{\"o}nke and Del Chiaro, Riccardo and
                    Mironov, Evgeny and van Rozendaal, Ties and Stasis, Spyridon and
                    Weitkamp, Laurens and Nagel, Markus},
    year         = {2024},
    url          = {https://github.com/Qualcomm-AI-research/fastforward}
}
```


## License

BSD-3-Clause-Clear. See [LICENSE](LICENSE).
