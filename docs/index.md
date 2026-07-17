---
hide:
  - navigation
  - toc
---

<div class="landing-page" style="display:none"></div>

<div align="center" markdown>

![FastForward logo](assets/ff-logo-purple.png){ width="140" }

# FastForward
**PyTorch-native quantization, built for fast experimentation and prototyping.**
<br>
</div>

---

## Why FastForward

<div class="grid cards" markdown>


-   :simple-pytorch:{ .lg .middle } **It is still PyTorch**

    ---
    A `Module` goes in, a `Module` comes out.
    FastForward extends the PyTorch dispatcher, so your training loop, your checkpoints and your optimizer do not change.


-   :material-autorenew:{ .lg .middle } **Made for fast iterations**

    ---

    `find_quantizers(…)` reaches any layer with a selector, 
    `.initialize(…)` sets bitwidth and granularity from the outside.
    Change the numbers, run again, you never edit the model.

-   :material-puzzle:{ .lg .middle } **Extensible**

    ---

    Quantizers, range estimators, and operators are all plug-in points. Ideal
    for research on new quantization methods.

</div>


## Features

<div class="grid cards" markdown>

-   :material-cube-outline: __Quantized Tensor__

    A versatile container for quantized data that supports multiple
    quantization formats while retaining metadata.

-   :material-ruler: __Range Estimation__

    General methods for range estimation, easy to extend to new quantization
    schemes.

-   :material-router-network: __Operator Dispatch__

    A dispatcher built on top of PyTorch's, specialized for different
    quantization schemes and methods.

-   :material-shield: __Safe Quantization__

    Ensures all the executed operations are quantized, with explicit exceptions if needed.
    This helps catch common quantization mistakes early.

-   :material-file-tree: __MPath__

    Describe, access, and update layers deep in a module hierarchy at a higher
    level of abstraction. [Tutorial](examples/mpath.nb.py).

-   :material-magic-staff: __Autoquant__ *(experimental)*

    Automatic conversion of any PyTorch model into an eager-mode quantized
    model. [Tutorial](examples/autoquant.md).

</div>


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

# 4. Run the quantized model like any PyTorch model — pdb and print still work.
output = model(**input_batch)
```

[Full walkthrough on Llama-v3 :material-arrow-right:](examples/quick_start_quantize_llms.nb.py){ .md-button .md-button--primary }
[All tutorials](examples/index.md){ .md-button }



## Install

Requires a working PyTorch install (≥ 2.4).

```bash
pip install git+https://github.com/Qualcomm-AI-research/fastforward@main
```


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

---

!!! info "Project status"
    FastForward is under active development. It is already used in research and
    production projects at Qualcomm AI Research, but core APIs may still
    evolve. Roadmap items include additional post-training methods and richer 
    export targets.

