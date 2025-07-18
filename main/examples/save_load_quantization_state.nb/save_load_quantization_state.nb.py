# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import difflib

from pathlib import Path
from tempfile import TemporaryDirectory

import fastforward as ff
import pygments
import torch

from IPython.display import HTML, display
from transformers import AutoModelForCausalLM

from doc_helpers.quick_start.quantized_models import quantized_llama as quantized_llama

# + [md]
# # Overview
#
# This notebook shows how to save and load the model quantization state. We need a model to play with:

# +
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

# + [md]
# Let's quantize it and initialize a few quantizers.

# +
ff.quantize_model(model)

ff.find_quantizers(model, "**/**/layers/*/self_attn/*/[quantizer:parameter/weight]").initialize(
    ff.nn.LinearQuantizer, num_bits=4, granularity=ff.PerChannel(), quantized_dtype=torch.float32
)

# + [md]
# Before the next steps, we need to estimate the quantization ranges for our quantizers.

# +
with (
    torch.no_grad(),
    ff.estimate_ranges(model, ff.range_setting.running_minmax),
    ff.set_strict_quantization(False),
):
    model(input_ids=torch.randint(0, 100, size=(1, 128)))

print(
    "Scale of the first quantizer: ", model.model.layers[0].self_attn.q_proj.weight_quantizer.scale
)

# + [md]
# Now we can save the quantization state.

# +
tmpdir = Path(TemporaryDirectory().name)
model.save_quantization_state(cache_dir=tmpdir)
for p in tmpdir.glob("**/*"):
    print(str(p))

# + [md]
# Both `save_quantization_state()` and `load_quantization_state()` methods accept several
# arguments to control where and how the quantization state is stored:
#
# * `cache_dir` argument
#   The `cache_dir` parameter specifies the base directory where quantization states are stored.
#   The quantization files will be saved in a subdirectory structure within this cache directory.
#   If not provided, the following directories will be used as a cache:
#    * `$FF_CACHE`
#    * `$XDG_CACHE_HOME/fastforward`
#    * `~/.cache/fastforward`
# * `tag` argument
#   The `tag` parameter allows you to create multiple versions or variants of quantization
#   states for the same model. This is useful when you want to save different quantization
#   configurations (e.g., different bit widths, granularities) for the same base model.
#   ```python
#   # Save with different tags
#   model.save_quantization_state(tag="4bit_perchannel")
#   model.save_quantization_state(tag="8bit_pertensor")
#   # Load specific tagged version
#   model.load_quantization_state(tag="4bit_perchannel")
#   ```
# * `name_or_path` argument
#   The `name_or_path` parameter specifies the model identifier used to organize quantization
#   states. By default, it uses the model's `name_or_path` from its config. If a model is not
#   a HuggingFace model (is not inhereted from `transformers.AutoModel`), you should either
#   manually add the `config.name_or_path` property or pass `name_or_path` to the function
#   explicitly.

# + [md]
# The quantization state consists of two files:
# * `config.yaml`
# * `model.safetensors`
#
# The `model.safetensors` is a binary file where state_dict (parameters and buffers) of all
# quantizers is saved.
# The `config.yaml` is a text file where other quantizer attributes are stored.

# +
config = pygments.highlight(
    next(tmpdir.glob("**/config.yaml")).read_text(encoding="utf8"),
    pygments.lexers.YamlLexer(),
    pygments.formatters.HtmlFormatter(),
)
display(HTML(f"<details><summary>config.yaml</summary>{config}</details>"))


# + [md]
# A `config.yaml` file might look scary due to its advanced, but valid, `yaml` syntax.
# Thus, it might be good to refresh your knowledge about yaml features:
# * sequence vs mapping
# * tags (`!` and `!!`)
# * anchors (`&` and `*`)
# * complex key definition (`?`)
#
# This [learnxinyminutes explanation](https://learnxinyminutes.com/yaml/) is a good reference guide
# for yaml ant its syntax.

# + [md]
# To load the quantization state you should use the [`load_quantization_state`](../../reference/fastforward/nn/quantized_module/#fastforward.nn.quantized_module.QuantizedModule.load_quantization_state) function:

# +
new_model = AutoModelForCausalLM.from_pretrained(model_name)
ff.quantize_model(new_model)
new_model_str = str(new_model)
new_model.load_quantization_state(cache_dir=tmpdir)
diff = pygments.highlight(
    "\n".join(difflib.unified_diff(new_model_str.splitlines(), str(new_model).splitlines())),
    pygments.lexers.DiffLexer(),
    pygments.formatters.HtmlFormatter(),
)
display(HTML(diff))

# + [md]
# As you can see, the quantizer stubs were replaced by the real quantizers. And the quantizer
# parameters are the same as well:

# +
torch.testing.assert_close(
    model.model.layers[0].self_attn.q_proj.weight_quantizer.scale,
    new_model.model.layers[0].self_attn.q_proj.weight_quantizer.scale,
)
