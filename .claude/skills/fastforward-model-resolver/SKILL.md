---
name: fastforward-model-resolver
description: Resolve a HuggingFace model ID into a working get_pipeline() factory function. Use when given an HF model ID and asked to implement model loading and input generation for a FastForward autoquant package. Uses transformers.pipeline when available (probe exit 0); returns needs_discovery with a local package path for non-standard models (probe exit 1) so fastforward-model-discovery can complete the work. Writes a single get_pipeline() function into __init__.py on the pipeline path.
---

# HuggingFace Model Resolver

Produces a single `get_pipeline()` factory function for a given HuggingFace model ID
when `transformers.pipeline` is available (probe exit 0).
When the probe exits 1 — `transformers` not installed, custom package, gated model —
the skill locates the installed package and returns `needs_discovery`, delegating
`CustomPipeline` construction to `$fastforward-model-discovery`.

The pipeline object returned on the pipeline path is a `transformers.Pipeline` instance
that exposes:

```python
pipe = get_pipeline(device, dtype)
pipe.model  # the raw torch.nn.Module, usable with FastForward
pipe.preprocess(x)  # raw input → model-ready tensors
pipe.forward(t)  # tensors → raw output
pipe(x)  # end-to-end call
```

`load_model` and `model_inputs` in `__init__.py` are thin wrappers over `get_pipeline()`:

```python
def load_model(device="cuda", dtype=None):
    return get_pipeline(device, dtype).model


def model_inputs(runs=1):
    pipe = get_pipeline()
    batch = pipe.preprocess(_dummy_input())
    return [((), batch)] * runs
```

**Required: `forward()` must return a plain tensor.** FastForward's export
pipeline traces the model to ONNX and then to QNN, both of which only
understand tensor graphs — a `ModelOutput` or tuple return breaks
tracing/export, not just the generated `verify` step (which calls
`model(*args, **kwargs).float().cpu()` directly on the output and fails
immediately if it's not a bare tensor). Many HF models return a `ModelOutput`
(has `.logits`/`.last_hidden_state`, no `.float()`) or a tuple when
`return_dict=False`. Register a forward hook in `load_model` to unwrap it to
a bare tensor:

```python
def _unwrap_output(module, args, output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def load_model(device="cuda", dtype=None):
    model = get_pipeline(device, dtype).model
    model.register_forward_hook(_unwrap_output)
    return model
```

Do this even if a quick manual check shows the model already returns a plain
tensor for the probe's dummy input — some models only return a bare tensor for
certain input shapes/kwargs and switch to `ModelOutput` otherwise.

---

## Workflow

### 1. Validate request

Required fields:
- `model_id` — HuggingFace model ID (e.g. `"bert-base-uncased"`)
- `output_file` — absolute path to `__init__.py` to write into

Optional fields:
- `hf_token` — token for private/gated repos
- `cache_dir` — local HF cache root. Passed to `probe_pipeline.py` as `--cache-dir`
  (sets `HF_HOME`) and to `fetch_model_info.py` as both `--model-dir` and
  `--tokenizer-dir`. When provided, `huggingface_hub` / `transformers` will use
  local files if present and only contact HuggingFace if a file is missing.
- `checkpoint` — path to a local `.pt` / `.bin` checkpoint file
- `dry_run` — if `true`, print synthesised code but do not write to file

Fail early with a clear message if `output_file` does not exist.

### 2. Fetch model metadata

Run the fetch script:

```
python scripts/fetch_model_info.py <model_id> \
    [--hf-token <token>] \
    [--model-dir <cache_dir>] \
    [--tokenizer-dir <cache_dir>] \
    [--checkpoint <path>]
```

This script has no `--cache-dir` flag (that flag belongs to `probe_pipeline.py`,
see step 3). If `cache_dir` was provided in the request, pass it as both
`--model-dir` and `--tokenizer-dir` — the script reads model files from
`--model-dir` and tokenizer files from `--tokenizer-dir` (falling back to the
other if a file is missing) directly from the on-disk `models--org--model/`
snapshot layout, entirely offline. The network is only used if a file is
absent from both dirs.

The script prints a JSON object with:
- `access` — `"ok"`, `"denied"`, or `"not_found"`
- `pipeline_tag` — from README frontmatter (may be null)
- `config` — `model_type`, `architectures`, key dims
- `tokenizer_class`, `model_max_length`
- `preprocessor_class`, `image_size`
- `readme_code_blocks` — all ```python blocks from README
- `model_dir`, `tokenizer_dir`, `checkpoint` — passed through

Exit codes: `0` = ok, `1` = unexpected error, `2` = access denied / not found.

**If exit code 2 — STOP immediately.** Show the `guidance` field verbatim and ask
for an `hf_token`:

```
I cannot access this model. Here is what the fetch script reported:

  <guidance field verbatim>

Please provide a HuggingFace token via the `hf_token` field in the request,
then I will retry.
```

### 3. Run the pipeline probe (mandatory — do not skip)

**Always run this script.** Its exit code is the only thing that determines which
path to follow — do not reason about whether the model supports transformers.pipeline.

```
python scripts/probe_pipeline.py <model_id> \
    [--pipeline-tag <pipeline_tag from fetch>] \
    [--cache-dir <cache_dir>] \
    [--hf-token <token>]
```

The probe sets `HF_HOME` from `--cache-dir` before loading anything.

Exit `0` → prints JSON with `status=success`, `pipeline_tag`, `modality`,
`auto_class`, `preprocessor_class`, `tensor_shapes`.

Exit `1` → prints JSON with `status=failed` and exception details.

**Exit 0 → Pipeline path (step 4). Exit 1 → needs_discovery handoff (step 5).**

---

### 4. Pipeline path (probe exit 0)

Generate a `get_pipeline()` function that constructs and returns a
`transformers.Pipeline` directly.

```python
def get_pipeline(
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype | None = None,
):
    import torch
    from transformers import pipeline as _pipeline
    _CACHE_DIR = <cache_dir or None>  # user-supplied
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = _pipeline(
        task="<pipeline_tag>",
        model="<model_id>",
        device=device,
        model_kwargs={
            "cache_dir": _CACHE_DIR,
            **({"torch_dtype": dtype if isinstance(dtype, torch.dtype)
                else getattr(torch, dtype)} if dtype else {}),
        },
        token=<hf_token or None>,
    )
    pipe.model.eval()
    return pipe
```

`_dummy_input()` must return a **raw**, modality-appropriate input — the same
shape of thing a user would pass to `pipe(...)` — because `model_inputs()`
below calls `pipe.preprocess(_dummy_input())`, and `preprocess()` expects raw
input, not pre-tokenized tensors. Branch on the `modality` field from the
probe's exit-0 JSON output (mirrors `probe_pipeline.py`'s own dispatch:
`text_tags` → string, `vision_tags` → PIL image, `audio_tags` → array dict):

```python
def _dummy_input():
    # modality from probe: "text" | "vision" | "audio" | "unknown"
    return "dummy text"  # text: raw string
    # return Image.new("RGB", (224, 224))  # vision: PIL.Image, from PIL import Image
    # return {"array": np.zeros(16000, dtype=np.float32), "sampling_rate": 16000}  # audio
```

Do not hardcode a pre-tokenized `{"input_ids": ..., "attention_mask": ...}`
tensor dict — `tensor_shapes` in the probe output describes the *preprocessed*
tensor shapes for reference/debugging only, not the shape `_dummy_input()`
itself should produce.

Then write `load_model` and `model_inputs` as wrappers. `load_model` must apply
the `_unwrap_output` forward hook shown above so `forward()` always yields a
plain tensor:

```python
def load_model(device="cuda", dtype=None):
    model = get_pipeline(device, dtype).model
    model.register_forward_hook(_unwrap_output)
    return model


def model_inputs(runs=1):
    import torch

    pipe = get_pipeline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        k: v.to(device)
        for k, v in pipe.preprocess(_dummy_input()).items()
        if isinstance(v, torch.Tensor)
    }
    return [((), batch)] * runs
```

---

### 5. needs_discovery handoff (probe exit 1)

Used when `pipeline()` fails — `transformers` not installed, custom package,
gated model, or bespoke forward signature. Do **not** implement `CustomPipeline`
here; delegate to `$fastforward-model-discovery` instead.

#### 5a. Find the local package path

```
python scripts/find_local_package.py <model_id>
```

The script tries in order: `pip show`, `importlib.util.find_spec`, `PYTHONPATH`
scan. It prints a JSON object:

```json
{
  "package_name": "sam3",
  "package_path": "/path/to/sam3",
  "source": "pip" | "importlib" | "pythonpath" | null,
  "error": null | "<message>"
}
```

If `package_path` is non-null, use it as `model_source` for the handoff.
If `package_path` is null, include the `error` field in the response so the
user knows they must supply `model_path` manually.

#### 5b. Return `needs_discovery`

Stop immediately and return:

```json
{
  "status": "needs_discovery",
  "model_id": "<model_id>",
  "output_file": "<output_file>",
  "model_path": "<package_path or null>",
  "checkpoint": "<checkpoint or null>",
  "probe_failure": "<reason from probe JSON>",
  "notes": [
    "probe exited 1 — transformers.pipeline unavailable",
    "run $fastforward-model-discovery with source_type=local_path and model_source=<model_path>"
  ]
}
```

The calling agent (or `$fastforward-autoquant`) must invoke
`$fastforward-model-discovery` with:
- `source_type`: `"local_path"`
- `model_source`: `model_path` from above
- `output_file`: same `output_file` as this request

Do not write anything to `output_file` — leave it for `$fastforward-model-discovery`.

---

### 6. Write to output_file

Before writing anything, read `output_file` and check whether `load_model` and
`model_inputs` already have real implementations (i.e. no `raise RuntimeError("Stub: ...")`
body). If both are already implemented and the request does not include `overwrite=true`,
skip writing and return `status="success"` with a note explaining nothing was changed.

For each function that still contains a stub body (or when `overwrite=true`):
replace the stub body and insert the generated code.

Insert before `load_model` (in order):
1. `get_pipeline()` function
2. `_dummy_input()` helper

Rules:
- Do not touch `quantized_model`, `bypass`, `custom_operators_table`,
  `replacement_patterns`, or module-level imports.
- Preserve existing function signatures.
- Keep all imports inside function/class bodies — no module-level side effects.
- Embed `cache_dir` and `checkpoint` as named constants inside the functions
  that use them, marked with `# user-supplied`.

### 7. Return structured response

```json
{
  "status": "success" | "partial" | "error" | "needs_discovery",
  "model_id": "<model_id>",
  "pipeline_path": "transformers" | "needs_discovery",
  "pipeline_tag": "<tag or null>",
  "output_file": "<path>",
  "get_pipeline_source": "<full source or null>",
  "load_model_source": "<full source or null>",
  "model_inputs_source": "<full source or null>",
  "model_path": "<local package path or null — present when status=needs_discovery>",
  "checkpoint": "<checkpoint path or null>",
  "notes": ["..."]
}
```

Notes must include: which path was taken and why, input shapes source,
any assumptions needing manual correction.

---

## Agent behaviour rules

- Run `fetch_model_info.py` and `probe_pipeline.py` once each — never skip either.
- **Probe exit code is the only decision criterion.** Never reason about whether
  a model supports `transformers.pipeline` — let the script decide.
- **Exit 0 → pipeline path (step 4). Exit 1 → run `find_local_package.py` and return `needs_discovery` (step 5). Never implement `CustomPipeline` here.**
- **Read `output_file` before writing anything.** If `load_model` and `model_inputs` already
  have real implementations (no `raise RuntimeError("Stub: ...")` body) and `overwrite` is
  not set, do not modify the file. The user may have edited it manually.
- When probe exits 0, derive `_dummy_input()`'s raw input from the `modality` field in
  probe output (`"text"` → string, `"vision"` → PIL image, `"audio"` → array dict) — not
  from `tensor_shapes`, which describes preprocessed tensor shapes for reference only.
- When probe exits 1, do not read README code blocks or package source — return `needs_discovery` immediately after running `find_local_package.py`.
- `get_pipeline()` is the primary output on the pipeline path. `load_model` and `model_inputs` are always
  thin wrappers over it — never implement them independently.
- Default device `"cuda"`; resolve at runtime with
  `"cuda" if torch.cuda.is_available() else "cpu"`.
- `model_inputs` tensors must be on the same device as the model.
- If `checkpoint` is provided, always use it — never ignore it.
- If `cache_dir` is provided, embed it in `get_pipeline()` and pass it to every
  `from_pretrained` / builder call in the generated code.

---

## Example — transformers.pipeline path (bert-base-uncased)

```json
{"model_id": "bert-base-uncased", "output_file": "/tmp/pkg/__init__.py",
 "cache_dir": "/data/hf_cache"}
```

Probe reports `pipeline_tag="fill-mask"`, `modality="text"` — a text task, so
`_dummy_input()` returns a raw string, not a pre-tokenized tensor dict.

```python
def _dummy_input():
    return "The capital of France is [MASK]."


def get_pipeline(device="cuda", dtype=None):
    import torch
    from transformers import pipeline as _pipeline

    _CACHE_DIR = "/data/hf_cache"  # user-supplied
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pipeline(
        "fill-mask",
        model="bert-base-uncased",
        device=device,
        model_kwargs={"cache_dir": _CACHE_DIR},
    )


def load_model(device="cuda", dtype=None):
    model = get_pipeline(device, dtype).model
    model.register_forward_hook(_unwrap_output)
    return model


def model_inputs(runs=1):
    import torch

    pipe = get_pipeline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        k: v.to(device)
        for k, v in pipe.preprocess(_dummy_input()).items()
        if isinstance(v, torch.Tensor)
    }
    return [((), batch)] * runs
```

---

## Example — transformers.pipeline path (vit-tiny-patch16-224)

```json
{"model_id": "WinKawaks/vit-tiny-patch16-224", "output_file": "/tmp/pkg/__init__.py"}
```

Probe reports `pipeline_tag="image-classification"`, `modality="vision"` — a
vision task, so `_dummy_input()` returns a PIL image, not a pre-tokenized
tensor dict.

```python
def _dummy_input():
    from PIL import Image

    return Image.new("RGB", (224, 224))


def get_pipeline(device="cuda", dtype=None):
    import torch
    from transformers import pipeline as _pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return _pipeline(
        "image-classification",
        model="WinKawaks/vit-tiny-patch16-224",
        device=device,
    )


def load_model(device="cuda", dtype=None):
    model = get_pipeline(device, dtype).model
    model.register_forward_hook(_unwrap_output)
    return model


def model_inputs(runs=1):
    import torch

    pipe = get_pipeline()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = {
        k: v.to(device)
        for k, v in pipe.preprocess(_dummy_input()).items()
        if isinstance(v, torch.Tensor)
    }
    return [((), batch)] * runs
```

---

## Example — needs_discovery handoff (facebook/sam3)

```json
{"model_id": "facebook/sam3", "output_file": "/tmp/sam3_pkg/__init__.py",
 "checkpoint": "/path/to/checkpoints/sam3.pt"}
```

Probe exits 1 (`transformers` not installed in the sam3 env). `find_local_package.py`
finds the package via importlib:

```json
{"package_name": "sam3", "package_path": "/path/to/models/sam3", "source": "importlib", "error": null}
```

Resolver returns immediately:

```json
{
  "status": "needs_discovery",
  "model_id": "facebook/sam3",
  "output_file": "/tmp/sam3_pkg/__init__.py",
  "model_path": "/path/to/models/sam3",
  "checkpoint": "/path/to/checkpoints/sam3.pt",
  "probe_failure": "transformers not installed",
  "notes": [
    "probe exited 1 — transformers.pipeline unavailable",
    "run $fastforward-model-discovery with source_type=local_path and model_source=/path/to/models/sam3"
  ]
}
```

The calling agent then invokes `$fastforward-model-discovery` with:
- `source_type`: `"local_path"`
- `model_source`: `"/path/to/models/sam3"`
- `output_file`: `"/tmp/sam3_pkg/__init__.py"`
- `checkpoint` (optional): `"/path/to/checkpoints/sam3.pt"`
