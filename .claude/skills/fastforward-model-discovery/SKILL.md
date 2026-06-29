---
name: fastforward-model-discovery
description: Resolve a local path or git URL to a model source directory, scan it for usable load_model/model_inputs patterns, and write working implementations into an existing __init__.py. Use when the autoquant skill returns needs_user_input for a local_path or git_url source.
---

# FastForward Model Discovery

Resolves `local_path` and `git_url` model sources and writes `load_model` and
`model_inputs` implementations into an already-bootstrapped `__init__.py`.

## Workflow

### 1. Validate request

Required fields:
- `source_type` — `"local_path"` or `"git_url"`
- `model_source` — path or git URL
- `output_file` — absolute path to an existing `__init__.py`

Optional fields:
- `source_ref` — git branch/tag/commit (git_url only)
- `overwrite` — if `true`, overwrite existing non-stub implementations (default `false`)

Fail early with a clear message if `output_file` does not exist.

### 2. Resolve source to a local directory

- `local_path`: verify the path exists and is a directory.
- `git_url`: clone into a deterministic local cache directory; fetch+checkout `source_ref`
  if provided. If clone fails, check for a local fallback under `./models/<name>` or `./<name>`.

Run the resolver:
```
python scripts/resolver.py \
    --source-type <local_path|git_url> \
    --model-source <path_or_url> \
    [--source-ref <ref>]
```

The script prints a JSON object:
- `model_path` — absolute path to the resolved local directory
- `resolved_revision` — git HEAD SHA (git_url only, null for local_path)
- `error` — error message if resolution failed (null on success)

If `error` is non-null, stop and report it verbatim.

### 3. Scan the resolved directory

Run the scanner:
```
python scripts/scanner.py <model_path>
```

The script scans Python files under `model_path` (prioritising `examples/`, `scripts/`,
`tests/`, `demo/`, `notebooks/`) and prints a JSON object:
- `module_classes` — list of `torch.nn.Module` subclass names found
- `factory_calls` — list of factory function/method calls found (e.g. `from_pretrained`, `build_model`)
- `parse_errors` — number of files that could not be parsed
- `files_scanned` — total files scanned

### 4. Check existing implementations before writing

Read `output_file` and determine, for each of `load_model` and `model_inputs`, whether it
already has a real implementation or still contains a stub body
(i.e. `raise RuntimeError("Stub: ...")`).

- **Real implementation present and `overwrite=false` (default):** skip that function entirely.
  Do not read, analyse, or modify it. Record a note: `"load_model already implemented — skipped"`.
- **Stub body present, OR `overwrite=true`:** proceed to explore and write.

If both functions already have real implementations and `overwrite=false`, skip step 5 and
return `status="success"` with a note explaining nothing was written.

### 5. Explore and write implementations

Use scan results as a map. Read the most promising files — constructors, factory
functions, example scripts — to understand exactly how the model is instantiated
and what inputs it takes.

Write only the functions identified in step 4 as needing an update:
- Replace only the target function body.
- Do not modify `quantized_model`, `bypass`, `custom_operators_table`,
  `replacement_patterns`, or any module-level imports.
- Keep all new imports inside function bodies.
- Make both implementations deterministic.
- **`load_model` must return a model whose `forward()` yields a plain
  `torch.Tensor`.** FastForward's export pipeline traces the model to ONNX and
  then to QNN, both of which only understand tensor graphs — a `ModelOutput`
  or tuple return breaks tracing/export, not just the generated `verify` step
  (which calls `model(*args, **kwargs).float().cpu()` directly on the output
  and fails immediately if it's not a bare tensor). If the model returns a
  `ModelOutput`-like object (has `.logits`/`.last_hidden_state`, no `.float()`)
  or a tuple, register a forward hook to unwrap it:

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
  ```

  Read the scanned source to confirm what `forward()` actually returns before
  assuming it's already a bare tensor.

If you cannot confidently determine an implementation from source alone, return
`needs_user_input` with specific guidance on what the user should provide.

### 5. Return structured response

```json
{
  "status": "success" | "needs_user_input" | "error",
  "output_file": "<path>",
  "model_path": "<resolved path>",
  "resolved_revision": "<sha or null>",
  "load_model_source": "<written source or null>",
  "model_inputs_source": "<written source or null>",
  "notes": ["..."]
}
```

---

## Agent behaviour rules

- Run `resolver.py` first — never assume a path is valid without it.
- **Read `output_file` before writing anything.** If a function already has a real
  implementation (no `raise RuntimeError("Stub: ...")` body) and `overwrite=false`,
  do not touch it — not even to "improve" it. The user may have edited it manually.
- Read actual source files before writing implementations; don't guess from class names alone.
- Prefer real usage examples (`examples/`, `scripts/`, `tests/`) over constructing inputs
  from scratch.
- Default device `"cpu"` for `load_model`; let the caller move to GPU.
- `load_model` must ensure `forward()` returns a plain tensor (see step 5) — verify
  depends on this unconditionally.
- `model_inputs` must return `list[tuple[tuple[Any,...], dict[str,Any]]]` — no `seed` parameter.
- Never modify `quantized_model`, `bypass`, `custom_operators_table`, `replacement_patterns`,
  or module-level code in `output_file`.
- If the scan returns zero files or the model class/factory cannot be found, return
  `needs_user_input` with actionable guidance rather than writing broken stubs.
