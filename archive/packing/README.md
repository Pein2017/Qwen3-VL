# Packing (archived)

This directory contains the removed grouped packing implementation (datasets, collator, metrics mixin, length-cache helpers) that formerly lived under `src/packing/`.

Key points:
- Packing is **disabled** in the runtime. `training.packing=true` is rejected and the module is not on the Python import path.
- Standard padded batching is the only supported mode.
- If you need to resurrect packing for research, copy this directory back under `src/packing/` on a feature branch and wire it manually; mainline code does not import from here.
