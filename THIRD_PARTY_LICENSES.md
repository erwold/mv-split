# Third-Party Code Notices

This project includes code derived from third-party open-source projects.
The canonical license text and a per-file list of modifications live in each
file's header. This document is a top-level summary for convenience.

## Unsloth — <https://github.com/unslothai/unsloth>

Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.

The following files are derived from Unsloth source. Each file's header
carries the full license text and the list of modifications.

| Local file | Upstream file | License |
| --- | --- | --- |
| `kernels/rmsnorm.py` | `unsloth/kernels/rms_layernorm.py` | Apache License 2.0 |
| `kernels/swiglu.py` | `unsloth/kernels/swiglu.py` | Apache License 2.0 |
| `kernels/rope.py` | `unsloth/kernels/rope_embedding.py` | LGPL v3 |

The Apache 2.0 license text is available at
<http://www.apache.org/licenses/LICENSE-2.0>.

The LGPL v3 license text is available at
<https://www.gnu.org/licenses/lgpl-3.0.html>.

Note: `kernels/rope.py` (LGPL v3) carries copyleft obligations distinct from
the Apache-2.0 files; downstream redistribution of modified versions of that
file must remain LGPL v3. The other files in this repository are released
under their respective per-file or repository-level licenses.
