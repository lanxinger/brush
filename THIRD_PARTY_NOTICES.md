# Third-party notices

## NVIDIA PPISP

The PPISP implementation in `crates/brush-appearance` is adapted from
[`nv-tlabs/ppisp`](https://github.com/nv-tlabs/ppisp) at commit
`5233d38e223b4685db86367dd53d4be31e733d9c`.

Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0. A copy of that license is in
the repository's `LICENSE` file.

## Spirulae-Splat hybrid PPISP grid

The hybrid implementation in `crates/brush-appearance/src/ppisp_grid.rs` and
`ppisp_grid_kernels.rs` is adapted from
[`ArthurBrussee/brush#483`](https://github.com/ArthurBrussee/brush/pull/483) and
the [`spirulae-splat`](https://github.com/harry7557558/spirulae-splat) hybrid
PPISP-grid formulation.

Those two source files are marked `GPL-3.0-only` and are not covered by the
repository's root Apache-2.0 license.
