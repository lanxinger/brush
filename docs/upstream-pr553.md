# Upstream PR #553: generated Fusion implementations

Port of [ArthurBrussee/brush#553](https://github.com/ArthurBrussee/brush/pull/553),
head `dc28d4fed217edb15286d3faf35e718614ddf7db`, onto fork
`26a5a732`. The upstream PR was merged and its checks passed when inspected on
2026-09-20. Local validation is recorded separately below.

## Changes

- Update Burn from `1414c8a1` to the PR's `3a93fbfc`, including its pinned
  CubeCL/CubeK dependencies, and enable Burn's `fusion` feature.
- Generate Fusion implementations for the shared image-loss operations and
  the backward-render trait. Metadata preserves the fork's minimum one-row
  raster buffer, optional refinement weight and deferred SH projection outputs.
- Register eagerly rendered outputs as `Init` tensors, preserving the selected
  rasterizer and tensor dtypes.
- Adopt the unified `Module` trait and `WgpuDevice::default()` throughout the
  fork, including checkpoint replay and finite-difference tests.
- Add the upstream FusionInspector regression for one fused gradient gather
  per parameter, adapted to the fork's scene batches.

## Fork adaptations

The fork still needs `brush-cube::fusion::register_custom` for appearance
compensation, saved SSIM partials and the private trusted-forward raster
bridge. Retain that API and its dependency, but replace the local closure
adapter with Burn's `OperationFn`. Keep the private forward-input constructor
and checked public raster methods separate, preserving native-MSL bounds-check
invariants. Autodiff/wrapping helpers used by appearance and training also stay.

Keep the Windows duplicate-version exceptions in `deny.toml`: this fork's
resolved graph still contains both versions. Upstream's blanket cleanup does
not apply here. The existing wgpu patch and revision are unchanged.

## Validation

Local host checks use Rust 1.98.1 on Apple Silicon. Cross-target checks use the
installed Rustup 1.98.0 toolchain, matching CI.

| Check | Result |
| --- | --- |
| Renderer/loss crates, all targets | Passed |
| Workspace tests, no default features, `brush-app/debug-validation` | 318 passed, 0 failed; 1 existing stress test ignored |
| Workspace tests, all features, `BRUSH_NATIVE_MSL_PRESET=1` | 325 passed, 0 failed; 2 existing stress tests ignored |
| Workspace check, no default features, all targets | Passed |
| Workspace Clippy, all targets/features, `-D warnings` | Passed |
| Browser libraries, no default features (`brush-app`, `brush-js`) | Passed |
| Browser library with `native-msl` enabled (`brush-app`) | Passed |
| iOS C library, `aarch64-apple-ios`, no default features | Passed |
| Workspace documentation, all features/private items, `-D warnings` | Passed |
| `cargo deny --locked check` | Advisories, bans, licenses and sources passed |
| Formatting and diff whitespace | Passed |

The FusionInspector regression passes on both WGSL and native MSL. The suites
cover numerical
gradients, culled splats, alternate rasterizers, appearance compensation,
refinement, and parameter trainability across module conversions. Native MSL
also passes deferred SH projection/optimizer regressions.

Browser execution, physical iOS execution, full-dataset quality/performance,
and hosted CI for this fork revision were not measured.
