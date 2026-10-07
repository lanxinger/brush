# Upstream PR #556: forward rendering and viewer memory

Adapted selected mechanisms from [ArthurBrussee/brush#556](https://github.com/ArthurBrussee/brush/pull/556),
using canonical commit `1388f74c6fe0236f68ee4915564bf00e9d2e3747`, onto fork
`7c8e7721407372a08da0d11018a765e49abbdd7b` on 2026-10-07. The complete upstream
patch conflicts in ten files; this is a focused port rather than a cherry-pick.

## Changes and fork adaptations

- Packed forward rendering skips both inverse-map writes and uses an unused
  one-element placeholder. Backward passes retain the complete global map,
  including zero entries for culled splats. Empty scenes retain a zero-length
  map. For a nonempty scene, the allocation shrinks by `4 * (N - 1)` bytes and
  the kernels avoid `N + V` stores, where `V` is the visible splat count.
- Viewer scale is passed through the internal rasterizer extension and shared
  projection uniforms. Projection adds the host-computed log offset to raw
  log scales and multiplies the minimum-scale floor by the original scalar.
  This preserves the fork's scaled-floor opacity compensation without making
  scaled transform or floor tensors. The separate scalar avoids rounding the
  floor multiplier through `exp(ln(scale))`.
- The viewer upload cache stores a weak pixel-allocation identity. Repaints
  still avoid duplicate uploads and reuse GPU buffers/bind groups, while the
  cache releases obsolete CPU pixels when their final real owner disappears.

The public training `SplatOps` interface and C ABI are unchanged. Training
passes scale `1.0`; Float inference retains backward bookkeeping. Both tile
layouts, Fusion, autodiff, and native SH optimizations are preserved. The
upstream SSIM, hard-cutoff gradient, and Adam arithmetic edits are excluded.
No dependencies or defaults changed.

Separately, [PR #524](https://github.com/ArthurBrussee/brush/pull/524)'s training
guide was adapted in [training-parameters.md](training-parameters.md), with
the fork's packed host-cache, filtering, and growth-stop behavior explained.

## Validation

Host tests use Rust 1.99.0 on an Apple M4 Pro. Cross-target checks use the
installed Rustup 1.98.0 compiler and target standard libraries. Linting uses
Rust 1.98.0, matching the repository's CI toolchain.

| Check | Result |
| --- | --- |
| Renderer library, native MSL, preset enabled | 51 passed; three existing large stress tests filtered out |
| Native MSL finite differences, Fusion, training integration | 43 passed |
| Renderer library, portable WGSL | 51 passed; same three stress tests filtered out |
| Portable WGSL finite differences, Fusion, training integration | 43 passed |
| Viewer upload identity and obsolete-frame release | 2 passed |
| Browser app/JS libraries, wasm32, no default features | Passed |
| Browser app, wasm32, native-MSL feature fallback configuration | Passed |
| C library, aarch64 iOS, no default features | Passed |
| Affected packages, Clippy all targets/features, Rust 1.98, `-D warnings` | Passed |
| Workspace formatting and diff whitespace | Passed |

Rust 1.99 Clippy reports redundant-field-name warnings in existing LPIPS
`Config` macro output. The strict lint check above uses CI's Rust 1.98 toolchain
without suppressing warnings.

The new projection regressions cover high global IDs, culled/all-culled scenes,
full backward maps, packed quantization, both rasterizer layouts, Float/Packed
output, Mip on/off, floors on/off, and viewer scaling compared with the previous
materialized-tensor mechanism. Existing tests check empty scenes, an independent
CPU raster oracle, gradients, scaled-floor baking, concurrent training/viewing,
and dataset-unit export.

The inverse-map allocation regression also fails on the original fork: a
4,097-splat packed scene returns a 4,097-element map instead of the new
one-element placeholder. Only the test's internal selector argument list was
adapted to the old API for this check.

A separate before/after native-MSL check rendered 200,000 deterministic splats
at 1280x720 with scales 1.0/1.6 and floors off/on. All four packed images were
byte-identical to the original fork, totaling 14,745,600 compared bytes. This is
a synthetic correctness check, not full-dataset quality evidence. Five paired
before/after runs, each taking the median of 27 warmed frames per case, had
median latency changes between -0.5% and +0.2%. This small workload establishes
no measurable speed improvement; the port removes allocations and writes.

Browser execution, physical iOS execution, and interactive viewer behavior were
not exercised.
