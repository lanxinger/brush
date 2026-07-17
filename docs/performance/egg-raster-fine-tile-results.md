# Egg raster fine-tile candidates

- Date: 2026-07-17
- Machine: M4 Pro
- Dataset: `/Users/markus/Downloads/GS_DATASETS/egg`
- Compiler: native MSL
- Views: four fixed masked training views
Timing: 20 samples x 4 synchronized steps after 4 warmup steps, except the
2400x3200 stress pairs (5 samples x 4 steps)

## Verdict

The first 8x8 candidate was consistently faster, but it failed the provisional
memory gate at the 1.16M checkpoint. A 16x8 specialization retains the speedup,
avoids the 8x8 intersection explosion, and passes the measured whole-step and
memory gates at every tested scale.

The 16x8 path remains an explicit experimental opt-in:

```sh
BRUSH_NATIVE_MSL_FINE_RASTER_TILES=1 ./target/release/brush
```

It is intentionally not included in `BRUSH_NATIVE_MSL_PRESET`. Fine tiles alone
use checked raster backward. If `BRUSH_NATIVE_MSL_UNCHECKED_RASTER_BWD=1` is
also requested, directly or through the preset, the candidate uses the same
host-validated unchecked launch contract as the 16x16 path.

## Gate

The provisional promotion gate was:

- at least 5% lower whole-step latency;
- no more than 25% growth in `/usr/bin/time -l` peak memory footprint;
- selector parity for forward output, geometry-invariant auxiliaries, expected
  tile-offset shape, and gradients.

The 16x8 candidate clears the performance and footprint gates. It should remain
outside the preset until a full 15k quality bake-off and longer training soak
also pass.

## 8x8 result: rejected on 1.16M footprint

| Workload | 16x16 | 8x8 | Latency change | Peak footprint change |
|---|---:|---:|---:|---:|
| 1.16M, algorithm-only ABBA | 88.512 ms | 81.786 ms | -7.60% | +30.38% |
| 1.16M, deployment ABBA | 70.082 ms | 64.679 ms | -7.71% | +30.47% (separate repeat) |
| 2.49M, 1440x1920 | 121.437 ms | 105.465 ms | -13.15% | +13.72% |

For the 1.16M algorithm-only ABBA, average peak footprint rose from 3.645 GiB
to 4.753 GiB. The stable full-preset repeat independently measured +30.47%.
That exceeds the 25% limit even though the larger checkpoint stayed within it.

## 16x8 result: selected candidate

| Workload | 16x16 | 16x8 | Latency change | Throughput change | Peak footprint change |
|---|---:|---:|---:|---:|---:|
| 1.16M, checked algorithm-only ABBA | 75.462 ms | 68.258 ms | **-9.55%** | +10.55% | -23.65% |
| 1.16M, full-preset deployment ABBA | 71.730 ms | 65.055 ms | **-9.31%** | +10.26% | -23.65% |
| 2.49M, 1440x1920 interleaved pair | 127.991 ms | 109.682 ms | **-14.30%** | +16.69% | -7.35% |
| 2.49M, 2400x3200, mean of two pairs | 218.741 ms | 204.009 ms | **-6.73%** | +7.22% | +3.74% |

At 1.16M, the two deployment pairs were individually 9.32% and 9.29% faster.
At 2400x3200, absolute time moved substantially with concurrent system load,
but the two adjacent A/B pairs remained 6.28% and 7.12% faster. The
interleaved ratios, rather than cross-phase absolute times, are the useful
comparison.

`maximum resident set size` stayed nearly unchanged between selectors. The
table uses the separate macOS `peak memory footprint` field as the end-to-end
memory guard; it should not be interpreted as GPU allocation telemetry alone.

## Intersection census

The census uses the same four 1.16M/1440x1920 views and reports exact GPU range
ends. All sampled CPU replays matched the shortened GPU range ends.

| Geometry | Reserved intersections | Relative | Post-raster intersections | Relative |
|---|---:|---:|---:|---:|
| 16x16 | 5.658M | 1.000x | 3.077M | 1.000x |
| 8x8 | 12.866M | 2.274x | 6.042M | 1.963x |
| 16x8 | 8.552M | 1.512x | 4.338M | 1.410x |

The rectangular grid is the useful middle point: it halves per-tile pixel work
while adding about 51% reserved and 41% surviving intersections, rather than
the 8x8 candidate's 127% and 96% increases. Its 128-thread Morton mapping is an
exact 16-column by 8-row cover.

## Correctness coverage

- Independent CPU raster oracle for hard and smooth cutoff.
- Selector parity for output image and projected splats.
- Boundary grids below, on, and above the 16x8 dimensions.
- Hard- and smooth-cutoff parity for visibility, radius, transform gradients,
  SH gradients, and opacity gradients.
- Candidate backward exercised through both checked and native-MSL unchecked
  launches.
- Rectangular census fixture verifies the 8-pixel y stride.

This is strong kernel-level and short-replay evidence, not a replacement for a
full training-quality comparison. Numerical differences from changed atomic
grouping remain within the existing test tolerances; checkpoint-replay final
losses differ at the normal low-order nondeterministic level.

## Reproduction

Build:

```sh
cargo build --release -p brush-bench-test \
  --bin brush-checkpoint-replay --features native-msl
```

Algorithm-only comparison (set fine tiles to `0` and `1` in separate A/B/B/A
processes):

```sh
BRUSH_NATIVE_MSL_PRESET=1 \
BRUSH_NATIVE_MSL_UNCHECKED_RASTER_BWD=0 \
BRUSH_NATIVE_MSL_FINE_RASTER_TILES=0 \
./target/release/brush-checkpoint-replay \
  --dataset /Users/markus/Downloads/GS_DATASETS/egg \
  --ply target/bench-checkpoints/egg-quality-15k-exact/egg_15000.ply \
  --max-resolution 1920 --views 4 --eval-split-every 20 \
  --alpha-mode masked --warmup-steps 4 --steps-per-sample 4 --samples 20
```

Deployment comparison uses the same command without the explicit unchecked
override, so both geometries receive the full preset. Census runs add the
`raster-census` feature and `--raster-census-tiles 256`; census readbacks are
excluded from timing runs.

## Next gate

Run a frozen 15k Legacy-versus-16x8 quality bake-off with identical seed,
views, and preset, then evaluate both checkpoints with the existing checkpoint
evaluator. Only after that result and a longer soak should 16x8 be considered
for inclusion in the macOS preset.
