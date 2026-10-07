# Training parameters: how to tune them

CLI help describes each flag; this guide explains how to adjust training duration, splat growth, and memory use. Run `brush --help` for the full list and the defaults in your build. The headless `brush-cli` accepts the same training flags.

## Goals and the main knobs

| Goal | Start here |
| --- | --- |
| Shorter / longer runs | `--total-train-iters`, `--growth-stop-iter` |
| Fewer / more splats | `--max-splats`, `--growth-grad-threshold`, `--growth-select-fraction`, `--refine-every` |
| Lower GPU memory use | `--max-resolution`, `--max-splats`, `--sh-degree` |
| Lower CPU image-cache memory use | `--max-scene-batch-cache-size` |
| Scout a large dataset | `--subsample-frames`, `--subsample-points`, `--max-frames` |

The defaults allow 30,000 training steps and up to 10 million splats during growth. For a preview, lower resolution and iterations first, then adjust growth. The native-MSL optimization preset selects implementation optimizations; it does not change these CLI defaults.

## Training duration

### `--total-train-iters` (default `30000`)

Total optimization steps for the initial training phase. Additional steps after growth stops can refine existing splats. With `--lod-levels` enabled, each LOD adds `--lod-refine-steps` beyond this budget.

For an initial preview, try 2,000–5,000 steps. Compare longer runs against the preview using held-out views and visual inspection; a larger step budget does not guarantee better results for every scene.

Lower `--growth-stop-iter` when shortening a run so that some steps remain for refinement after growth.

### `--growth-stop-iter` (default `15000`)

Stops gradient-driven growth at this iteration, clamped to `total_train_iters`. Start near half the total budget, then adjust:

- If the cloud still has poor coverage, increase this and usually the total training budget.
- If coverage is sufficient and you want more time refining it, decrease it.

In this fork, pruning, replacement of dead splats, and splitting oversized splats can continue at refinement steps after gradient-driven growth stops. `--split-at-screen-size` controls oversized splitting (default `0.5`; `0` disables it).

### `--refine-every` (default `200`)

Interval between refinement steps that prune, replace, and densify splats. A starting estimate is the number of training views needed to cover the scene.

- With fewer images, try a smaller interval.
- With a large dataset, a larger interval can reduce refinement overhead.
- Smaller intervals provide more opportunities for growth and may reach the splat budget sooner.

## Splat count

`--max-splats` controls the growth ceiling, not a target count. Actual count depends on growth, splitting, and pruning.

### `--max-splats` (default `10000000`)

Limits new splat growth and subsamples an oversized loaded initialization cloud. For a first memory-constrained run, try 500,000–2,000,000; for a larger budget, try 2,000,000–5,000,000. These are starting points, not device guarantees.

If training reaches the cap early and coverage remains poor, either increase it if memory allows or slow growth with a higher `--growth-grad-threshold` or lower `--growth-select-fraction`.

### `--growth-grad-threshold` (default `0.0025`)

A lower threshold marks more splats as candidates for gradient-driven growth. Raise it to grow more slowly or keep a smaller cloud.

### `--growth-select-fraction` (default `0.25`)

Fraction of candidates used to determine the growth budget. Raise it for more aggressive growth; lower it to stay compact. Available headroom and pruning also affect how many splats are added.

### `--opac-decay` (default `0.004`)

Gradually reduces opacity so weak splats can be pruned. A slightly higher value may keep counts down; excessive decay can remove useful coverage.

## GPU memory and system RAM

Memory comes from several separate allocations:

- Splat parameters, gradients, and optimizer state grow with splat count. SH coefficient storage per splat grows with `(degree + 1)^2`.
- Training images, rendered images, and loss buffers grow with pixel count. For the same aspect ratio, doubling the long edge roughly quadruples the pixels.
- Projection, sorting, and raster buffers depend on visible splats and their overlaps with image tiles; scene geometry and view direction affect these sizes.
- The packed scene-batch cache resides in CPU memory and has its own budget.

On Apple Silicon, CPU and GPU allocations share physical memory, so reducing the host cache can still ease overall memory pressure. Its budget does not cap total process memory or GPU allocations.

### `--max-resolution` (default `1920`)

Long-edge cap for loaded training images. Lower it to reduce per-view pixel buffers and image-cache size. Try `800`–`1280` for a preview, then increase it if detail and memory headroom justify another run.

Resolution also affects projection, refinement, and the fork's per-splat minimum-scale filter. Evaluate a final run at its intended resolution rather than assuming a low-resolution preview has identical growth behavior.

### `--sh-degree` (default `3`)

Spherical-harmonics degree (`0`–`4`). Higher degrees support richer view-dependent color and require more coefficient memory and computation per splat.

If memory or step time is limiting, try `2` or `1` and inspect the loss of view-dependent detail.

### `--max-scene-batch-cache-size` (default `6GiB` native / `2GiB` wasm)

CPU cache budget for packed training batches: one 32-bit RGBA value per pixel, with shared host buffers on cache hits. This avoids repeated image decoding and packing, but the trainer still uploads a batch to the GPU when it uses it. It is not a cache of resident GPU images.

Try `2GiB` or `1GiB` when system RAM is limited. Increase the budget when repeated decoding is a bottleneck and there is RAM available. Images that do not fit bypass the cache and are decoded and packed again on later visits. Temporary decoding and prefetch buffers also use RAM outside this budget.

### `--subsample-frames` / `--max-frames` / `--subsample-points`

Fewer views reduce dataset loading and cache demand, but each training step still processes one view; dropping views can also reduce scene coverage. Fewer SfM points produce a smaller loaded initialization cloud.

- Try `--subsample-frames 2` or `4` to scout settings.
- Use `--max-frames` to cap loaded views.
- Use `--subsample-points` when the initialization cloud is large; `--max-splats` also caps loaded initialization points.

## Learning rates and loss

Usually keep `--lr-mean`, `--lr-mean-end`, `--lr-coeffs-dc`, `--lr-opac`, `--lr-scale`, `--lr-rotation`, `--ssim-weight`, and related settings at their defaults. Change them when investigating convergence or a specific scene issue. Prefer resolution, growth, and iteration budget for initial tuning.

## Evaluation and export

| Flag | Practical note |
| --- | --- |
| `--eval-every` | Smaller intervals evaluate more often and add work. |
| `--eval-split-every` | Holds out every Nth view for evaluation. Use at least `2` for a held-out split. |
| `--eval-save-to-disk` | Saves evaluation renders under the export path for visual inspection. |
| `--export-every` / `--export-path` | Controls checkpoint cadence and location; more exports cost time and disk space. |

With appearance compensation enabled, `--train-on-eval` includes evaluation views in training and enables their learned corrections during evaluation. Those metrics measure training-view fit rather than held-out generalization. See the [appearance compensation notes](../README.md#appearance-compensation).

## Suggested starting recipes

These examples have not been benchmarked as quality or memory tiers. Compare results on your scene and hardware, and replace `<dataset>` with a dataset path or URL.

**Short preview**

```sh
brush <dataset> --total-train-iters 4000 --growth-stop-iter 2500 \
  --max-resolution 800 --max-splats 1500000 --sh-degree 2
```

**Moderate budget**

```sh
brush <dataset> --total-train-iters 20000 --growth-stop-iter 12000 \
  --max-resolution 1280 --max-splats 3000000
```

**Longer run at the default resolution**

```sh
brush <dataset> --total-train-iters 30000 --growth-stop-iter 15000 \
  --max-resolution 1920 --max-splats 5000000 --sh-degree 3
```

For a GPU allocation failure, start by lowering resolution, the splat budget, or SH degree according to which buffers dominate. For system RAM pressure, also lower the packed batch-cache budget. If coverage remains sparse, inspect the splat count against the cap, then adjust the growth threshold, selection fraction, or growth duration.

Adapted from the [upstream training-parameter guide](https://github.com/ArthurBrussee/brush/blob/ff959c609a03bf4eb0ab5115a2a01951f58db49c/docs/training-parameters.md), with defaults and memory behavior checked against this fork.
