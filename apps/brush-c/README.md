# brush-c

`brush-c` is Brush's blocking native C ABI. It builds an `rlib`, `cdylib`, and
`staticlib`; Apple hosts should embed the static library.

## iOS device build

Install the Rust device target and build the release archive:

```sh
rustup target add aarch64-apple-ios --toolchain stable
RUSTC="$(rustup which --toolchain stable rustc)" \
  rustup run stable cargo build \
  --locked --release -p brush-c --target aarch64-apple-ios
```

The host app needs:

- `target/aarch64-apple-ios/release/libbrush_c.a`
- `apps/brush-c/include/brush_c.h`
- the Metal, QuartzCore, CoreGraphics, Foundation, and CoreFoundation frameworks
- `libobjc` and `libiconv` (the Apple toolchain supplies libSystem, libc, and libm)

Link `libbrush_c.a` by its full path. Do not use `-lbrush_c`: both the static
archive and a dynamic library are emitted with that stem, and selecting the
dynamic library leaves an iOS app with an unembedded dependency at launch.

For a distributable app dependency, package device and simulator archives with
the header as a static XCFramework rather than referencing this checkout from
the Xcode project.

## Mobile options

Existing callers can keep using `train_and_save` and `BrushTrainOptions`.
Memory-constrained hosts should use `train_and_save_v2` with
`BrushTrainOptionsV2`, which adds:

- `alpha_mode`: `0` for automatic handling, `1` for masks, `2` for transparency.
- `max_splats`: an explicit splat-count ceiling; `0` keeps Brush's default.

The call blocks, so invoke it off the main actor. The current ABI has no
cancellation entry point and shares global GPU initialization; host apps should
allow one foreground training job at a time.
