# Temporary MCAP dependency patch

This is the published `mcap` 0.25.0 crate from foxglove/mcap commit
`0e1528a3dad6942384dae40b64827567413a18cf`, under the accompanying MIT
license. Its only source-package change is to require `zstd` 0.14.3 in
`Cargo.toml` on both native and WebAssembly targets. This supplies the fixes
for RUSTSEC-2026-0391, RUSTSEC-2026-0398 and RUSTSEC-2026-0403 while Rerun
requires MCAP 0.25. Remove the patch when Rerun adopts a compatible release
using a fixed Zstandard dependency.
