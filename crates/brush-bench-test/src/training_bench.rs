#![recursion_limit = "256"]
mod benches;

fn main() {
    // Benchmark the GPU work, not debug-validation's synchronous readbacks.
    // Keeping this at the executable boundary leaves the shared helpers fully
    // validated when the ordinary test harness calls them.
    brush_render::validation::set_enabled(false);
    divan::main();
}
