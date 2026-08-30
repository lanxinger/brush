#![cfg(target_family = "wasm")]

mod support;

use brush_js::BrushApp;
use support::request_host_device;
use wasm_bindgen_test::wasm_bindgen_test;

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn internal_initialization_rejects_a_later_host_device() {
    let app = BrushApp::new();
    app.init().await.expect("initialize Brush-owned device");
    app.init().await.expect("repeat Brush-owned initialization");

    let host = request_host_device()
        .await
        .expect("request host WebGPU device");
    let error = app
        .init_existing(host.adapter, host.device, host.queue)
        .expect_err("host device must not replace the Brush-owned device");
    assert!(
        error
            .as_string()
            .is_some_and(|message| message.contains("different GPU device"))
    );
}
