use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::js_sys::{Array, Promise};

#[wasm_bindgen(inline_js = r#"
export async function requestBrushTestDevice() {
  if (!navigator.gpu) {
    throw new Error("WebGPU is unavailable");
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("WebGPU returned no adapter");
  }

  // Match the public demo: Brush's kernels need the adapter's subgroup
  // support and large storage limits. Chrome advertises one experimental
  // feature that it then rejects at requestDevice(), so leave that one out.
  const requiredFeatures = [...adapter.features]
    .filter((feature) => feature !== "mappable-primary-buffers");
  const requiredLimits = {};
  for (const key in adapter.limits) {
    const value = adapter.limits[key];
    if (typeof value === "number") {
      requiredLimits[key] = value;
    }
  }

  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  return [adapter, device, device.queue];
}
"#)]
extern "C" {
    #[wasm_bindgen(js_name = requestBrushTestDevice)]
    fn request_brush_test_device() -> Promise;
}

pub struct HostDevice {
    pub adapter: JsValue,
    pub device: JsValue,
    pub queue: JsValue,
}

pub async fn request_host_device() -> Result<HostDevice, JsValue> {
    let handles = JsFuture::from(request_brush_test_device()).await?;
    let handles = Array::from(&handles);
    if handles.length() != 3 {
        return Err(JsValue::from_str(
            "requestBrushTestDevice returned an invalid handle tuple",
        ));
    }

    Ok(HostDevice {
        adapter: handles.get(0),
        device: handles.get(1),
        queue: handles.get(2),
    })
}
