#![cfg(target_family = "wasm")]

mod support;

use brush_js::{BrushApp, BrushMessageKind};
use support::request_host_device;
use wasm_bindgen::{JsCast, prelude::*};
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::wasm_bindgen_test;
use web_sys::js_sys::{Function, Promise, Uint8Array};

wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen(inline_js = r#"
async function writeFile(directory, name, bytes) {
  const file = await directory.getFileHandle(name, { create: true });
  const writable = await file.createWritable();
  await writable.write(bytes);
  await writable.close();
}

export async function createBrushTestDataset(transforms, initPly, image) {
  const root = await navigator.storage.getDirectory();
  const name = `brush-host-device-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const directory = await root.getDirectoryHandle(name, { create: true });
  const train = await directory.getDirectoryHandle("train", { create: true });
  await writeFile(directory, "transforms.json", transforms);
  await writeFile(directory, "init.ply", initPly);
  await writeFile(train, "r_0.png", image);
  return directory;
}

export function brushOneStepConfig() {
  return async (config) => {
    config["total-train-iters"] = 1;
    config["max-frames"] = 1;
    config["max-resolution"] = 50;
    config["sh-degree"] = 0;
    config["refine-every"] = 200;
    config["growth-stop-iter"] = 1;
    config["eval-every"] = 1000;
    config["export-every"] = 1000;
    config["rerun-enabled"] = false;
    return config;
  };
}

export async function exerciseBrushBuffers(device, transforms, shCoeffs, rawOpacities) {
  device.pushErrorScope("validation");
  let thrown = null;
  let readback = null;
  let output = null;

  try {
    const shader = device.createShaderModule({ code: `
      @group(0) @binding(0) var<storage, read> transforms: array<f32>;
      @group(0) @binding(1) var<storage, read> sh_coeffs: array<f32>;
      @group(0) @binding(2) var<storage, read> raw_opacities: array<f32>;
      @group(0) @binding(3) var<storage, read_write> output: array<f32>;

      @compute @workgroup_size(1)
      fn main() {
        output[0] = transforms[0] + sh_coeffs[0] + raw_opacities[0];
      }
    ` });
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module: shader, entryPoint: "main" },
    });
    output = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    readback = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: transforms } },
        { binding: 1, resource: { buffer: shCoeffs } },
        { binding: 2, resource: { buffer: rawOpacities } },
        { binding: 3, resource: { buffer: output } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  } catch (error) {
    thrown = error;
  }

  const validationError = await device.popErrorScope();
  if (thrown) throw thrown;
  if (validationError) throw validationError;

  await readback.mapAsync(GPUMapMode.READ);
  const value = new Float32Array(readback.getMappedRange())[0];
  readback.unmap();
  readback.destroy();
  output.destroy();
  if (!Number.isFinite(value)) {
    throw new Error(`host-device buffer dispatch returned ${value}`);
  }
}

export async function removeBrushTestDataset(directory) {
  const root = await navigator.storage.getDirectory();
  await root.removeEntry(directory.name, { recursive: true });
}
"#)]
extern "C" {
    #[wasm_bindgen(js_name = createBrushTestDataset)]
    fn create_brush_test_dataset(
        transforms: &Uint8Array,
        init_ply: &Uint8Array,
        image: &Uint8Array,
    ) -> Promise;

    #[wasm_bindgen(js_name = brushOneStepConfig)]
    fn brush_one_step_config() -> Function;

    #[wasm_bindgen(js_name = exerciseBrushBuffers)]
    fn exercise_brush_buffers(
        device: &JsValue,
        transforms: &JsValue,
        sh_coeffs: &JsValue,
        raw_opacities: &JsValue,
    ) -> Promise;

    #[wasm_bindgen(js_name = removeBrushTestDataset)]
    fn remove_brush_test_dataset(directory: &JsValue) -> Promise;
}

fn bytes(value: &[u8]) -> Uint8Array {
    Uint8Array::from(value)
}

#[wasm_bindgen_test]
async fn host_device_owns_training_buffers_and_rejects_replacement() {
    let host = request_host_device()
        .await
        .expect("request host WebGPU device");
    let app = BrushApp::new();

    app.init_existing(
        host.adapter.clone(),
        host.device.clone(),
        host.queue.clone(),
    )
    .expect("register host device");
    app.init_existing(
        host.adapter.clone(),
        host.device.clone(),
        host.queue.clone(),
    )
    .expect("same host device should be idempotent");
    app.init()
        .await
        .expect("internal initialization after host registration");

    let directory = JsFuture::from(create_brush_test_dataset(
        &bytes(include_bytes!(
            "../../brush-c/tests/data/test_dataset/transforms.json"
        )),
        &bytes(include_bytes!(
            "../../brush-c/tests/data/test_dataset/init.ply"
        )),
        &bytes(include_bytes!(
            "../../brush-c/tests/data/test_dataset/train/r_0.png"
        )),
    ))
    .await
    .expect("create OPFS test dataset");
    let handle = directory
        .clone()
        .dyn_into::<web_sys::FileSystemDirectoryHandle>()
        .expect("OPFS returned a directory handle");

    let training = app.start_training_from_directory(handle, brush_one_step_config());
    let messages = training
        .train_steps(1)
        .await
        .expect("run one training step");
    assert!(
        messages
            .iter()
            .any(|message| matches!(message.kind(), BrushMessageKind::TrainStep)),
        "one-step run produced no TrainStep message"
    );

    let splats = training
        .current_splats()
        .expect("training produced no splats");
    let buffers = splats.buffers().expect("WebGPU splat buffers");
    JsFuture::from(exercise_brush_buffers(
        &host.device,
        &buffers.transforms(),
        &buffers.sh_coeffs(),
        &buffers.raw_opacities(),
    ))
    .await
    .expect("bind and submit exported buffers on the host device");

    let replacement = request_host_device()
        .await
        .expect("request replacement WebGPU device");
    let error = app
        .init_existing(replacement.adapter, replacement.device, replacement.queue)
        .expect_err("a different host device must be rejected");
    assert!(
        error
            .as_string()
            .is_some_and(|message| message.contains("different GPU device"))
    );

    training.cancel();
    JsFuture::from(remove_brush_test_dataset(&directory))
        .await
        .expect("remove OPFS test dataset");
}
