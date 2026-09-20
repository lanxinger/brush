//! Running brush's own kernels under burn's `Fusion` backend.
//!
//! A custom op takes concrete `CubeTensor`s once the fusion stream reaches
//! it, so the elementwise ops on either side still fuse among themselves.

use burn::tensor::{DType, Shape};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_fusion::FusionHandle;
use burn_fusion::custom::{
    CustomOpIr, HandleContainer, OperationFn, OperationIr, OperationOutput, StreamId, TensorIr,
};

/// Handle container a fusion custom op executes against.
pub type FusionHandles = HandleContainer<FusionHandle<FusionCubeRuntime>>;
/// A tensor living in the fusion stream.
pub type FusionTensor = burn_fusion::FusionTensor<FusionCubeRuntime>;
/// The fusion client that owns the stream.
pub type FusionClient = burn_fusion::Client<FusionCubeRuntime>;

/// Register a concrete-backend function as a custom op on the fusion stream.
///
/// `inputs` are handed to the op once the stream reaches it, and each
/// `(shape, dtype)` in `outputs` becomes a fresh handle the op must fill in
/// through the handle container. The op gets the description so it can look
/// both up by id with `desc.as_fixed()`.
pub fn register_custom<const N: usize, const M: usize, F>(
    client: &FusionClient,
    name: &'static str,
    inputs: [FusionTensor; N],
    outputs: [(Shape, DType); M],
    op: F,
) -> [FusionTensor; M]
where
    F: Fn(&CustomOpIr, &mut FusionHandles) + Send + Sync + 'static,
{
    let outputs =
        outputs.map(|(shape, dtype)| TensorIr::uninit(client.create_empty_handle(), shape, dtype));
    let desc = CustomOpIr::new(name, &inputs.map(|t| t.into_ir()), &outputs);
    let run = {
        let desc = desc.clone();
        OperationFn(move |handles: &mut FusionHandles| {
            op(&desc, handles);
            Ok(())
        })
    };
    client
        .register(StreamId::current(), OperationIr::Custom(desc), run)
        .outputs()
}
