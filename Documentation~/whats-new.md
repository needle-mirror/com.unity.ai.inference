# What's new in Inference Engine 2.3

This is a summary of the changes from Inference Engine 2.2 to Inference Engine 2.3.

## Added

- Model Visualizer for inspecting models as node-based graphs inside the Unity Editor.
- `GatherND` and `Pow` operators now support `Tensor<int>` inputs more widely.
- `ConvTranspose` and `Constant` operators now support more input arguments.


# What's new in Inference Engine 2.2

Inference Engine is the new name for the [Sentis package](https://docs.unity3d.com/Packages/com.unity.sentis@latest/manual/index.html).

This is a summary of the changes from Sentis 2.1 to Inference Engine 2.2.

For information on how to upgrade, refer to the [Upgrade Guide](upgrade-guide.md).

## Added

- Dynamic input shape dimensions support at import time for better model optimization.
- Custom input and output names for models created with the functional API.
- The model stores the shapes and data types of intermediate and output tensors and displays them in the **Model Asset Inspector**.
- New `Mish` operator.
- Improved shape inference for model optimization.

## Updated

- `ScatterElements` and `ScatterND` operators now support `min` and `max` reduction modes.
- `DepthToSpace` and `SpaceToDepth` now support integer tensors.
- `TopK` supports integer tensors.
- `Functional.OneHot` now allows negative indices.
- `RoiAlign` now supports the `coordinate_transformation_mode` parameter.
- Reduction operators return correct results when reducing a tensor along an axis of length 0.
- `Reshape` operator can now infer unknown dimensions even when reshaping a length 0 tensor like in PyTorch.
- Improved documentation for **Model Asset Inspector**.

## Removed

- Obsolete Unity Editor menu items.
- Slow CPU support for 4-dimensional and higher `Convolution` layers.

## Fixed

- Out-of-bounds errors for certain operators on `GPUCompute` backend.
- The `TextureConverter` methods now correctly performs sRGB to RGB conversions.
- Incorrect graph optimizations for certain models.
- Issues with negative padding values in pooling and convolutions.
- Accurate handling of large and small integer values in the `GPUPixel` backend.
- Proper destruction of allocated render textures in the `GPUPixel` backend.
- `LeakyRelu` now supports `alpha` greater than 1 on all platforms.
- Fixed Async behaviour for CPU tensor data.
