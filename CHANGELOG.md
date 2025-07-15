# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [2.3.0] - 2025-07-15

### Added
- Model Visualizer for inspecting models as node-based graphs inside the Unity Editor
- Support for `Tensor<int>` input for `GatherND` operator on `GPUPixel` backend
- Support for `Tensor<int>` input for the base of the `Pow` operator on all backends
- Support for the `group` and `dilations` arguments for the `ConvTranspose` operator on all backends
- Support for `value_float`, `value_floats`, `value_int` and `value_ints` values in ONNX `Constant` operators

### Changed
- Optimized single-argument operators on `CPU` backend
- Optimized deserialization of models to avoid reflection at runtime

### Fixed
- Einsum operator now works correctly on fallback path

## [2.2.1] - 2025-05-28

### Fixed
- Issue with incorrect TensorShape in Conv layer when dilations are greater than 1 and auto-padding is used
- Incorrect Third Party Notices

## [2.2.0] - 2025-05-15

### Added
- First version of Inference Engine
