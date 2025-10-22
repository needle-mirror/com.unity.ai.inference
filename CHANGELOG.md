# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [2.4.0] - 2025-10-22

### Added
- LiteRT model import
- Tokenization API
- STFT and DFT ONNX operators
- BlackmanWindow, HammingWindow, HannWindow and MelWeightMatrix ONNX operators
- BitwiseAnd, BitwiseOr, BitwiseXor, BitwiseNot ONNX operators and functional methods
- AsStrided, Atan2, Expm1, Log10, Log1p, Log2, Rsqrt, Trunc, ReduceVariance, Diagonal layers, functional methods and optimizer passes
- NotEqual, FloorDiv, TrueDiv layers and LiteRT operators

### Changed
- Renamed Inference Engine to Sentis in package name and documentation
- Improved model import time for ONNX models
- ONNX model import operator order now consistent with the original model
- Improved optimization passes to reduce operator count in imported models
- Improved visualizer loading times and consistency in displaying attributes
- ScatterND operator can now run on much larger tensors, enabling new models
- ScatterND operator now allows negative indices
- ONNX model outputs that are not connected to any inputs are no longer incorrectly pruned
- Improve model import warning and error display in the inspector

### Fixed
- Small errors in documentation
- Faulty optimization passes that could lead to inference issues
- Memory leaks on model constants
- Non-matching ProfilerMarker calls
- Issues in CPU callback which could lead to incorrect inference on some models
- Enable missing modes for GridSample and Upsample operators

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
