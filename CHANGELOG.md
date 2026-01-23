# Changelog
All notable changes to this package will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [2.5.0] - 2026-01-23

### Added
- `PyTorch` model import
- `LRN (Local Response Normalization)` operator implemented on all backends
- `3D MaxPool` and `AveragePool` operators implemented on all backends
- Sentis Importer: Allow users to specify dynamic dimensions as static on Sentis model import, same as we do for ONNX
- Tokenizer Additions
    - `Hugging Face` parser
	- Sequence decoder
	- Regex replace decoder
	- String split pre-tokenizer
	- Unigram Mapper
	- Byte-based substring feature to SubString
	- Padding: support "pad multiple of" option
	- Split pre-tokenizers: support "invert"
	- StripAccents normalizer
	- Rune split pre-tokenizer
	- Strip normalizer
	- WordLevel model
	- WhitespaceSplit pre-tokenizer
	- Metaspace pre-tokenizer and decoder
	- Whitespace pre-tokenizer
	- NMT normalizer
	- Punctuation pre-tokenizer
	- Digits pre-tokenizer
	- CharDelimiterSplit pre-tokenizer
	- BPE decoder

### Changed
- Model Visualizer: Async loading of model
- Model Visualizer: updating com.unity.dt.app-ui to 1.3.3
- Resize operator on CPU no longer uses main (mono) thread path
- All model converters use switch-case instead of if-else cascade
- Migrate Mono APIs to CoreCLR-compatible APIs

### Fixed
- Editor crash when quitting in play mode
- Memory Leak in FuseConstantPass
- `Clip` operator improvement: no longer need CPU fallback for min/max parameters
- `Mod` operator fix: on some platform with float operands, could return incorrect value when one of them was 0
- Faulty optimization pass
- Fix in existing burst code for 2D pooling vectorization calculations
- `TopK` issue on `GPUCompute` when dimension is specified
- Fix source generator empty array
- Tokenizer Fixes
	- Special added token decoding condition
	- Fix added token whole word handling
	- Gpt2Splitter subtring length computation
	- Added vocabulary pre-tokenization.
	- ByteLevelDecoder empty-byte guard in string generation
	- DefaultDecoder: joining tokens with whitespace
	- BPE: fix merging, applying on each word instead of the whole string
	- DefaultPostProcessor: apply the proper type id
	- RobertaPostProcessor: fix attention and type id assignment
	- TemplatePostProcessor: fix type id assignment
	- Assign default type id to sequences
	- Better surrogate characters support
	- Fix ByteFallback: inserting the right amount of \ufffd char
	- Fix BertPreTokenizer
	- Default model determination based of chain of responsibility

## [2.4.1] - 2025-10-31

### Fixed
- Small error in documentation preventing user manual publication

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
