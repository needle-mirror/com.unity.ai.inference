# Supported models

You can import open-source models into your Sentis project. Explore the following sections to understand the models Sentis supports and find an appropriate model for your project.

## Pre-trained models

[!include[](snippets/model-registry.md)]

### Models from Hugging Face

You can access validated AI models for use with Sentis from [Hugging Face](https://huggingface.co/models).

To browse and download models from Hugging Face navigate to the [Unity Hugging Face](https://huggingface.co/unity) space and select a model under the **Models** section.

Each model page includes a **How to Use** section with instructions for importing the model into your Unity project.

### ONNX models

You can also import models in the [ONNX format](https://github.com/onnx/models). Sentis supports most ONNX models with [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions) 7 to 15. Models with opset versions outside this range (for example, <7 or >15) might still import, but results can be unpredictable.

## Unsupported models

Sentis doesn't support the following:

- Models that use tensors with more than eight dimensions
- Sparse input tensors or constants
- String tensors
- Complex number tensors

Sentis also converts some tensor data types like booleans to floats or integers. This might increase the memory your model uses.

## Additional resources

- [Unity Hugging Face](https://huggingface.co/unity)
- [ONNX model zoo](https://github.com/onnx/models)
- [Sentis models](models-concept.md)
- [Import a model file](import-a-model-file.md)
- [Supported ONNX operators](supported-operators.md)
- [Supported LiteRT operators](supported-litert-operators.md)
