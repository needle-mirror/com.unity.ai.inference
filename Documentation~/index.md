# Inference Engine overview

Inference Engine is a neural network inference library for Unity. It lets you import trained neural network models into Unity and run them in real-time with your target device’s compute resources, such as central processing unit (CPU) or graphics processing unit (GPU).

Inference Engine supports real-time applications across all Unity-supported platforms.

The package is officially released and available to all Unity users through the **Package Manager**.

> [!TIP]
> Prior experience with machine learning frameworks like [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/) is helpful, but not required. It can make it easier to understand how to work with models in Inference Engine.

|Section|Description|
|-|-|
|[Get started](get-started.md)|Learn how to install Inference Engine, explore sample projects, and understand the Inference Engine workflow.|
|[Create a model](create-a-model.md)|Create a runtime model by importing an ONNX model file or using the Inference Engine model API.|
|[Run a model](run-an-imported-model.md)|Create input data for a model, create an engine to run the model, and get output.|
|[Use Tensors](use-tensors.md)|Learn how to get, set, and modify input and output data.|
|[Profile a model](profile-a-model.md)|Use Unity tools to profile the speed and performance of a model.|

## Supported platforms

Inference Engine supports [all Unity runtime platforms](https://docs.unity3d.com/Documentation/Manual/PlatformSpecific.html).

Performance might vary based on:
* Model operators and complexity
* Hardware and software platform constraints of your device
* Type of engine used

   For more information, refer to [Models](models-concept.md) and [Create an engine](create-an-engine.md).

## Supported model types

Inference Engine supports most models in Open Neural Network Exchange (ONNX) format with an [opset version](https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions) between 7 and 15. For more information, refer to [Supported models](supported-models.md) and [Supported ONNX operators](supported-operators.md).

## Places to find pre-trained models

[!include[](snippets/model-registry.md)]

## Additional resources

- [Sample scripts](package-samples.md)
- [Unity Discussions group](https://discussions.unity.com/tag/ai)
- [Understand the Inference Engine workflow](understand-inference-engine-workflow.md)
- [Understand models in Inference Engine](models-concept.md)
- [Tensor fundamentals in Inference Engine](tensor-fundamentals.md)
