# Understand the Inference Engine workflow

To use Inference Engine to run a neural network in Unity, follow these steps:

1. Use the `Unity.InferenceEngine` namespace.
2. Load a neural network model file.
3. Create input for the model.
4. Create an inference engine (a worker).
5. Run the model with the input to compute a result (inference).
6. Get the result.

> [!TIP]
> Use the [Workflow example](workflow-example.md) to understand the workflow applied to a simple example.

## Use the `Unity.InferenceEngine` namespace

At the top of your script, include the `Unity.InferenceEngine` namespace as follows:

```
using Unity.InferenceEngine;
```

## Load a model

Inference Engine can import model files in [Open Neural Network Exchange](https://onnx.ai/) (ONNX) format. To load a model, follow these steps:

1. [Export a model to ONNX format from a machine learning framework](export-convert-onnx.md) or download an ONNX model from the Internet.
2. Add the model file to the `Assets` folder of the **Project** window.
3. Create a runtime model in your script as follows:

```
ModelAsset modelAsset = Resources.Load("model-file-in-assets-folder") as ModelAsset;
var runtimeModel = ModelLoader.Load(modelAsset);
```
You can also add `public ModelAsset modelAsset` as a public variable in GameObjects. In this case, specify the model manually.

For more information, refer to [Import a model file](import-a-model-file.md).

## Create input for the model

Use the [Tensor](xref:Unity.InferenceEngine.Tensor) API to create a tensor with data for the model. You can convert an array or a texture to a tensor. For example:

```
// Convert a texture to a tensor
Texture2D inputTexture = Resources.Load("image-file") as Texture2D;
Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 4, inputTexture.height, inputTexture.width));
TextureConverter.ToTensor(inputTexture, inputTensor);
// Convert an array to a tensor
int[] array = new int[] {1,2,3,4};
Tensor<int> inputTensor = new Tensor<int>(new TensorShape(4), array);
```

For more information, refer to [Create input for a model](create-an-input-tensor.md).

## Create an inference engine (a worker)

In Inference Engine, a worker is the inference engine. You create a worker to break down the model into executable tasks, run the tasks on the central processing unit (CPU) memory or graphics processing unit (GPU), and retrieve the result.

For example, the following creates a worker that runs on the GPU using Inference Engine compute shaders:

```
Worker worker = new Worker(runtimeModel, BackendType.GPUCompute);
```

For more information, refer to [Create an engine](create-an-engine.md).

## Schedule the model

To run the model, use the [`Schedule`](xref:Unity.InferenceEngine.Worker.Schedule*) method of the worker object with the input tensor.

```
worker.Schedule(inputTensor);
```
Inference Engine schedules the model layers on the given backend. Because processing is asynchronous, some tensor operations might still be running after this call.

For more information, refer to [Run a model](run-a-model.md).

## Get the output

You can use methods, such as [`PeekOutput`](xref:Unity.InferenceEngine.Worker.PeekOutput*), to get the output data from the model. For example:

```
Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;
```

For more information, refer to [Get output from a model](get-the-output.md).

## Additional resources

- [Workflow example](workflow-example.md)
- [Samples](package-samples.md)
- [Unity Discussions group for Inference Engine](https://discussions.unity.com/c/10)
- [Understand models in Inference Engine](models-concept.md)
- [Tensor fundamentals in Inference Engine](tensor-fundamentals.md)