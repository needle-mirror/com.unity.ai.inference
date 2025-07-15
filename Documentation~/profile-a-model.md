# Profile a model

The performance of a model depends on the following factors:

- The complexity of the model
- Whether the model uses performance-heavy operators such as `Conv` or `MatMul`
- The features of the platform you run the model on, for example central processing unit (CPU) memory, graphics processing unit (GPU) memory, and number of cores
- Whether Inference Engine downloads data to CPU memory when you access a tensor

   For more information, refer to [Get output from a model](get-the-output.md).

## Profile a model in the Profiler window

To get performance information when you run a model, can use the following:

- [The Profiler window](https://docs.unity3d.com/Documentation/Manual/Profiler.html)
- [RenderDoc](https://docs.unity3d.com/Documentation/Manual/RenderDocIntegration.html), a third-party graphics debugger

The **Profiler** window displays each Inference Engine layer as a dropdown item in the **Module Details** panel. Open a layer to get a detailed timeline of the implementation of the layer.

When a layer implements methods that include **Download** or **Upload**, Inference Engine transfers data to or from the CPU or the GPU. This might slow down the model.

If your model runs slower than you expect, refer to the following links:

- For information about how the complexity of a model might affect performance, refer to [Understand models in Inference Engine](models-concept.md).
- For information about different types of worker, refer to [Create an engine to run a model](create-an-engine.md).

## Additional resources

- [Inspect a model](inspect-a-model.md)
- [How Inference Engine optimizes a model](models-concept.md#how-inference-engine-optimizes-a-model)
- [Supported ONNX operators](supported-operators.md)