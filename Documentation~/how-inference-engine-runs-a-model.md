# How Inference Engine runs a model

Inference Engine runs optimized tensor operations across multiple threads on the central processing unit (CPU). It also runs operations in parallel on the graphics processing unit (GPU) through compute or pixel shaders.

When the [worker](create-an-engine.md) schedules a model, it processes each layer sequentially. For each layer, it schedules the corresponding operation on the input tensors to compute one or more output tensors.

The [`BackendType`](xref:Unity.InferenceEngine.BackendType) you choose determines how and when the worker performs each operation.

The following tables defines the types of backend available:

|`BackendType`|Runs on|Description|
|-|-|-|
|[`CPU`](xref:Unity.InferenceEngine.BackendType.CPU)|CPU, using [Burst](https://docs.unity3d.com/Packages/com.unity.burst@latest/)|Inference Engine creates, sets up, and [schedules](https://docs.unity3d.com/Manual/JobSystemCreatingJobs.html) a Burst job for the operation. If the input tensors are output from other jobs, the worker creates a [job dependency](https://docs.unity3d.com/Manual/JobSystemJobDependencies.html) to ensure correct inference without blocking.|
|[`GPUCompute`](xref:Unity.InferenceEngine.BackendType.GPUCompute)|GPU, using Inference Engine compute shaders with [command buffers](xref:UnityEngine.Rendering.CommandBuffer)|Inference Engine creates, sets up, and adds a compute shader the command buffer. Inference Engine runs the command buffer to perform the operations.|
|[`GPUPixel`](xref:Unity.InferenceEngine.BackendType.GPUPixel)|GPU, using Inference Engine pixel shaders|Inference Engine creates, sets up, and runs a pixel shader by blitting.|

## Tensor outputs

When Inference Engine returns a tensor object, the tensor’s values might not yet be fully calculated. This is because some scheduled work might still be pending. This deferred processing lets you schedule additional tensor operations without waiting for earlier tasks to finish.

To complete the processing of the work on the backend, move the tensor data to the CPU.

Call [`ReadbackAndClone`](xref:Unity.InferenceEngine.Tensor.ReadbackAndClone*) to get a CPU copy of the tensor. This is a blocking call that waits synchronously for the backend to finish processing and return the data. Note that this process can be slow, especially when reading back from the GPU.

To avoid blocking calls on the main thread, use one of the following:
* [`ReadbackAndCloneAsync`](xref:Unity.InferenceEngine.Tensor.ReadbackAndCloneAsync*) for an `Awaitable` version of this method.
* [`ReadbackRequest`](xref:Unity.InferenceEngine.Tensor.ReadbackRequest*) to trigger an async download. When [`IsReadbackRequestDone`](xref:Unity.InferenceEngine.Tensor.IsReadbackRequestDone*) return true, [`ReadbackAndClone`](xref:Unity.InferenceEngine.Tensor.ReadbackAndClone*) is immediate.

To move the tensor data to the CPU with a non-blocking, non-destructive download, use one of the following:

* [`ReadbackRequest`](xref:Unity.InferenceEngine.Tensor.ReadbackRequest*) on your tensor.
* [`ReadbackAndCloneAsync`](xref:Unity.InferenceEngine.Tensor.ReadbackAndCloneAsync*) on your tensor.
* [`DownloadToNativeArray`](xref:Unity.InferenceEngine.Tensor`1.DownloadToNativeArray*) or [`DownloadToArray`](xref:Unity.InferenceEngine.Tensor`1.DownloadToArray*).
* [`Download`](xref:Unity.InferenceEngine.ITensorData.Download*) on the [`dataOnBackend`](xref:Unity.InferenceEngine.Tensor.dataOnBackend) of your tensor.

## CPU fallback

Inference Engine doesn't support all operator on every backend type. For more information, refer to [Supported ONNX operators](supported-operators.md).

If Inference Engine supports an operator on the CPU but not the GPU, Inference Engine might automatically fall back to running on the CPU. This requires Inference Engine to sync with the GPU and read back the input tensors to the CPU. If a GPU operation uses the output tensor, Inference Engine completes the operation and uploads the tensor to the GPU.

If a model has many layers that use CPU fallback, Inference Engine might spend significant time to upload and read back from the CPU. This can impact the performance of your model. To reduce CPU fallback, build the model so that Inference Engine runs effectively on your chosen backend type or use the CPU backend.

Sometimes, Inference Engine needs to read tensor data on the main thread to schedule operations. For example:

* The `shape` input tensor for an `Expand` operation.
* The `axes` input for a `Reduce` operation.

These input tensors might be outputs from other operations. During model input handling, the engine automatically optimizes and determines which tensors must run on the CPU, regardless of the selected backend.

## Additional resources

- [Use output data](use-model-output.md)
- [Read output from a model asynchronously](read-output-async.md)