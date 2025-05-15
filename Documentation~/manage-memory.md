# Manage memory with tensors

As an Inference Engine user, it's important to call [`Dispose`](xref:Unity.InferenceEngine.Worker.Dispose) on any workers and tensors you instantiate. Additionally, ensure you call `Dispose` on cloned output tensors returned from the [`ReadbackAndClone`](Unity.InferenceEngine.Tensor.ReadbackAndClone*) method.

> [!NOTE]
> You must call `Dispose` to free up graphics processing unit (GPU) resources.

For example:

```
void OnDestroy()
{
    worker?.Dispose();

    // Assuming model with multiple inputs that were passed as a array
    foreach (var input in inputs)
    {
        input.Dispose();
    }
}
```

When you get a handle to a tensor from a worker using the [`PeekOutput`](xref:Unity.InferenceEngine.Worker.PeekOutput*) method, the memory allocator remains responsible for that memory. You don't need to call `Dispose` on it. For more information, refer to [Get output from a model](get-the-output.md).

## Compute buffer size limit

When working with tensors in Inference Engine, the size of the compute buffer is subject to certain limits. While these limits can vary depending on hardware and platform specifications, the following general guidelines apply:

* **Maximum tensor size**: A tensor can have no more than 2^31 elements due to internal indexing limitations.
* **Auxiliary resource allocations**: Some operators might require additional auxiliary resources, which can further reduce the maximum allowable size of tensors in specific scenarios. These resources depend on the input tensor size and the type of operation.

## Additional resources

- [Profile a model](profile-a-model.md)
- [Create an engine to run a model](create-an-engine.md)
- [Create and modify tensors](do-basic-tensor-operations.md)