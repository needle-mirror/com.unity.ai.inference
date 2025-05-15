# Tensor fundamentals in Inference Engine

In Inference Engine, models input and output data using multi-dimensional arrays called tensors. These tensors function similarly to tensors in machine learning frameworks, such as TensorFlow and PyTorch.

Tensors can have up to eight dimensions. A tensor with zero dimensions holds a single value called a scalar.

You can create the following types of tensor:

- [`Tensor<float>`](xref:Unity.InferenceEngine.Tensor`1), which stores the data as floats
- [`Tensor<int>`](xref:Unity.InferenceEngine.Tensor`1), which stores the data as integers

For more information, refer to [Create and modify tensors](do-basic-tensor-operations.md).

## Memory layout

Inference Engine stores tensors in memory in row-major order, so the values of the last dimension are stored adjacently.

Here's an example of a 2 × 2 × 3 tensor with values from `0` to `11` and how Inference Engine stores the tensor in memory.

![An example of a 2 × 2 × 3 tensor with the values 0 to 11](images/tensor-memory-layout.svg)

## Format

A model usually needs an input tensor in a certain format. For example, a model that processes images might need a 3-channel 240 × 240 image in one of the following formats:

- 1 × 240 × 240 × 3, where the order of the dimensions is batch size, height, width, channels (NHWC)
- 1 × 3 × 240 × 240, where the order of the dimensions is batch size, channels, height, width (NCHW)

If your tensor format doesn't match the expected input, the model might return incorrect or unexpected results.

You can use the Inference Engine functional API to convert a tensor to a different format. For more information, refer to [Edit a model](edit-a-model.md).

To convert a texture to a tensor in a specific format, refer to [Create input for a model](create-an-input-tensor.md).

## Memory location

Inference Engine stores tensor data in either graphics processing unit (GPU) memory or central processing unit (CPU) memory, depending on the [backend type](create-an-engine.md#back-end-types) you select. For example, if you use the [`BackendType.GPUCompute`](xref:Unity.InferenceEngine.BackendType.GPUCompute) backend type, tensors are typically stored in GPU memory.

Directly reading from and writing to tensor elements is only possible when the tensor is on the CPU, which can be slow. For better performance, it's more efficient to modify your model using the functional API.

To prevent Inference Engine from performing a blocking readback and upload, use a compute shader, Burst, or a native array. This lets you to read from and write to the tensor data directly in memory. For more information, refer to [Access tensor data directly](access-tensor-data-directly.md).

When you need to read an output tensor, perform an asynchronous readback. This prevents Inference Engine from blocking the main code thread while waiting for the model to finish and download the entire tensor. For more information, refer to [Read output from a model asynchronously](read-output-async.md).

## Additional resources

- [Understand the Inference Engine workflow](understand-inference-engine-workflow.md)
- [Understand models in Inference Engine](models-concept.md)
- [Create and modify tensors](do-basic-tensor-operations.md)