# Samples

The Sentis package includes samples to help you learn and use the API.

The following samples are available:

- [Sample projects](#sample-projects) from the Sentis GitHub
- [Sample scripts](#sample-scripts) from the Package Manager

Validated models are also available to use in your project. To understand and download available models, refer to [Supported models](supported-models.md).

## Sample projects

Full sample projects are available on GitHub to demonstrate various Sentis use cases.

To explore these projects:

* Visit the [Sentis samples](https://github.com/Unity-Technologies/sentis-samples) GitHub repository.
* Each project includes setup instructions, and some feature a video walkthrough in the `README` file.

## Sample scripts

Use the sample scripts to implement specific features in your own project.

To find the sample scripts, follow these steps:

1. Go to **Window** > **Package Manager**, and select **Sentis** from the package list.
2. Select **Samples**.
3. To import a sample folder into your project, select **Import**.

   Unity creates a `Samples` folder in your project and adds the selected sample script.

The following table describes the available samples:

| Sample folder | Description |
|---------------|-------------|
| **Convert tensors to textures** | Examples of converting tensors to textures. For more information, refer to [Use output data](use-model-output.md). |
| **Convert textures to tensors** | Examples of converting textures to tensors. For more information, refer to [Create input for a model](create-an-input-tensor.md). |
| **Copy a texture tensor to the screen** | An example of using [`TextureConverter.RenderToScreen`](xref:Unity.InferenceEngine.TextureConverter.RenderToScreen*) to copy a texture tensor to the screen. For more information, refer to [Use output data](use-model-output.md). |
| **Encrypt a model** | Example of serializing an encrypted model to disk using a custom editor window and loading that encrypted model at runtime. For more information, refer to [Encrypt a model](encrypt-a-model.md). |
| **Quantize a model** | Example of serializing a quantized model to disk using a custom editor window and loading that quantized model at runtime. For more information, refer to [Quantize a model](quantize-a-model.md). |
| **Read output asynchronously** | Examples of reading the output from a model asynchronously using compute shaders. For more information, refer to [Read output from a model asynchronously](read-output-async.md).                 |
| **Run a model a layer at a time** | An example of using [`ScheduleIterable`](xref:Unity.InferenceEngine.Worker.ScheduleIterable*) to run a model a layer a time. For more information, refer to [Run a model](run-a-model.md). |
| **Run a model**  | Examples of running models with different numbers of inputs and outputs. For more information, refer to [Run a model](run-a-model.md). |
| **Use the functional API with an existing model**  | An example of using the functional API to extend an existing model. For more information, refer to [Edit a model](edit-a-model.md).  |
| **Use a compute buffer**  | An example of using a compute shader to write data to a tensor on the GPU.  |
| **Use Burst to write data**  | An example of using Burst to write data to a tensor in the Job system. |
| **Use tensor indexing methods** | Examples of using tensor indexing methods to get and set tensor values. |

## Additional resources

- [Unity Discussions group for Sentis](https://discussions.unity.com/tag/sentis)
- [Understand the Sentis workflow](understand-sentis-workflow.md)
- [Sentis models](models-concept.md)
- [Tensor fundamentals in Sentis](tensor-fundamentals.md)
- [Supported models](supported-models.md)
