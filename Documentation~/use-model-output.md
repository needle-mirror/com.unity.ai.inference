# Use output data

After you [get the output from a model](get-the-output.md) as a tensor, you can post-process the data to use it in your project.

## Download to a CPU tensor

Use [`ReadbackAndClone`](xref:Unity.InferenceEngine.Tensor.ReadbackAndClone) or [`ReadbackAndCloneAsync`](xref:Unity.InferenceEngine.Tensor.ReadbackAndCloneAsync*) to move a tensor on the graphics processing unit (GPU) to the central processing unit (CPU) to read it.

For best practices on how to do this efficiently, refer to [read the tensor data asynchronously](read-output-async.md).

For example:

```
var outputTensor = worker.Schedule(inputTensor).PeekOutput() as Tensor<float>;
var cpuTensor = outputTensor.ReadbackAndClone();
```
The returned tensor is a CPU-based, read-writable copy of the output tensor.

For details on how to index and access the tensor, refer to [tensor fundamentals](tensor-fundamentals.md) and [access tensor data directly](access-tensor-data-directly.md).

## Convert to a render texture

To convert a tensor to a render texture, use [`TextureConverter.RenderToTexture`](xref:TextureConverter.RenderToTexture*) to write tensor data to an existing render texture.

When you use [`TextureConverter.RenderToTexture`](xref:Unity.InferenceEngine.TextureConverter.ToTexture*), if the dimensions of the tensor doesn't match those of the render texture, Inference Engine makes the following adjustments:

- Samples the tensor linearly if the dimensions don't match.
- Removes channels from the end if the render texture has fewer channels than the tensor.
- Sets values in RGB channels to `0` and values in the alpha channel to `1` if the render texture has more channels than the tensor.

For working examples, refer to the `Convert tensors to textures` example in the [sample scripts](package-samples.md).

### Example

```
// Instantiate the render texture
public RenderTexture rt = new RenderTexture(24, 32, 0, RenderTextureFormat.ARGB32);
Worker worker;
Tensor inputTensor;

void Start()
{
    // Get the output of the model as a tensor
    Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

    // Convert the tensor to a texture and store it in the render texture
    TextureConverter.RenderToTexture(outputTensor, rt);
}
```

> [!NOTE]
> Avoid unnecessary resource allocation and don't re-allocate tensors or textures every frame.

## Copy to the screen

To copy an output tensor to the screen, follow these steps:

1. Set the [`Camera.targetTexture`](xref:UnityEngine.Camera.targetTexture) property of [`Camera.main`](xref:UnityEngine.Camera.main) to null.
2. Create a script and attach it to the Camera.
3. In the script, use [`TextureConverter.RenderToScreen`](xref:Unity.InferenceEngine.TextureConverter.RenderToScreen*) in an event function, such as [`OnRenderImage`](xref:MonoBehaviour.OnRenderImage).

If the image is too bright, the output tensor might be using values from `0` to `255` instead of `0` to `1`. You can use [Edit a model](edit-a-model.md) to remap the values in the output tensor before calling `RenderToScreen`.

The following script uses a model to change a texture, then copies the result to the screen. Set `modelAsset` to one of the [style transfer models](https://github.com/onnx/models/tree/main/validated/vision/style_transfer/fast_neural_style) from ONNX and `inputImage` to a texture. [Check the Texture import settings](convert-texture-to-tensor.md) to make sure the texture matches the shape and layout the model needs.

```
using UnityEngine;
using Unity.InferenceEngine;

public class StyleTransfer : MonoBehaviour
{
    public ModelAsset modelAsset;
    public Model runtimeModel;
    public Texture2D inputImage;
    public RenderTexture outputTexture;

    Worker worker;
    Tensor<float> inputTensor;

    void Start()
    {
        var sourceModel = ModelLoader.Load(modelAsset);
        var graph = new FunctionalGraph();
        var input = graph.AddInput(sourceModel, 0);
        var output = Functional.Forward(sourceModel, input)[0];
        // rescale output of source model
        output /= 255f;
        graph.AddOutput(output);
        var runtimeModel = graph.Compile();

        worker = new Worker(runtimeModel, BackendType.GPUCompute);
        inputTensor = new Tensor<float>(new TensorShape(1, 3, 256, 256));
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        // Create the input tensor from the texture
        TextureConverter.ToTensor(inputImage, inputTensor, new TextureTransform());

        // Run the model and get the output as a tensor
        worker.Schedule(inputTensor);
        Tensor<float> outputTensor = worker.PeekOutput() as Tensor<float>;

        // Copy the rescaled tensor to the screen as a texture
        TextureConverter.RenderToScreen(outputTensor);
    }

    void OnDisable()
    {
        worker.Dispose();
    }
}
```

When using Universal Render Pipeline (URP) or the High-Definition Render Pipeline (HDRP), call [`RenderToScreen`](xref:Unity.InferenceEngine.TextureConverter.RenderToScreen*) in the [`RenderPipelineManager.endFrameRendering`](UnityEngine.Rendering.RenderPipelineManager.endFrameRendering(System.Action`2<UnityEngine.Rendering.ScriptableRenderContext,UnityEngine.Camera[]>)) or [`RenderPipelineManager.endContextRendering`](xref:UnityEngine.Rendering.RenderPipelineManager.endContextRendering(System.Action2<UnityEngine.Rendering.ScriptableRenderContext,System.Collections.Generic.List1<UnityEngine.Camera>>)) callbacks. For more information, refer to [Rendering.RenderPipelineManager](xref:UnityEngine.Rendering.RenderPipelineManager).

For an example, refer to the `Copy a texture tensor to the screen` example in the [sample scripts](package-samples.md).

## Additional resources

- [Get output from a model](get-the-output.md)
- [Create and modify tensors](do-basic-tensor-operations.md)