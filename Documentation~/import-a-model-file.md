# Import a model file

To import an ONNX or LiteRT model file into your Unity project, drag the file from your computer into the `Assets` folder of the **Project** window.

If your model has external weights files, put them in the same directory as the model file so that Sentis imports them automatically.

For more information on supported model formats, refer to [Supported models](supported-models.md).

## Create a runtime model

To use an imported model in your scene, use [`ModelLoader.Load`](xref:Unity.InferenceEngine.ModelLoader.Load*) to create a runtime [`Model`](xref:Unity.InferenceEngine.Model) object.

```
using UnityEngine;
using Unity.InferenceEngine;

public class CreateRuntimeModel : MonoBehaviour
{
    public ModelAsset modelAsset;
    Model runtimeModel;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
    }
}
```

After the model is loaded, you can [create an engine to run a model](create-an-engine.md).

## Additional resources

- [How Sentis optimizes a model](models-concept.md#how-sentis-optimizes-a-model)
- [Export an ONNX file from a machine learning framework](export-convert-onnx.md)
- [Export a LiteRT file from a machine learning framework](export-convert-litert.md)
- [Model Asset Inspector](model-asset-inspector.md)
- [Supported models](supported-models.md)
