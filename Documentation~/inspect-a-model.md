# Inspect a model

You can inspect a runtime model to view its inputs, outputs, and layers. This helps debug, understand the model structure, or verify model data after import.

## Get model inputs

To inspect a model’s inputs, use the [`inputs`](xref:Unity.InferenceEngine.Model.inputs) property of the runtime model. This returns a list of all input tensors, including their names and shapes.

### Example

```
using UnityEngine;
using System.Collections.Generic;
using Unity.InferenceEngine;

public class GetModelInputs : MonoBehaviour
{
    public ModelAsset modelAsset;

    void Start()
    {
        Model runtimeModel = ModelLoader.Load(modelAsset);

        List<Model.Input> inputs = runtimeModel.inputs;

        // Loop through each input
        foreach (var input in inputs)
        {
            // Log the name of the input, for example Input3
            Debug.Log(input.name);

            // Log the tensor shape of the input, for example (1, 1, 28, 28)
            Debug.Log(input.shape);
        }
    }
}
```

Input dimensions can be fixed or dynamic. For more information, refer to [Model inputs](models-concept.md#model-inputs).

## Get model outputs

Use the [`outputs`](xref:Unity.InferenceEngine.Model.outputs) property of the runtime model to get the names of the output layers of the model.

### Example

```
List<Model.Output> outputs = runtimeModel.outputs;

// Loop through each output
foreach (var output in outputs)
{
    // Log the name of the output
    Debug.Log(output.name);
}
```

## Get layers and layer properties

Use the [`layers`](xref:Unity.InferenceEngine.Model.layers) property of the runtime model to get the neural network layers in the model. Each layer includes input and output identifiers, and other relevant properties.

## Open a model as a graph

To open an ONNX model as a graph, follow these steps:

1. Install [Netron](https://github.com/lutzroeder/netron), a third-party viewer for neural networks.
2. In the Unity Editor, you can open a model in Netron in one of the following ways:

   * Double-click the model asset in the **Project** window.
   * Select the model asset, then select **Open** in the **Model Asset Import Settings** window.

## Additional resources

- [Profile a model](profile-a-model.md)
- [Tensor fundamentals](tensor-fundamentals.md)
- [Supported ONNX operators](supported-operators.md)