# Create an engine to run a model

To run a model, you need to create a worker. A worker is the engine that breaks the model down into runnable tasks. It schedules the tasks to run on a backend, such as the graphics processing unit (GPU) or central processing unit (CPU).

## Create a Worker

Use [`new Worker(...)`](xref:Unity.InferenceEngine.Worker.#ctor*) to create a worker. You must specify a backend type, which tells Inference Engine where to run the worker and a [runtime model](import-a-model-file.md#create-a-runtime-model).

For example, the following code creates a worker that runs on the GPU with Inference Engine compute shaders.

```
using UnityEngine;
using Unity.InferenceEngine;

public class CreateWorker : MonoBehaviour
{
    ModelAsset modelAsset;
    Model runtimeModel;
    Worker worker;

    void Start()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
    }
}
```

## Backend types

Inference Engine provides CPU and GPU backend types. To understand how Inference Engine runs operations with the different backends, refer to [How Inference Engine runs a model](how-inference-engine-runs-a-model.md).

If a backend type doesn't support an Inference Engine layer in a model, the worker will assert. For more information, refer to [Supported ONNX operators](supported-operators.md).

| BackendType | Usage |
| ----------- | ----- |
| [`BackendType.CPU`](xref:Unity.InferenceEngine.BackendType.CPU)               | - Faster than GPU for small models or when inputs/outputs are on the CPU.<br>- On WebGL, Burst compiles to WebAssembly, which may result in slower performance. For more information, refer to [Getting started with WebGL development](https://docs.unity3d.com/Documentation/Manual/webgl-gettingstarted.html).  |
| [`BackendType.GPUCompute`](xref:Unity.InferenceEngine.BackendType.GPUCompute) | - Generally the fastest backend for most models.<br>- Avoids expensive data transfer when outputs remain on the GPU.<br>- Uses [DirectML](https://learn.microsoft.com/en-us/windows/ai/directml/dml) for inference acceleration when running on DirectX12-supported platforms. For more information, refer to [Supported ONNX operators](supported-operators.md). |
| [`BackendType.GPUPixel`](xref:Unity.InferenceEngine.BackendType.GPUPixel)     | - Use only on platforms that lack compute shader support.<br>- Check platform support using [SystemInfo.supportsComputeShaders](xref:UnityEngine.SystemInfo.supportsComputeShaders). |

The speed of model performance depends on the platform's support for multithreading in Burst, its full support for compute shaders,
and the resource usage of the game or application.

To understand a model's performance, it’s important to [Profile a model](profile-a-model.md).

## Additional resources

- [Create a runtime model](import-a-model-file.md#create-a-runtime-model)
- [How Inference Engine runs a model](how-inference-engine-runs-a-model.md)
- [Supported ONNX operators](supported-operators.md)
- [Run a model](run-a-model.md)