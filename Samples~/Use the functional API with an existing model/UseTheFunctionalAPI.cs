using UnityEngine;
using Unity.InferenceEngine;

public class UseTheFunctionalAPI : MonoBehaviour
{
    [SerializeField]
    BackendType backendType = BackendType.GPUCompute;

    [SerializeField]
    Texture2D inputTexture;

    [SerializeField]
    ModelAsset sourceModelAsset;

    Model m_RuntimeModel;
    Worker m_Worker;

    void OnEnable()
    {
        // Load the source model.
        // It expects input as sRGB values in the range[-1..1], however the input texture is an RGB texture in the range[0..1].
        var sourceModel = ModelLoader.Load(sourceModelAsset);

        // Declare a functional graph.
        var graph = new FunctionalGraph();

        // Get the input functional tensor from the graph with input data type and shape matching that of the original model input.
        var RGB = graph.AddInput(sourceModel, 0);

        // Apply f(x) = x^(1/2.2) element-wise to transform from RGB to sRGB.
        var sRGB = Functional.Pow(RGB, Functional.Constant(1 / 2.2f));

        // Apply f(x) = x * 2 - 1 element-wise to transform values from the range [0, 1] to the range [-1, 1].
        var sRGB_normalised = sRGB * 2 - 1;

        // Apply the forward method of the source model to the transformed functional input and add the outputs to the graph.
        var outputs = Functional.Forward(sourceModel, sRGB_normalised);
        graph.AddOutputs(outputs);

        // Compile the graph to return the final model.
        m_RuntimeModel = graph.Compile();

        // Create worker to run the model.
        m_Worker = new Worker(m_RuntimeModel, backendType);

        // Create the input from our input texture.
        using var input = new Tensor<float>(new TensorShape(1, 3, 64, 64));
        TextureConverter.ToTensor(inputTexture, input);
        Debug.Log($"input shape = {input.shape}");

        // Execute the model.
        m_Worker.Schedule(input);

        // Get the output and download onto the CPU to view it.
        var output = m_Worker.PeekOutput() as Tensor<float>;
        Debug.Log($"output shape = {output.shape}");

        var outputCPU = output.ReadbackAndClone();
        Debug.Log($"average values = {outputCPU[0, 0]}, {outputCPU[0, 1]}, {outputCPU[0, 2]}");
        outputCPU.Dispose();
    }

    void OnDisable()
    {
        // Clean up Inference Engine resources
        m_Worker?.Dispose();
    }
}

