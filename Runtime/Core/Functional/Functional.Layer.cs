using System.Runtime.CompilerServices;
using UnityEngine.Assertions;

[assembly: InternalsVisibleTo("Unity.InferenceEngine.Tests")]

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents the static functional methods for model building and compilation.
    /// </summary>
    public static partial class Functional
    {
        internal static FunctionalTensor[] FromLayerMultiOutput(Layer layer, FunctionalTensor[] inputs)
        {
            layer.inputs = new int[inputs.Length];
            layer.outputs = new int[layer.OutputCount];
            var node = new LayerNode(inputs, layer);
            return node.CreateOutputs();
        }

        internal static FunctionalTensor FromLayer(Layer layer, FunctionalTensor[] inputs)
        {
            return FromLayerMultiOutput(layer, inputs)[0];
        }

        internal static FunctionalTensor FromLayer(Layer layer, FunctionalTensor input)
        {
            return FromLayerMultiOutput(layer, new[] { input })[0];
        }
    }
}
