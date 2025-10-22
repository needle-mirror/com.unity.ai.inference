using System.Runtime.CompilerServices;
using Unity.InferenceEngine.Graph;

[assembly: InternalsVisibleTo("Unity.InferenceEngine.Tests")]

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents the static functional methods for model building and compilation.
    /// </summary>
    public static partial class Functional
    {
        /// <summary>
        /// Returns functional tensor array for a given op target and set of arguments.
        /// Node values in the argument array should be of type (FakeNode), i.e. functional tensor wrappers.
        /// </summary>
        internal static FunctionalTensor[] FromLayer(string target, Argument[] args)
        {
            var output = FunctionalLayer.InferPartial(target, args);
            var layerNode = new LayerNode(target, args);

            if (output is PartialTensor partialTensor)
                return new[] { new FunctionalTensor(partialTensor, layerNode) };

            if (output is PartialTensor[] partialTensors)
            {
                // output is a list of tensors, insert indexer nodes equivalent to "getitem" functions in the graph
                var outputs = new FunctionalTensor[partialTensors.Length];
                for (var i = 0; i < partialTensors.Length; i++)
                    outputs[i] = new FunctionalTensor(partialTensors[i], new IndexerNode(layerNode, i));
                return outputs;
            }

            return null;
        }
    }
}
