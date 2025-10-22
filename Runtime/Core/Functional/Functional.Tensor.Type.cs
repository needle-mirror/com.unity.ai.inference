using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the input cast to the data type element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dataType">The data type.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Type(this FunctionalTensor input, DataType dataType)
        {
            if (input.dataType == dataType)
                return input;
            return FunctionalLayer.Cast(input, dataType);
        }

        /// <summary>
        /// Returns the input cast to integers element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Int(this FunctionalTensor input)
        {
            return input.Type(DataType.Int);
        }

        /// <summary>
        /// Returns the input cast to floats element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Float(this FunctionalTensor input)
        {
            return input.Type(DataType.Float);
        }

        // Promotes a and b to the same type that is the lowest type compatible with both.
        static (FunctionalTensor, FunctionalTensor) PromoteTypes(FunctionalTensor a, FunctionalTensor b)
        {
            return a.dataType == b.dataType ? (a, b) : (a.Float(), b.Float());
        }

        // Asserts if any of the input tensors have a type different to a type.
        static void DeclareType(DataType dataType, params FunctionalTensor[] tensors)
        {
            for (var i = 0; i < tensors.Length; i++)
                Logger.AssertIsTrue(tensors[i].dataType == dataType, "FunctionalTensor has incorrect type.");
        }

        static void DeclareRank(FunctionalTensor tensor, int rank)
        {
            tensor.shape.DeclareRank(rank);
        }
    }
}
