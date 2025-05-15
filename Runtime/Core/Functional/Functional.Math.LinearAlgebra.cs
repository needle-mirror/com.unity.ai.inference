using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the matrix product input @ other.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MatMul(FunctionalTensor input, FunctionalTensor other)
        {
            input = input.Float();
            other = other.Float();
            return FromLayer(new Layers.MatMul(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Performs a batch matrix-matrix product of matrices : y = x @ a + b.
        /// B : (N)
        /// A : (K, N)
        /// X : (..., M, K)
        /// O : (..., M, K)
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="weight">The second input tensor.</param>
        /// <param name="bias">The bias input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AddBMM(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias)
        {
            input = input.Float();
            weight = weight.Float();
            bias = bias.Float();
            return FromLayer(new Layers.Dense(-1, -1, -1, -1, Layers.FusableActivation.None), new[] { input, weight, bias });
        }
    }
}
