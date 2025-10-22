using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the indices of the maximum value of the elements of the input along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ArgMax(FunctionalTensor input, int dim = 0, bool keepdim = false)
        {
            return FunctionalLayer.ArgMax(input, dim, keepdim, false);
        }

        /// <summary>
        /// Returns the indices of the minimum value of the elements of the input along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ArgMin(FunctionalTensor input, int dim = 0, bool keepdim = false)
        {
            return FunctionalLayer.ArgMin(input, dim, keepdim, false);
        }

        /// <summary>
        /// Returns the maximum value of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMax(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalLayer.ReduceMax(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the maximum value of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMax(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceMax(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the minimum value of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMin(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalLayer.ReduceMin(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the minimum value of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMin(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceMin(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the L1 norm of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceL1(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalLayer.ReduceL1(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the L1 norm of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceL1(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceL1(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the L2 norm of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceL2(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalLayer.ReduceL2(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the L2 norm of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceL2(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceL2(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the log of summed exponentials of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceLogSumExp(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            input = input.Float();
            return FunctionalLayer.ReduceLogSumExp(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the log of summed exponentials of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceLogSumExp(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceLogSumExp(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the mean of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMean(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            input = input.Float();
            return FunctionalLayer.ReduceMean(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the mean  of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceMean(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceMean(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the product of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceProd(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalLayer.ReduceProd(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the product of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceProd(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceProd(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the sum of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceSum(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalLayer.ReduceSum(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the sum of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceSum(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceSum(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the sum of the square of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceSumSquare(FunctionalTensor input, int[] dim, bool keepdim = false)
        {
            return FunctionalLayer.ReduceSumSquare(input, Constant(dim), keepdim, false);
        }

        /// <summary>
        /// Returns the sum of the square of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimension in the output.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceSumSquare(FunctionalTensor input, int dim, bool keepdim = false)
        {
            return ReduceSumSquare(input, new[] { dim }, keepdim);
        }

        /// <summary>
        /// Returns the variance of the elements of the input tensor along the dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <param name="correction">The difference between the sample size and sample degrees of freedom. Defaults to Bessel’s correction.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceVariance(FunctionalTensor input, int[] dim, bool keepdim = false, float correction = 1f)
        {
            input = input.Float();
            return FunctionalLayer.ReduceVariance(input, Constant(dim), keepdim, false, correction);
        }

        /// <summary>
        /// Returns the variance of the elements of the input tensor along the dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension to reduce.</param>
        /// <param name="keepdim">Whether to keep the reduced dimensions in the output.</param>
        /// <param name="correction">The difference between the sample size and sample degrees of freedom. Defaults to Bessel’s correction.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ReduceVariance(FunctionalTensor input, int dim, bool keepdim = false, float correction = 1f)
        {
            return ReduceVariance(input, new[] { dim }, keepdim, correction);
        }
    }
}
