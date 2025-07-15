using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns input == other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Equals(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Equal(), new[] { input, other });
        }

        /// <summary>
        /// Returns input == value element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="value">The integer value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Equals(FunctionalTensor input, int value)
        {
            return input.dataType == DataType.Float ? Equals(input, (float)value) : Equals(input, Constant(value));
        }

        /// <summary>
        /// Returns input == value element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="value">The float value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Equals(FunctionalTensor input, float value)
        {
            return Equals(input, Constant(value));
        }

        /// <summary>
        /// Returns input ≥ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor GreaterEqual(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.GreaterOrEqual(), new[] { input, other });
        }

        /// <summary>
        /// Returns input > other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Greater(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Greater(), new[] { input, other });
        }

        /// <summary>
        /// Returns an integer tensor with elements representing if each element of the input is finite.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor IsFinite(FunctionalTensor input)
        {
            // TODO add to backend and layers
            if (input.dataType == DataType.Int)
                return OnesLike(input);
            return LogicalNot(LogicalOr(IsInf(input), IsNaN(input)));
        }

        /// <summary>
        /// Returns an integer tensor with elements representing if each element of the input is positive or negative infinity.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor IsInf(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return ZerosLike(input);
            return FromLayer(new Layers.IsInf(true, true), input);
        }

        /// <summary>
        /// Returns an integer tensor with elements representing if each element of the input is NaN.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor IsNaN(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return ZerosLike(input);
            return FromLayer(new Layers.IsNaN(), input);
        }

        /// <summary>
        /// Returns input ≤ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LessEqual(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.LessOrEqual(), new[] { input, other });
        }

        /// <summary>
        /// Returns input &lt; other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Less(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Less(), new[] { input, other });
        }

        /// <summary>
        /// Returns the element-wise maximum of input and other.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Max(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Max(), new[] { input, other });
        }

        /// <summary>
        /// Returns the element-wise minimum of input and other.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Min(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Min(), new[] { input, other });
        }

        /// <summary>
        /// Returns input ≠ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor NotEqual(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            // TODO implement backend and layer
            return LogicalNot(Equals(input, other));
        }

        /// <summary>
        /// Returns the k largest elements of the input tensor along a given dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="k">The number of elements to calculate.</param>
        /// <param name="dim">The axis along which to perform the top-K operation.</param>
        /// <param name="largest">Whether to calculate the top-K largest elements. If this is `false` the layer calculates the top-K smallest elements.</param>
        /// <param name="sorted">Whether to return the elements in sorted order in the output tensor.</param>
        /// <returns>The output values and indices tensors in an array.</returns>
        public static FunctionalTensor[] TopK(FunctionalTensor input, int k, int dim = -1, bool largest = true, bool sorted = true)
        {
            return FromLayerMultiOutput(new Layers.TopK(dim, largest, sorted), new[] { input, Constant(new[] { k }) });
        }
    }
}
