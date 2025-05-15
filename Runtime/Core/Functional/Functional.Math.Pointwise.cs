using System;
using UnityEngine;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns |input| element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Abs(FunctionalTensor input)
        {
            return FromLayer(new Layers.Abs(-1, -1), input);
        }

        /// <summary>
        /// Returns acos(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Acos(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Acos(-1, -1), input);
        }

        /// <summary>
        /// Returns acosh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Acosh(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Acosh(-1, -1), input);
        }

        /// <summary>
        /// Returns input + other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Add(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Add(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns asin(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Asin(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Asin(-1, -1), input);
        }

        /// <summary>
        /// Returns asinh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Asinh(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Asinh(-1, -1), input);
        }

        /// <summary>
        /// Returns atan(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Atan(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Atan(-1, -1), input);
        }

        /// <summary>
        /// Returns atanh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Atanh(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Atanh(-1, -1), input);
        }

        /// <summary>
        /// Returns ⌈input⌉ element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Ceil(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return input;
            return FromLayer(new Layers.Ceil(-1, -1), input);
        }

        /// <summary>
        /// Returns input clamped to the range [min, max] element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="min">The min value.</param>
        /// <param name="max">The max value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Clamp(FunctionalTensor input, float min, float max)
        {
            input = input.Float();
            return FromLayer(new Layers.Clip(-1, -1, -1, -1), new[] { input, Constant(min), Constant(max) });
        }

        /// <summary>
        /// Returns input clamped to the range [min, max] element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="min">The min value.</param>
        /// <param name="max">The max value.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Clamp(FunctionalTensor input, int min, int max)
        {
            if (input.dataType == DataType.Float)
                return Clamp(input, (float)min, max);
            return FromLayer(new Layers.Clip(-1, -1, -1, -1), new[] { input, Constant(min), Constant(max) });
        }

        /// <summary>
        /// Returns cos(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Cos(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Cos(-1, -1), input);
        }

        /// <summary>
        /// Returns cosh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Cosh(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Cosh(-1, -1), input);
        }

        /// <summary>
        /// Returns the input values converted from angles in degrees to radians element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Deg2Rad(FunctionalTensor input)
        {
            return Mathf.Deg2Rad * input;
        }

        /// <summary>
        /// Returns input / other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Div(FunctionalTensor input, FunctionalTensor other)
        {
            // TODO support rounding mode: string roundingMode
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Div(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns the error function of input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Erf(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Erf(-1, -1), input);
        }

        /// <summary>
        /// Returns e^input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Exp(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Exp(-1, -1), input);
        }

        /// <summary>
        /// Returns input^exponent element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="exponent">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FloatPower(FunctionalTensor input, FunctionalTensor exponent)
        {
            input = input.Float();
            exponent = exponent.Float();
            return FromLayer(new Layers.Pow(-1, -1, -1), new[] { input, exponent });
        }

        /// <summary>
        /// Returns ⌊input⌋ element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Floor(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return input;
            return FromLayer(new Layers.Floor(-1, -1), input);
        }

        /// <summary>
        /// Returns ⌊input/other⌋ element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FloorDivide(FunctionalTensor input, FunctionalTensor other)
        {
            return Floor(input / other);
        }

        /// <summary>
        /// Returns input % other element-wise. The sign of the output is the same as that of the dividend.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FMod(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Mod(-1, -1, -1, true), new[] { input, other });
        }

        /// <summary>
        /// Returns the fractional part of the input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Frac(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return ZerosLike(input);
            // TODO add frac to backend and layers
            return input - Trunc(input);
        }

        /// <summary>
        /// Returns the linear interpolation input + weight * (end - input) element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="end">The second input tensor.</param>
        /// <param name="weight">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Lerp(FunctionalTensor input, FunctionalTensor end, float weight)
        {
            // TODO weight tensor
            // TODO add to layers and backend
            return input + weight * (end - input);
        }

        /// <summary>
        /// Returns log(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Log(-1, -1), input);
        }

        /// <summary>
        /// Returns log10(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log10(FunctionalTensor input)
        {
            return Log(input) * 0.4342944819f;
        }

        /// <summary>
        /// Returns log(input + 1) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log1P(FunctionalTensor input)
        {
            return Log(input + 1);
        }

        /// <summary>
        /// Returns log2(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log2(FunctionalTensor input)
        {
            return Log(input) * 1.44269504089f;
        }

        /// <summary>
        /// Returns log(e^input + e^other) element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogAddExp(FunctionalTensor input, FunctionalTensor other)
        {
            return Log(Exp(input) + Exp(other));
        }

        /// <summary>
        /// Returns the logical AND input &#38; other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalAnd(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FromLayer(new Layers.And(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns the logical NOT ~input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalNot(FunctionalTensor input)
        {
            DeclareType(DataType.Int, input);
            return FromLayer(new Layers.Not(-1, -1), input);
        }

        /// <summary>
        /// Returns the logical OR input | other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalOr(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FromLayer(new Layers.Or(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns the logical XOR input | other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalXor(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FromLayer(new Layers.Xor(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns input * other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Mul(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Mul(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns -input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Neg(FunctionalTensor input)
        {
            return FromLayer(new Layers.Neg(-1, -1), input);
        }

        /// <summary>
        /// Returns the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Positive(FunctionalTensor input)
        {
            return input;
        }

        /// <summary>
        /// Returns input^exponent element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="exponent">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pow(FunctionalTensor input, FunctionalTensor exponent)
        {
            input = input.Float();
            return FromLayer(new Layers.Pow(-1, -1, -1), new[] { input, exponent });
        }

        /// <summary>
        /// Returns input^exponent element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="exponent">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pow(FunctionalTensor input, float exponent)
        {
            input = input.Float();
            return FromLayer(new Layers.Pow(-1, -1, -1), new[] { input, Constant(exponent) });
        }

        /// <summary>
        /// Returns the input values converted from angles in radians to degrees element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Rad2Deg(FunctionalTensor input)
        {
            return Mathf.Rad2Deg * input;
        }

        /// <summary>
        /// Returns 1/input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Reciprocal(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Reciprocal(-1, -1), input);
        }

        /// <summary>
        /// Returns input % other element-wise. The sign of the output is the same as that of the divider.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Remainder(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Mod(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns [input] element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Round(FunctionalTensor input)
        {
            // TODO implement 'decimals' arg
            if (input.dataType == DataType.Int)
                return input;
            return FromLayer(new Layers.Round(-1, -1), input);
        }

        /// <summary>
        /// Returns 1/√input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor RSqrt(FunctionalTensor input)
        {
            return Reciprocal(Sqrt(input));
        }

        /// <summary>
        /// Returns the sign of the input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sign(FunctionalTensor input)
        {
            return FromLayer(new Layers.Sign(-1, -1), input);
        }

        /// <summary>
        /// Returns sin(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sin(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Sin(-1, -1), input);
        }

        /// <summary>
        /// Returns sinh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sinh(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Sinh(-1, -1), input);
        }

        /// <summary>
        /// Returns √(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sqrt(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Sqrt(-1, -1), input);
        }

        /// <summary>
        /// Returns input*input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Square(FunctionalTensor input)
        {
            return FromLayer(new Layers.Square(-1, -1), input);
        }

        /// <summary>
        /// Returns input - other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sub(FunctionalTensor input, FunctionalTensor other)
        {
            (input, other) = PromoteTypes(input, other);
            return FromLayer(new Layers.Sub(-1, -1, -1), new[] { input, other });
        }

        /// <summary>
        /// Returns tan(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tan(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Tan(-1, -1), input);
        }

        /// <summary>
        /// Returns tanh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tanh(FunctionalTensor input)
        {
            input = input.Float();
            return FromLayer(new Layers.Tanh(-1, -1), input);
        }

        /// <summary>
        /// Returns the truncated integer values of the elements of input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Trunc(FunctionalTensor input)
        {
            return Floor(Abs(input)) * Sign(input);
        }
    }
}
