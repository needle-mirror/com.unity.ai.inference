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
            return FunctionalLayer.Abs(input);
        }

        /// <summary>
        /// Returns acos(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Acos(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Acos(input);
        }

        /// <summary>
        /// Returns acosh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Acosh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Acosh(input);
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
            return FunctionalLayer.Add(input, other);
        }

        /// <summary>
        /// Returns Atan2(input, other) element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Atan2(FunctionalTensor input, FunctionalTensor other)
        {
            input = input.Float();
            other = other.Float();
            return FunctionalLayer.Atan2(input, other);
        }

        /// <summary>
        /// Returns asin(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Asin(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Asin(input);
        }

        /// <summary>
        /// Returns asinh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Asinh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Asinh(input);
        }

        /// <summary>
        /// Returns atan(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Atan(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Atan(input);
        }

        /// <summary>
        /// Returns atanh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Atanh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Atanh(input);
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
            return FunctionalLayer.Ceil(input);
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
            return FunctionalLayer.Clip(input, Constant(min), Constant(max));
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
            return FunctionalLayer.Clip(input, Constant(min), Constant(max));
        }

        /// <summary>
        /// Returns cos(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Cos(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Cos(input);
        }

        /// <summary>
        /// Returns cosh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Cosh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Cosh(input);
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
            return Div(input, other, roundingMode: null);
        }

        /// <summary>
        /// Returns input / other element-wise with rounding mode.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <param name="roundingMode">The type of rounding applied to the result:
        ///
        /// null - default behavior. Promotes the inputs to float tensors and performs no rounding.
        ///
        /// "trunc" - rounds the results of the division towards zero.
        ///
        /// "floor" - rounds the results of the division down.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Div(FunctionalTensor input, FunctionalTensor other, string roundingMode)
        {
            switch (roundingMode)
            {
                case null:
                {
                    input = input.Float();
                    other = other.Float();
                    return FunctionalLayer.Div(input, other);
                }
                case "trunc":
                {
                    (input, other) = PromoteTypes(input, other);
                    return FunctionalLayer.TruncDiv(input, other);
                }
                case "floor":
                {
                    (input, other) = PromoteTypes(input, other);
                    return FunctionalLayer.FloorDiv(input, other);
                }
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        /// <summary>
        /// Returns the error function of input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Erf(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Erf(input);
        }

        /// <summary>
        /// Returns e^input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Exp(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Exp(input);
        }

        /// <summary>
        /// Returns e^input - 1 element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Expm1(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Expm1(input);
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
            return FunctionalLayer.Pow(input, exponent);
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
            return FunctionalLayer.Floor(input);
        }

        /// <summary>
        /// Returns ⌊input/other⌋ element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor FloorDivide(FunctionalTensor input, FunctionalTensor other)
        {
            return Div(input, other, "floor");
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
            return FunctionalLayer.Mod(input, other, true);
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
            return FunctionalLayer.Log(input);
        }

        /// <summary>
        /// Returns log10(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log10(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Log10(input);
        }

        /// <summary>
        /// Returns log(input + 1) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log1P(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Log1p(input);
        }

        /// <summary>
        /// Returns log2(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Log2(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Log2(input);
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
            return FunctionalLayer.And(NotEqual(input, Constant(0)), NotEqual(other, Constant(0)));
        }

        /// <summary>
        /// Returns the logical NOT ~input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalNot(FunctionalTensor input)
        {
            var zero = Constant(0);
            (input, zero) = PromoteTypes(input, zero);
            return FunctionalLayer.Equal(input, zero);
        }

        /// <summary>
        /// Returns the logical OR input | other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalOr(FunctionalTensor input, FunctionalTensor other)
        {
            return FunctionalLayer.Or(NotEqual(input, Constant(0)), NotEqual(other, Constant(0)));
        }

        /// <summary>
        /// Returns the logical XOR input ^ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor LogicalXor(FunctionalTensor input, FunctionalTensor other)
        {
            return FunctionalLayer.Xor(NotEqual(input, Constant(0)), NotEqual(other, Constant(0)));
        }

        /// <summary>
        /// Returns the bitwise AND input &#38; other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor BitwiseAnd(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FunctionalLayer.BitwiseAnd(input, other);
        }

        /// <summary>
        /// Returns the bitwise NOT ~input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor BitwiseNot(FunctionalTensor input)
        {
            DeclareType(DataType.Int, input);
            return FunctionalLayer.BitwiseNot(input);
        }

        /// <summary>
        /// Returns the bitwise OR input &#124; other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor BitwiseOr(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FunctionalLayer.BitwiseOr(input, other);
        }

        /// <summary>
        /// Returns the bitwise XOR input ^ other element-wise.
        /// </summary>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor BitwiseXor(FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, input, other);
            return FunctionalLayer.BitwiseXor(input, other);
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
            return FunctionalLayer.Mul(input, other);
        }

        /// <summary>
        /// Returns -input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Neg(FunctionalTensor input)
        {
            return FunctionalLayer.Neg(input);
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
            return FunctionalLayer.Pow(input, exponent);
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
            return FunctionalLayer.Pow(input, Constant(exponent));
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
            return FunctionalLayer.Reciprocal(input);
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
            return FunctionalLayer.Mod(input, other, false);
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
            return FunctionalLayer.Round(input);
        }

        /// <summary>
        /// Returns 1/√input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor RSqrt(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Rsqrt(input);
        }

        /// <summary>
        /// Returns the sign of the input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sign(FunctionalTensor input)
        {
            return FunctionalLayer.Sign(input);
        }

        /// <summary>
        /// Returns sin(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sin(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Sin(input);
        }

        /// <summary>
        /// Returns sinh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sinh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Sinh(input);
        }

        /// <summary>
        /// Returns √(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Sqrt(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Sqrt(input);
        }

        /// <summary>
        /// Returns input*input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Square(FunctionalTensor input)
        {
            return FunctionalLayer.Square(input);
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
            return FunctionalLayer.Sub(input, other);
        }

        /// <summary>
        /// Returns tan(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tan(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Tan(input);
        }

        /// <summary>
        /// Returns tanh(input) element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tanh(FunctionalTensor input)
        {
            input = input.Float();
            return FunctionalLayer.Tanh(input);
        }

        /// <summary>
        /// Returns the truncated integer values of the elements of input element-wise.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Trunc(FunctionalTensor input)
        {
            if (input.dataType == DataType.Int)
                return input;
            return FunctionalLayer.Trunc(input);
        }
    }
}
