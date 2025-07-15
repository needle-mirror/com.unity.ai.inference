using System;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents an element-wise `Abs` math layer: f(x) = |x|.
    /// </summary>
    [Operator(category = "Math")]
    partial class Abs : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Abs(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Abs(X as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Add` math operation layer: f(a, b) = a + b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Add : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return a + b;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Add(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Add(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Ceil` math layer: f(x) = ceil(x).
    /// </summary>
    [Operator(category = "Math")]
    partial class Ceil : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Ceil(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Clip` math layer: f(x, xmin, xmax) = min(max(x, xmin), xmax)
    /// </summary>
    [Operator(category = "Math")]
    [Inputs(names = new[] { "input", "min", "max" }, inputCPURead = new[] { 1, 2 })]
    partial class Clip : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            setPartialTensor(0, PartialTensor.Create(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            // TODO don't switch data type at runtime
            if (X is Tensor<int>)
            {
                var min = ctx.storage.GetInt(inputs[1], int.MinValue);
                var max = ctx.storage.GetInt(inputs[2], int.MaxValue);
                ctx.backend.Clip(X as Tensor<int>, O as Tensor<int>, min, max);
            }
            else
            {
                var min = ctx.storage.GetFloat(inputs[1], float.MinValue);
                var max = ctx.storage.GetFloat(inputs[2], float.MaxValue);
                ctx.backend.Clip(X as Tensor<float>, O as Tensor<float>, min, max);
            }
        }
    }

    /// <summary>
    /// Represents a `CumSum` math layer that performs the cumulative sum along a given axis.
    /// </summary>
    [Operator(category = "Math")]
    [Inputs(names = new[] { "input", "axis" }, inputCPURead = new[] { 1 })]
    partial class CumSum : Layer
    {
        public bool reverse;
        public bool exclusive;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            setPartialTensor(0, PartialTensor.Create(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var axis = ctx.storage.GetInt(inputs[1]);
            if (X is Tensor<int>)
                ctx.backend.CumSum(X as Tensor<int>, O as Tensor<int>, axis, reverse, exclusive);
            else
                ctx.backend.CumSum(X as Tensor<float>, O as Tensor<float>, axis, reverse, exclusive);
        }
    }

    /// <summary>
    /// Represents a `Dense` math operation layer which performs a matrix multiplication operation: f(x, w, b) = X x W + B.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    [Inputs(names = new[] { "input", "weights", "bias" })]
    partial class Dense : FusedActivation
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var W = getPartialTensor(1);
            var B = getPartialTensor(2);
            var shapeOut = X.shape.MatMul(W.shape);
            if (shapeOut.hasRank)
                shapeOut[-1] = DynamicTensorDim.MaxDefinedDim(B.shape[0], shapeOut[-1]);
            setPartialTensor(0, new PartialTensor<float>(shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var W = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.MatMul(W.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Dense(X as Tensor<float>, W as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O, fusedActivation);
        }
    }

    [Operator(category = "Math")]
    [Inputs(names = new[] { "input", "weights", "bias" })]
    partial class DenseBatched : FusedActivation
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var W = getPartialTensor(1);
            var B = getPartialTensor(2);
            var shapeOut = X.shape.MatMul(W.shape);
            if (shapeOut.hasRank)
                shapeOut[-1] = DynamicTensorDim.MaxDefinedDim(B.shape[-1], shapeOut[-1]);
            setPartialTensor(0, new PartialTensor<float>(shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var W = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.MatMul(W.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DenseBatched(X as Tensor<float>, W as Tensor<float>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O, fusedActivation);
        }
    }

    /// <summary>
    /// Represents an element-wise `Div` math operation layer: f(a, b) = a / b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Div : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return a / b;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Div(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Div(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an `Einsum` math operation layer.
    /// </summary>
    /// <description>
    /// The Einsum operator evaluates algebraic tensor operations on a sequence of tensors, using the Einstein summation convention. The equation string contains a comma-separated sequence of lower case letters. Each term corresponds to an operand tensor, and the characters within the terms correspond to operands dimensions.
    /// This sequence may be followed by "->" to separate the left and right hand side of the equation. If the equation contains "->" followed by the right-hand side, the explicit (not classical) form of the Einstein summation is performed, and the right-hand side indices indicate output tensor dimensions. In other cases, output indices are (implicitly) set to the alphabetically sorted sequence of indices appearing exactly once in the equation.
    /// When a dimension character is repeated in the left-hand side, it represents summation along the dimension.
    /// The equation may contain ellipsis ("...") to enable broadcasting. Ellipsis must indicate a fixed number of dimensions. Specifically, every occurrence of ellipsis in the equation must represent the same number of dimensions. The right-hand side may contain exactly one ellipsis. In implicit mode, the ellipsis dimensions are set to the beginning of the output. The equation string may contain space (U+0020) character.
    /// </description>
    [Operator(category = "Math")]
    [Inputs(isVariadic = true)]
    partial class Einsum : Layer
    {
        public string equation;

        TensorShape[] operandShapes;
        TensorIndex[] operandIndices;
        Tensor<float>[] operandTensors;

        void Initialize()
        {
            operandShapes = new TensorShape[inputs.Length];
            operandIndices = new TensorIndex[inputs.Length];
            operandTensors = new Tensor<float>[inputs.Length];
        }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            if (operandShapes is null)
                Initialize();
            var inputShapes = new DynamicTensorShape[inputs.Length];
            for (var i = 0; i < inputShapes.Length; i++)
                inputShapes[i] = getPartialTensor(i).shape;
            var shape = EinsumHelper.ParseEquationStringShape(equation, inputShapes, ref operandIndices, out _, out _);
            setPartialTensor(0, new PartialTensor<float>(shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            if (operandShapes is null)
                Initialize();
            for (int i = 0; i < inputs.Length; i++)
            {
                operandTensors[i] = ctx.storage.GetTensor(inputs[i]) as Tensor<float>;
                operandShapes[i] = operandTensors[i].shape;
            }
            EinsumHelper.ParseEquationString(equation, operandShapes, ref operandIndices, out var outputIndices, out var outputShape, out var sumIndices, out var sumShape, out var numIndices);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], outputShape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (inputs.Length > 2)
                CPUBackend.EinsumND(operandTensors, O, operandShapes, operandIndices, outputIndices, outputShape, sumIndices, sumShape, numIndices);
            else
                ctx.backend.Einsum(operandTensors, O, operandIndices, outputIndices, sumIndices, sumShape);
        }
    }

    /// <summary>
    /// Represents an element-wise `Exp` math layer: f(x) = e^{x}.
    /// </summary>
    [Operator(category = "Math")]
    partial class Exp : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Exp(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Floor` math layer: f(x) = floor(x).
    /// </summary>
    [Operator(category = "Math")]
    partial class Floor : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Floor(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Log` math layer: f(x) = log(x).
    /// </summary>
    [Operator(category = "Math")]
    partial class Log : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Log(X, O);
        }
    }

    /// <summary>
    /// Represents a `MatMul` math operation layer which performs a matrix multiplication operation: f(a, b) = a x b.
    /// </summary>
    [Operator(category = "Math")]
    [Inputs(names = new[] { "input0", "input1" })]
    partial class MatMul : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var A = getPartialTensor(0);
            var B = getPartialTensor(1);
            setPartialTensor(0, PartialTensor.Create(A.dataType, A.shape.MatMul(B.shape)));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.MatMul(B.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (A.shape.HasZeroDims() || B.shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul(A as Tensor<float>, B as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents a `MatMul2D` math operation layer which performs a matrix multiplication operation with optional transposes: f(a, b) = a' x b'.
    /// </summary>
    [Operator(category = "Math")]
    [Inputs(names = new[] { "input0", "input1" })]
    partial class MatMul2D : Layer
    {
        public bool transposeA;
        public bool transposeB;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var A = getPartialTensor(0);
            var B = getPartialTensor(1);

            var shapeA = A.shape;
            var shapeB = B.shape;

            shapeA.DeclareRank(2);
            shapeB.DeclareRank(2);

            var mulXDim = transposeA ? shapeA[0] : shapeA[1];
            var mulYDim = transposeB ? shapeB[1] : shapeB[0];
            Logger.AssertIsFalse(mulXDim != mulYDim, "MatMul2D.ValueError: failed, dims not equal");

            var shapeOut = new DynamicTensorShape(transposeA ? shapeA[1] : shapeA[0], transposeB ? shapeB[0] : shapeB[1]);
            setPartialTensor(0, PartialTensor.Create(A.dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Gemm(A.shape, B.shape, transposeA, transposeB), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (A.shape.HasZeroDims() || B.shape.HasZeroDims())
                ctx.backend.MemSet(O, 0.0f);
            else
                ctx.backend.MatMul2D(A as Tensor<float>, B as Tensor<float>, O, transposeA, transposeB);
        }
    }

    /// <summary>
    /// Represents an element-wise `Min` math operation layer: f(a, b) = min(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Min : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Min(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Min(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Min(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(a, b) = max(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Max : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Max(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Max(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Max(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Max` math operation layer: f(a, b) = a % b.
    ///
    /// If fmod is false the sign of the remainder is the same as that of the divisor as in Python.
    ///
    /// If fmod is true the sign of the remainder is the same as that of the dividend as in C#.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Mod : Broadcast
    {
        public bool fmod;

        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            if (!fmod)
                return PartialTensorElement<T>.Mod(a, b);
            return PartialTensorElement<T>.FMod(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (!fmod)
            {
                if (A is Tensor<int>)
                    ctx.backend.Mod(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
                else
                    ctx.backend.Mod(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
            }
            else
            {
                if (A is Tensor<int>)
                    ctx.backend.FMod(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
                else
                    ctx.backend.FMod(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
            }
        }
    }

    /// <summary>
    /// Represents an element-wise `Mul` math operation layer: f(a, b) = a * b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Mul : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return a * b;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Mul(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Mul(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Neg` math layer: f(x) = -x.
    /// </summary>
    [Operator(category = "Math")]
    partial class Neg : Unary
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a)
        {
            return -a;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Neg(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Neg(X as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Pow` math operation layer: f(a, b) = pow(a, b).
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Pow : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Pow(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;

            if (A is Tensor<int>)
            {
                if (B is Tensor<int>)
                    ctx.backend.Pow(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
                else
                    ctx.backend.Pow(A as Tensor<int>, B as Tensor<float>, O as Tensor<int>);
            }
            else
            {
                if (B is Tensor<int>)
                    ctx.backend.Pow(A as Tensor<float>, B as Tensor<int>, O as Tensor<float>);
                else
                    ctx.backend.Pow(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
            }
        }
    }

    /// <summary>
    /// Represents an element-wise `Reciprocal` math layer: f(x) = 1 / x.
    /// </summary>
    [Operator(category = "Math")]
    partial class Reciprocal : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reciprocal(X as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Round` math layer: f(x) = round(x).
    /// </summary>
    [Operator(category = "Math")]
    partial class Round : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Round(X as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Mad` math operation: multiplies and adds bias to a tensor: f(T, s, b) = s * T + b.
    /// </summary>
    [Operator(category = "Math")]
    partial class ScalarMad : Activation
    {
        public DataType dataType;
        public float sFloat;
        public float bFloat;
        public int sInt;
        public int bInt;

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Float)
                ctx.backend.ScalarMad(X as Tensor<float>, O as Tensor<float>, sFloat, bFloat);
            else
                ctx.backend.ScalarMad(X as Tensor<int>, O as Tensor<int>, sInt, bInt);
        }
    }

    /// <summary>
    /// Represents an element-wise `Shrink` math layer: f(x) = x + bias if x &lt; lambd. f(x) = x - bias if x &gt; lambd. Otherwise f(x) = 0.
    /// </summary>
    [Operator(category = "Math")]
    partial class Shrink : Layer
    {
        public float bias;
        public float lambd;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            setPartialTensor(0, PartialTensor.Create(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Shrink(X as Tensor<float>, O, bias, lambd);
        }
    }

    /// <summary>
    /// Represents an element-wise `Sign` math layer: f(x) = 1 if x > 0. f(x) = -1 if x &lt; 0. Otherwise f(x) = 0.
    /// </summary>
    [Operator(category = "Math")]
    partial class Sign : Unary
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a)
        {
            return PartialTensorElement<T>.Sign(a);
        }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            setPartialTensor(0, PartialTensor.Create(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Sign(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Sign(X as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Sqrt` math layer: f(x) = sqrt(x).
    /// </summary>
    [Operator(category = "Math")]
    partial class Sqrt : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sqrt(X as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Square` math layer: f(x) = x * x.
    /// </summary>
    [Operator(category = "Math")]
    partial class Square : Unary
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a)
        {
            return a * a;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.Square(X as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Square(X as Tensor<float>, O as Tensor<float>);
        }
    }

    /// <summary>
    /// Represents an element-wise `Sub` math operation layer: f(a, b) = a - b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Math")]
    partial class Sub : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return a - b;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], TensorShapeHelper.BroadcastShape(A, B), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Sub(A as Tensor<int>, B as Tensor<int>, O as Tensor<int>);
            else
                ctx.backend.Sub(A as Tensor<float>, B as Tensor<float>, O as Tensor<float>);
        }
    }
}
