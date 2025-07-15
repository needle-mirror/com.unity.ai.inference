using System;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents an element-wise comparison layer.
    /// </summary>
    [Inputs(names = new[] { "a", "b" })]
    abstract class Comparison : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var a = getPartialTensor(0);
            var b = getPartialTensor(1);

            var shapeOut = a.shape.Broadcast(b.shape);
            var tensorOut = new PartialTensor<int>(shapeOut);

            if (shapeOut.IsStatic() && shapeOut.rank <= 1 && a.isPartiallyKnown && b.isPartiallyKnown)
            {
                if (a is PartialTensor<int> aInt && b is PartialTensor<int> bInt)
                {
                    for (var i = 0; i < tensorOut.length; i++)
                        tensorOut[i] = InferPartial(aInt[aInt.length > 1 ? i : 0], bInt[bInt.length > 1 ? i : 0]);
                }
                else if (a is PartialTensor<float> aFloat && b is PartialTensor<float> bFloat)
                {
                    for (var i = 0; i < tensorOut.length; i++)
                        tensorOut[i] = InferPartial(aFloat[aFloat.length > 1 ? i : 0], bFloat[bFloat.length > 1 ? i : 0]);
                }
            }

            setPartialTensor(0, tensorOut);
        }

        /// <summary>
        /// Returns the optional function that calculates an output partial tensor element from input partial tensor elements.
        /// </summary>
        internal abstract PartialTensorElement<int> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b) where T : unmanaged;
    }

    /// <summary>
    /// Represents an element-wise `And` logical operation layer: f(a, b) = a &amp; b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class And : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            if (a.IsFalse() || b.IsFalse())
                return PartialTensorElement<T>.Zero;
            if (a.IsTrue() && b.IsTrue())
                return PartialTensorElement<T>.One;
            return PartialTensorElement<T>.Unknown;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var B = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.And(A, B, O);
        }
    }

    /// <summary>
    /// Represents a `Compress` logical layer that selects slices of an input tensor along a given axis according to a condition tensor.
    /// If you don't provide an axis, the layer flattens the input tensor.
    /// </summary>
    [Operator(category = "Logical")]
    [Inputs(names = new[] { "input", "condition" })]
    partial class Compress : Layer
    {
        public bool hasAxis;
        public int axis;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var condition = getPartialTensor(1);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var isZero = shapeX.Length() * condition.shape.Length() == 0;
            if (!hasAxis)
            {
                setPartialTensor(0, PartialTensor.Create(dataType, new DynamicTensorShape(isZero ? DynamicTensorDim.Zero : DynamicTensorDim.Unknown)));
                return;
            }

            var shapeOut = shapeX;
            shapeOut[axis] = isZero ? DynamicTensorDim.Zero : DynamicTensorDim.Unknown;
            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var isTempX = !hasAxis;
            if (isTempX)
            {
                var flattenedShape = new TensorShape(X.shape.length);
                var tempX = ctx.storage.AllocateTensor(flattenedShape, X.dataType, ctx.backend.backendType);
                ctx.backend.Reshape(X, tempX);
                X = tempX;
            }

            var condition = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var numCondition = condition.shape.length;

            var indices = ctx.storage.AllocateTensor(condition.shape, DataType.Int, BackendType.CPU) as Tensor<int>;
            CPUTensorData.Pin(indices);

            var numIndices = 0;
            for (var i = 0; i < numCondition; i++)
            {
                if (condition[i] == 0)
                    continue;
                indices[numIndices] = i;
                numIndices++;
            }

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Compress(X.shape, numIndices, axis), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
            {
                if (isTempX)
                    ctx.storage.Dispose(X);
                ctx.storage.Dispose(indices);
                return;
            }
            ctx.backend.CompressWithIndices(X, indices, O, numIndices, axis);
            if (isTempX)
                ctx.storage.Dispose(X);
            ctx.storage.Dispose(indices);
        }
    }

    /// <summary>
    /// Represents an element-wise `Equal` logical operation layer: f(a, b) = 1 if a == b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class Equal : Comparison
    {
        internal override PartialTensorElement<int> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Eq(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Equal(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.Equal(A as Tensor<float>, B as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Greater` logical operation layer: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class Greater : Comparison
    {
        internal override PartialTensorElement<int> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Gt(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Greater(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.Greater(A as Tensor<float>, B as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `GreaterOrEqual` logical operation layer: f(a, b) = 1 if a >= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class GreaterOrEqual : Comparison
    {
        internal override PartialTensorElement<int> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Ge(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.GreaterOrEqual(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.GreaterOrEqual(A as Tensor<float>, B as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `IsInf` logical layer: f(x) = 1 elementwise if x is +Inf and detectPositive, or x is -Inf and `detectNegative` is true. Otherwise f(x) = 0.
    /// </summary>
    [Operator(category = "Logical")]
    partial class IsInf : Layer
    {
        public bool detectNegative;
        public bool detectPositive;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            setPartialTensor(0, new PartialTensor<int>(getPartialTensor(0).shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.IsInf(A, O, detectNegative, detectPositive);
        }
    }

    /// <summary>
    /// Represents an element-wise `IsNaN` logical layer: f(x) = 1 if x is NaN, otherwise f(x) = 0.
    /// </summary>
    [Operator(category = "Logical")]
    partial class IsNaN : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            setPartialTensor(0, new PartialTensor<int>(getPartialTensor(0).shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.IsNaN(A, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Less` logical operation layer: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class Less : Comparison
    {
        internal override PartialTensorElement<int> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Lt(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.Less(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.Less(A as Tensor<float>, B as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `LessOrEqual` logical operation layer: f(a, b) = 1 if a &lt;= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class LessOrEqual : Comparison
    {
        internal override PartialTensorElement<int> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            return PartialTensorElement<T>.Le(a, b);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]);
            var B = ctx.storage.GetTensor(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (A is Tensor<int>)
                ctx.backend.LessOrEqual(A as Tensor<int>, B as Tensor<int>, O);
            else
                ctx.backend.LessOrEqual(A as Tensor<float>, B as Tensor<float>, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Not` logical layer: f(x) = ~x.
    /// </summary>
    [Operator(category = "Logical")]
    partial class Not : Unary
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a)
        {
            if (a.IsTrue())
                return PartialTensorElement<T>.Zero;
            if (a.IsFalse())
                return PartialTensorElement<T>.One;
            return PartialTensorElement<T>.Unknown;
        }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var input = getPartialTensor(0) as PartialTensor<int>;
            var output = new PartialTensor<int>(input.shape);
            if (output.isPartiallyKnown)
            {
                for (var i = 0; i < output.length; i++)
                {
                    if (input[i].IsTrue())
                        output[i] = PartialTensorElement<int>.Zero;
                    else if (input[i].IsFalse())
                        output[i] = PartialTensorElement<int>.One;
                }
            }

            setPartialTensor(0, output);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Not(A, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Or` logical operation layer: f(a, b) = a | b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class Or : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            if (a.IsFalse() && b.IsFalse())
                return PartialTensorElement<T>.Zero;
            if (a.IsTrue() || b.IsTrue())
                return PartialTensorElement<T>.One;
            return PartialTensorElement<T>.Unknown;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var B = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Or(A, B, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Xor` logical operation layer: f(a, b) = a ^ b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    partial class Xor : Broadcast
    {
        internal override PartialTensorElement<T> InferPartial<T>(PartialTensorElement<T> a, PartialTensorElement<T> b)
        {
            if (a.IsTrue())
            {
                if (b.IsTrue())
                    return PartialTensorElement<T>.Zero;
                if (b.IsFalse())
                    return PartialTensorElement<T>.One;
            }

            if (a.IsFalse())
            {
                if (b.IsTrue())
                    return PartialTensorElement<T>.One;
                if (b.IsFalse())
                    return PartialTensorElement<T>.Zero;
            }

            return PartialTensorElement<T>.Unknown;
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var A = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var B = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape), DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Xor(A, B, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Where` logical operation layer: f(condition, a, b) = a if `condition`, otherwise f(condition, a, b) = b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    [Operator(category = "Logical")]
    [Inputs(names = new[] { "condition", "input1", "input2" })]
    partial class Where : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var condition = getPartialTensor(0) as PartialTensor<int>;
            var input1 = getPartialTensor(1);
            var input2 = getPartialTensor(2);

            var shapeOut = condition.shape.Broadcast(input1.shape.Broadcast(input2.shape));
            var tensorOut = PartialTensor.Create(input1.dataType, shapeOut);

            if (shapeOut.IsStatic() && shapeOut.rank <= 1 && condition.isPartiallyKnown && input1.isPartiallyKnown && input2.isPartiallyKnown)
            {
                for (var i = 0; i < tensorOut.length; i++)
                {
                    var c = condition[i % condition.length];
                    if (c.IsTrue())
                        tensorOut.CopyElement(i, input1, i % input1.length);
                    else if (c.IsFalse())
                        tensorOut.CopyElement(i, input2, i % input2.length);
                }
            }

            setPartialTensor(0, tensorOut);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var C = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var A = ctx.storage.GetTensor(inputs[1]);
            var B = ctx.storage.GetTensor(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], A.shape.Broadcast(B.shape.Broadcast(C.shape)), A.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Where(C, A, B, O);
        }
    }
}
