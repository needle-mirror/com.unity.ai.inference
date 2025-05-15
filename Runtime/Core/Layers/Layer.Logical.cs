using System;
using Unity.Profiling;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents an element-wise comparison layer.
    /// </summary>
    abstract class Comparison : Layer
    {
        protected Comparison(int output, int a, int b)
            : base(new[] { output }, new[] { a, b }) { }

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
    class And : Broadcast
    {
        static readonly string k_OpName = "And";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public And(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Compress` logical layer that selects slices of an input tensor along a given axis according to a condition tensor.
    /// If you don't provide an axis, the layer flattens the input tensor.
    /// </summary>
    class Compress : Layer
    {
        static readonly string k_OpName = "Compress";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public bool hasAxis;
        public int axis;

        public Compress(int output, int input, int condition, int? axis)
            : base(new[] { output }, new[] { input, condition })
        {
            hasAxis = axis.HasValue;
            this.axis = axis ?? 0;
        }

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

        public override string ToString()
        {
            return $"{base.ToString()}, hasAxis: {hasAxis}, axis: {axis}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Equal` logical operation layer: f(a, b) = 1 if a == b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Equal : Comparison
    {
        static readonly string k_OpName = "Equal";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Equal(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Greater` logical operation layer: f(a, b) = 1 if a > b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Greater : Comparison
    {
        static readonly string k_OpName = "Greater";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Greater(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `GreaterOrEqual` logical operation layer: f(a, b) = 1 if a >= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class GreaterOrEqual : Comparison
    {
        static readonly string k_OpName = "GreaterOrEqual";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public GreaterOrEqual(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `IsInf` logical layer: f(x) = 1 elementwise if x is +Inf and detectPositive, or x is -Inf and `detectNegative` is true. Otherwise f(x) = 0.
    /// </summary>
    class IsInf : Layer
    {
        static readonly string k_OpName = "IsInf";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public bool detectNegative;
        public bool detectPositive;

        public IsInf(int output, int input, bool detectNegative, bool detectPositive)
            : base(new[] { output }, new[] { input })
        {
            this.detectNegative = detectNegative;
            this.detectPositive = detectPositive;
        }

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

        public override string ToString()
        {
            return $"{base.ToString()}, detectNegative: {detectNegative}, detectPositive: {detectPositive}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `IsNaN` logical layer: f(x) = 1 if x is NaN, otherwise f(x) = 0.
    /// </summary>
    class IsNaN : Layer
    {
        static readonly string k_OpName = "IsNaN";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public IsNaN(int output, int input)
            : base(new[] { output }, new[] { input }) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Less` logical operation layer: f(a, b) = 1 if a &lt; b, otherwise f(x) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Less : Comparison
    {
        static readonly string k_OpName = "Less";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Less(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `LessOrEqual` logical operation layer: f(a, b) = 1 if a &lt;= b, otherwise f(a,b) = 0.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class LessOrEqual : Comparison
    {
        static readonly string k_OpName = "LessOrEqual";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public LessOrEqual(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Not` logical layer: f(x) = ~x.
    /// </summary>
    class Not : Unary
    {
        static readonly string k_OpName = "Not";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Not(int output, int input)
            : base(output, input) { }

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
        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Or` logical operation layer: f(a, b) = a | b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Or : Broadcast
    {
        static readonly string k_OpName = "Or";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Or(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Xor` logical operation layer: f(a, b) = a ^ b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Xor : Broadcast
    {
        static readonly string k_OpName = "Xor";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Xor(int output, int a, int b)
            : base(output, a, b) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents an element-wise `Where` logical operation layer: f(condition, a, b) = a if `condition`, otherwise f(condition, a, b) = b.
    ///
    /// This supports numpy-style broadcasting of input tensors.
    /// </summary>
    class Where : Layer
    {
        static readonly string k_OpName = "Where";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Where(int output, int condition, int input1, int input2)
            : base(new[] { output }, new[] { condition, input1, input2 }) { }

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

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
