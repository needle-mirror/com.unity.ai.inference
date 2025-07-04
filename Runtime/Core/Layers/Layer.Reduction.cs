using System;
using Unity.Profiling;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents the abstract base class for reduction layers.
    /// </summary>
    abstract class Reduce : Layer
    {
        public bool keepdims;
        public bool noopWithEmptyAxes;

        protected Reduce(int output, int data, int axes, bool keepdims, bool noopWithEmptyAxes)
            : base(new[] { output }, new[] { data, axes })
        {
            this.keepdims = keepdims;
            this.noopWithEmptyAxes = noopWithEmptyAxes;
        }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var axes = getPartialTensor(1) as PartialTensor<int>;
            var shapeAxes = axes?.shape ?? new DynamicTensorShape(DynamicTensorDim.Zero);
            if (axes != null && axes.isPartiallyKnown && axes.length != 0)
            {
                var reducedShape = new DynamicTensorShape(shapeX);
                if (!axes.IsStatic() && reducedShape.hasRank)
                {
                    // replace any non 1 dims with unknown (1 stays the same whether reduced or not)
                    for (var i = 0; i < reducedShape.rank; i++)
                    {
                        if (reducedShape[i] == 1)
                            continue;
                        reducedShape[i] = DynamicTensorDim.Unknown;
                    }
                }

                for (var i = 0; i < axes.length; i++)
                {
                    if (!axes[i].isValue)
                        continue;
                    var axis = axes[i].value;
                    reducedShape[axis] = DynamicTensorDim.One;
                }

                var tensorOut = PartialTensor.Create(dataType, reducedShape);
                if (!keepdims)
                {
                    tensorOut = tensorOut.Reshape(!axes.IsStatic() ? DynamicTensorShape.DynamicOfRank(tensorOut.shape.rank - axes.length) : tensorOut.shape.Squeeze(axes));
                }

                setPartialTensor(0, tensorOut);
                return;
            }

            if (shapeAxes.IsStatic())
            {
                if (shapeAxes[0].value != 0)
                    setPartialTensor(0, PartialTensor.Create(dataType, keepdims ? DynamicTensorShape.DynamicOfRankLike(shapeX) : DynamicTensorShape.DynamicRank));
                else if (noopWithEmptyAxes)
                    setPartialTensor(0, PartialTensor.Create(dataType, shapeX));
                else
                    setPartialTensor(0, PartialTensor.Create(dataType, keepdims ? DynamicTensorShape.OnesLike(shapeX) : new DynamicTensorShape()));
                return;
            }

            setPartialTensor(0, PartialTensor.Create(dataType, keepdims && !noopWithEmptyAxes ? DynamicTensorShape.DynamicOfRankLike(shapeX) : DynamicTensorShape.DynamicRank));
        }

        public override string ToString()
        {
            return $"{base.ToString()}, keepdims: {keepdims}, noopWithEmptyAxes: {noopWithEmptyAxes}";
        }
    }

    /// <summary>
    /// Represents a `ReduceL1` reduction layer along the given axes: f(x1, x2 ... xn) = |x1| + |x2| + ... + |xn|.
    /// </summary>
    class ReduceL1 : Reduce
    {
        static readonly string k_OpName = "ReduceL1";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceL1(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X.shape.HasZeroDims())
            {
                ctx.backend.MemClear(O);
                return;
            }
            if (X is Tensor<int>)
                ctx.backend.ReduceL1(X as Tensor<int>, O as Tensor<int>, axes);
            else
                ctx.backend.ReduceL1(X as Tensor<float>, O as Tensor<float>, axes);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceL2` reduction layer along the given axes: f(x1, x2 ... xn) = sqrt(x1² + x2² + ... + xn²).
    /// </summary>
    class ReduceL2 : Reduce
    {
        static readonly string k_OpName = "ReduceL2";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceL2(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (X.shape.HasZeroDims())
                ctx.backend.MemClear(O);
            else
                ctx.backend.ReduceL2(X, O, axes);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceLogSum` reduction layer along the given axes: f(x1, x2 ... xn) = log(x1 + x2 + ... + xn).
    /// </summary>
    class ReduceLogSum : Reduce
    {
        static readonly string k_OpName = "ReduceLogSum";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceLogSum(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (X.shape.HasZeroDims())
                ctx.backend.MemSet(O, float.NegativeInfinity);
            else
                ctx.backend.ReduceLogSum(X, O, axes);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceLogSumExp` reduction layer along the given axes: f(x1, x2 ... xn) = log(e^x1 + e^x2 + ... + e^xn).
    /// </summary>
    class ReduceLogSumExp : Reduce
    {
        static readonly string k_OpName = "ReduceLogSumExp";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceLogSumExp(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (X.shape.HasZeroDims())
                ctx.backend.MemSet(O, float.NegativeInfinity);
            else
                ctx.backend.ReduceLogSumExp(X, O, axes);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceMax` reduction layer along the given axes: f(x1, x2 ... xn) = max(x1, x2, ... , xn).
    /// </summary>
    class ReduceMax : Reduce
    {
        static readonly string k_OpName = "ReduceMax";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceMax(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
            {
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<int>, int.MinValue);
                else
                    ctx.backend.ReduceMax(X as Tensor<int>, O as Tensor<int>, axes);
            }
            else
            {
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<float>, float.NegativeInfinity);
                else
                    ctx.backend.ReduceMax(X as Tensor<float>, O as Tensor<float>, axes);
            }
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceMean` reduction layer along the given axes: f(x1, x2 ... xn) = (x1 + x2 + ... + xn) / n.
    /// </summary>
    class ReduceMean : Reduce
    {
        static readonly string k_OpName = "ReduceMean";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceMean(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            if (X.shape.HasZeroDims())
                ctx.backend.MemClear(O);
            else
                ctx.backend.ReduceMean(X, O, axes);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceMin` reduction layer along the given axes: f(x1, x2 ... xn) = min(x1, x2, ... , xn).
    /// </summary>
    class ReduceMin : Reduce
    {
        static readonly string k_OpName = "ReduceMin";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceMin(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
            {
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<int>, int.MaxValue);
                else
                    ctx.backend.ReduceMin(X as Tensor<int>, O as Tensor<int>, axes);
            }
            else
            {
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<float>, float.PositiveInfinity);
                else
                    ctx.backend.ReduceMin(X as Tensor<float>, O as Tensor<float>, axes);
            }
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceProd` reduction layer along the given axes: f(x1, x2 ... xn) = x1 * x2 * ... * xn.
    /// </summary>
    class ReduceProd : Reduce
    {
        static readonly string k_OpName = "ReduceProd";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceProd(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
            {
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<int>, 1);
                else
                    ctx.backend.ReduceProd(X as Tensor<int>, O as Tensor<int>, axes);
            }
            else
            {
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<float>, 1);
                else
                    ctx.backend.ReduceProd(X as Tensor<float>, O as Tensor<float>, axes);
            }
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceSum` reduction layer along the given axes: f(x1, x2 ... xn) = x1 + x2 + ... + xn.
    /// </summary>
    class ReduceSum : Reduce
    {
        static readonly string k_OpName = "ReduceSum";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceSum(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X.shape.HasZeroDims())
            {
                ctx.backend.MemClear(O);
                return;
            }
            if (X is Tensor<int>)
                ctx.backend.ReduceSum(X as Tensor<int>, O as Tensor<int>, axes);
            else
                ctx.backend.ReduceSum(X as Tensor<float>, O as Tensor<float>, axes);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `ReduceSumSquare` reduction layer along the given axes: f(x1, x2 ... xn) = x1² + x2² + ... + xn².
    /// </summary>
    class ReduceSumSquare : Reduce
    {
        static readonly string k_OpName = "ReduceSumSquare";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public ReduceSumSquare(int output, int data, int axes = -1, bool keepdims = true, bool noopWithEmptyAxes = false)
            : base(output, data, axes, keepdims, noopWithEmptyAxes) { }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            if (noopWithEmptyAxes && (axes == null || axes.Length == 0))
            {
                var copyX = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
                ctx.backend.MemCopy(X, copyX);
                return;
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(axes, keepdims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (X.shape.HasZeroDims())
            {
                ctx.backend.MemClear(O);
                return;
            }
            if (X is Tensor<int>)
                ctx.backend.ReduceSumSquare(X as Tensor<int>, O as Tensor<int>, axes);
            else
                ctx.backend.ReduceSumSquare(X as Tensor<float>, O as Tensor<float>, axes);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
