using System;
using Unity.Profiling;
using UnityEngine;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Takes two 1D int tensors representing shapes and returns an int tensor representing the shape when you broadcast these shapes together with numpy-style broadcasting.
    ///
    /// e.g. BroadcastArgs([1, 1, 2, 3], [5, 2, 1]) returns [1, 5, 2, 3]
    /// </summary>
    class BroadcastArgs : Layer
    {
        static readonly string k_OpName = "BroadcastArgs";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public BroadcastArgs(int output, int a, int b)
            : base(new[] { output }, new[] { a, b }) { }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var a = getPartialTensor(0) as PartialTensor<int>;
            var b = getPartialTensor(1) as PartialTensor<int>;
            var shapeOut = DynamicTensorShape.DynamicOfRank(1);
            if (a.shape.IsStatic() && b.shape.IsStatic())
                shapeOut[0] = DynamicTensorDim.Int(Mathf.Max(a.shape[0].ToInt(), b.shape[0].ToInt()));
            var tensorOut = new PartialTensor<int>(shapeOut);
            if (tensorOut.isPartiallyKnown)
            {
                for (var i = 0; i < tensorOut.length; i++)
                    tensorOut[i] = PartialTensorElement<int>.Value(1);
                for (var i = 0; i < a.length; i++)
                    tensorOut[tensorOut.length - 1 - i] = (PartialTensorElement<int>)DynamicTensorDim.Broadcast((DynamicTensorDim)a[a.length - 1 - i], (DynamicTensorDim)tensorOut[tensorOut.length - 1 - i]);
                for (var i = 0; i < b.length; i++)
                    tensorOut[tensorOut.length - 1 - i] = (PartialTensorElement<int>)DynamicTensorDim.Broadcast((DynamicTensorDim)b[b.length - 1 - i], (DynamicTensorDim)tensorOut[tensorOut.length - 1 - i]);
            }
            setPartialTensor(0, tensorOut);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var a = ctx.storage.GetInts(inputs[0]);
            var b = ctx.storage.GetInts(inputs[1]);
            var shape = new TensorShape(a).Broadcast(new TensorShape(b));
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(shape.rank), DataType.Int, BackendType.CPU) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;

            ctx.cpuBackend.SetShape(O, shape);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Shape` layer. This computes the shape of an input tensor as a 1D `Tensor<int>`.
    /// </summary>
    class Shape : Layer
    {
        static readonly string k_OpName = "Shape";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int start;
        public int end;

        public Shape(int output, int input, int start = 0, int end = TensorShape.maxRank)
            : base(new[] { output }, new[] { input })
        {
            this.start = start;
            this.end = end;
        }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            if (start == end)
            {
                setPartialTensor(0, new PartialTensor<int>(new DynamicTensorShape(DynamicTensorDim.Zero)));
                return;
            }

            var shapeX = getPartialTensor(0).shape;

            if (!shapeX.hasRank)
            {
                setPartialTensor(0, new PartialTensor<int>(DynamicTensorShape.DynamicOfRank(1)));
                return;
            }

            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "PartialTensorFromSymbolicShape.InputError: start value cannot be greater than end value for shape slicing");

            var tensorOut = new PartialTensor<int>(new DynamicTensorShape(endX - startX));
            for (var i = startX; i < endX; i++)
            {
                tensorOut[i - startX] = (PartialTensorElement<int>)shapeX[i];
            }

            setPartialTensor(0, tensorOut);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var shapeX = ctx.storage.GetTensorShape(inputs[0]);
            var startX = start < 0 ? start + shapeX.rank : start;
            var endX = end < 0 ? end + shapeX.rank : end;
            startX = Mathf.Clamp(startX, 0, shapeX.rank);
            endX = Mathf.Clamp(endX, 0, shapeX.rank);

            Logger.AssertIsTrue(endX >= startX, "Shape.InputError: start value cannot be greater than end value for shape slicing");
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(endX - startX), DataType.Int, BackendType.CPU) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;

            var shape = TensorShape.Ones(endX - startX);
            for (var i = startX; i < endX; i++)
                shape[i - startX] = shapeX[i];
            ctx.cpuBackend.SetShape(O, shape);
        }

        public override string ToString()
        {
            return $"{base.ToString()}, start: {start}, end: {end}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Size` layer. This computes the number of elements of an input tensor as a scalar `Tensor<int>`.
    /// </summary>
    class Size : Layer
    {
        static readonly string k_OpName = "Size";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Size(int output, int input)
            : base(new[] { output }, new[] { input }) { }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            setPartialTensor(0, new PartialTensor<int>(new DynamicTensorShape())
            {
                [0] = (PartialTensorElement<int>)X.shape.Length()
            });
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var shapeX = ctx.storage.GetTensorShape(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(), DataType.Int, BackendType.CPU) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;

            ctx.cpuBackend.MemSet(O, shapeX.length);
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
