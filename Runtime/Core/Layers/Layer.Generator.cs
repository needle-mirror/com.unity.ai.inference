using System;
using Unity.Profiling;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents a `ConstantOfShape` layer. This generates a tensor with the shape given by the `input` tensor and filled with a given value.
    /// </summary>
    class ConstantOfShape : Layer
    {
        static readonly string k_OpName = "ConstantOfShape";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public DataType dataType;
        public float floatValue;
        public int intValue;

        public ConstantOfShape(int output, int input, int value)
            : base(new[] { output }, new[] { input })
        {
            dataType = DataType.Int;
            intValue = value;
        }

        public ConstantOfShape(int output, int input, float value)
            : base(new[] { output }, new[] { input })
        {
            dataType = DataType.Float;
            floatValue = value;
        }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var shape = DynamicTensorShape.FromPartialTensor(getPartialTensor(0) as PartialTensor<int>);
            var tensorOut = PartialTensor.Create(dataType, shape);
            if (!tensorOut.isPartiallyKnown)
            {
                setPartialTensor(0, tensorOut);
                return;
            }

            if (tensorOut is PartialTensor<int> tensorInt)
            {
                for (var i = 0; i < tensorInt.length; i++)
                    tensorInt[i] = PartialTensorElement<int>.Value(intValue);
            }
            else if (tensorOut is PartialTensor<float> tensorFloat)
            {
                for (var i = 0; i < tensorFloat.length; i++)
                    tensorFloat[i] = PartialTensorElement<float>.Value(floatValue);
            }

            setPartialTensor(0, tensorOut);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            TensorShape shape = new TensorShape(ctx.storage.GetInts(inputs[0]));
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Int)
                ctx.backend.MemSet(O as Tensor<int>, intValue);
            else
                ctx.backend.MemSet(O as Tensor<float>, floatValue);
            return;
        }

        public override string ToString()
        {
            return $"{base.ToString()}, dataType: {dataType}, floatValue: {floatValue}, intValue: {intValue}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `OneHot` layer. This generates a one-hot tensor with a given `depth`, `indices` and `values`.
    /// </summary>
    class OneHot : Layer
    {
        static readonly string k_OpName = "OneHot";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);
        public int axis;
        public bool allowNegativeIndexes;

        public OneHot(int output, int indices, int depth, int values, int axis, bool allowNegativeIndexes)
            : base(new[] { output }, new[] { indices, depth, values })
        {
            this.axis = axis;
            this.allowNegativeIndexes = allowNegativeIndexes;
        }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var values = getPartialTensor(2);
            var shapeX = X.shape;
            var dataType = values.dataType;
            if (!shapeX.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(dataType));
                return;
            }

            var shapeOut = shapeX.Unsqueeze(axis);
            shapeOut[axis] = (DynamicTensorDim)(getPartialTensor(1) as PartialTensor<int>)[0];

            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var indices = ctx.storage.GetTensor(inputs[0]) as Tensor<int>;
            var depth = ctx.storage.GetInt(inputs[1]);
            var dataType = ctx.storage.GetDataType(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.OneHot(indices.shape, axis, depth), dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (dataType == DataType.Int)
            {
                var valuesi = ctx.storage.GetInts(inputs[2]);
                ctx.backend.OneHot(indices, O as Tensor<int>, axis, depth, valuesi[0], valuesi[1], allowNegativeIndexes);
            }
            else
            {
                var valuesf = ctx.storage.GetFloats(inputs[2]);
                ctx.backend.OneHot(indices, O as Tensor<float>, axis, depth, valuesf[0], valuesf[1], allowNegativeIndexes);
            }
        }

        public override string ToString()
        {
            return $"{base.ToString()}, axis: {axis}";
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }

    /// <summary>
    /// Represents a `Range` layer. This generates a 1D output tensor where the values form an arithmetic progression defined by the `start`, `limit` and `delta` scalar input tensors.
    /// </summary>
    class Range : Layer
    {
        static readonly string k_OpName = "Range";
        static readonly ProfilerMarker k_ProfilerMarker = new(k_ProfilerMarkerPrefix + k_OpName);

        public Range(int output, int start, int limit, int delta)
            : base(new[] { output }, new[] { start, limit, delta }) { }

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var start = getPartialTensor(0);
            var limit = getPartialTensor(1);
            var delta = getPartialTensor(2);

            start.shape.DeclareRank(0);
            limit.shape.DeclareRank(0);
            delta.shape.DeclareRank(0);

            var shape = DynamicTensorShape.DynamicOfRank(1);

            if (start.dataType == DataType.Int && start.Get<int>() == PartialTensorElement<int>.Zero && delta.Get<int>() == PartialTensorElement<int>.One)
                shape[0] = (DynamicTensorDim)limit.Get<int>();

            if (start.dataType == DataType.Int && start.IsStatic() && limit.IsStatic() && delta.IsStatic())
                shape[0] = DynamicTensorDim.Int(ShapeInference.Range(start.Get<int>().value, limit.Get<int>().value, delta.Get<int>().value)[0]);

            if (start.dataType == DataType.Float && start.IsStatic() && limit.IsStatic() && delta.IsStatic())
                shape[0] = DynamicTensorDim.Int(ShapeInference.Range(start.Get<float>().value, limit.Get<float>().value, delta.Get<float>().value)[0]);

            setPartialTensor(0, PartialTensor.Create(start.dataType, shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            if (ctx.storage.GetDataType(inputs[0]) == DataType.Int)
            {
                int starti = ctx.storage.GetInt(inputs[0]);
                int limiti = ctx.storage.GetInt(inputs[1]);
                int deltai = ctx.storage.GetInt(inputs[2]);
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Range(starti, limiti, deltai), DataType.Int, ctx.backend.backendType) as Tensor<int>;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, starti, deltai);
            }
            else
            {
                float startf = ctx.storage.GetFloat(inputs[0]);
                float limitf = ctx.storage.GetFloat(inputs[1]);
                float deltaf = ctx.storage.GetFloat(inputs[2]);
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Range(startf, limitf, deltaf), DataType.Float, ctx.backend.backendType) as Tensor<float>;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Range(O, startf, deltaf);
            }
        }

        public override string opName => k_OpName;
        public override ProfilerMarker profilerMarker => k_ProfilerMarker;
    }
}
