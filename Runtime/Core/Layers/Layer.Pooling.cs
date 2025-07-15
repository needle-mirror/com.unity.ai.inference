using System;
using UnityEngine;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents a local pooling layer.
    /// </summary>
    abstract class LocalPool : Layer
    {
        public int[] kernelShape;
        public int[] strides;
        public int[] pads;
        public AutoPad autopad;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var dataType = X.dataType;
            var shapeX = X.shape;
            shapeX.DeclareRank(2 + kernelShape.Length);

            Logger.AssertIsTrue(strides == null || shapeX.rank - 2 == strides.Length, "Pool.InputError: strides must have same number of values as spatial dimensions or be null");
            Logger.AssertIsTrue(pads == null || (shapeX.rank - 2) * 2 == pads.Length, "Pool.InputError: padding must have twice the number of values as spatial dimensions or be null");

            var shapeOut = new DynamicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                var s = strides == null ? 1 : strides[i - 2];
                var p = (pads == null || autopad != AutoPad.NotSet) ? 0 : (pads[i - 2] + pads[i - 2 + (shapeX.rank - 2)]);
                shapeOut[i] = shapeX[i].Pool(kernelShape[i - 2], s, p, 1, false, autopad);
            }

            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }

        public override string ToString()
        {
            return $"{base.ToString()}, kernelShape: [{string.Join(", ", kernelShape)}], strides: [{string.Join(", ", strides)}], pads: [{string.Join(", ", pads)}], autopad: {autopad}";
        }
    }

    /// <summary>
    /// Represents a global pooling layer.
    /// </summary>
    abstract class GlobalPool : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var dataType = X.dataType;
            var shapeX = X.shape;
            if (!shapeX.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 3 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 3, shapeX.rank);

            var shapeOut = new DynamicTensorShape(shapeX);

            for (var i = 2; i < shapeOut.rank; i++)
            {
                shapeOut[i] = DynamicTensorDim.One;
            }

            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }
    }

    /// <summary>
    /// Represents an `AveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    [Operator(category = "Pooling")]
    partial class AveragePool : LocalPool
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            ShapeInference.UpdatePadForPoolAutoPadding(X.shape, kernelShape, strides, pads, false, autopad);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.ApplyPool(X.shape, kernelShape, strides, pads), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.AveragePool(X, O, kernelShape, strides, pads);
        }
    }

    /// <summary>
    /// Represents a `GlobalAveragePool` pooling layer. This calculates an output tensor by pooling the mean values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    [Operator(category = "Pooling")]
    partial class GlobalAveragePool : GlobalPool
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GlobalPool(X.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GlobalAveragePool(X, O);
        }
    }

    /// <summary>
    /// Represents a `GlobalMaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across all of its spatial dimensions. The spatial dimensions of the output are size 1.
    /// </summary>
    [Operator(category = "Pooling")]
    partial class GlobalMaxPool : GlobalPool
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GlobalPool(X.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GlobalMaxPool(X, O);
        }
    }

    /// <summary>
    /// Represents a `MaxPool` pooling layer. This calculates an output tensor by pooling the maximum values of the input tensor across its spatial dimensions according to the given pool and stride values.
    /// </summary>
    [Operator(category = "Pooling")]
    partial class MaxPool : LocalPool
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            ShapeInference.UpdatePadForPoolAutoPadding(X.shape, kernelShape, strides, pads, false, autopad);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.ApplyPool(X.shape, kernelShape, strides, pads), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.MaxPool(X, O, kernelShape, strides, pads);
        }
    }
}
