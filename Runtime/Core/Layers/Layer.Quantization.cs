using System;
using UnityEngine;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents a dequantize layer where four uint8 values are packed per int value.
    /// The final float values are calculated as y = (x - zeroPoint) * scale.
    /// </summary>
    [Operator(category = "Quantization")]
    partial class DequantizeUint8 : Layer
    {
        public float scale;
        public byte zeroPoint;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var shapeX = getPartialTensor(0).shape;
            setPartialTensor(0, new PartialTensor<float>(shapeX));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<byte>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DequantizeLinear(X, O, scale, zeroPoint);
        }
    }
}
