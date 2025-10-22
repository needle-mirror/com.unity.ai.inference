using System;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.InferenceEngine.Editor.LiteRT
{
    static class LiteRTTensorExtensions
    {
        public static int[] Shape(this Tensor tensor)
        {
            var shapeSignatureArray = tensor.GetShapeSignatureArray();
            var shapeArray = tensor.GetShapeArray();
            shapeArray ??= Array.Empty<int>();
            return shapeSignatureArray ?? shapeArray;
        }

        public static DynamicTensorShape DynamicShape(this Tensor tensor)
        {
            var shapeSignatureArray = tensor.GetShapeSignatureArray();
            var shapeArray = tensor.GetShapeArray();
            return new DynamicTensorShape(shapeSignatureArray ?? shapeArray);
        }

        public static DataType GetDataType(this Tensor tensor)
        {
            return ToDataType(tensor.Type);
        }

        static DataType ToDataType(this TensorType tensorType)
        {
            return tensorType switch
            {
                TensorType.FLOAT32 or TensorType.FLOAT16 or TensorType.FLOAT64 or TensorType.BFLOAT16 => DataType.Float,
                TensorType.INT32 or TensorType.UINT8 or TensorType.INT64 or TensorType.BOOL or TensorType.INT16 or TensorType.INT8 or TensorType.UINT64 or TensorType.UINT32 or TensorType.UINT16 or TensorType.INT4 => DataType.Int,
                TensorType.STRING or TensorType.COMPLEX64 or TensorType.COMPLEX128 or TensorType.RESOURCE or TensorType.VARIANT => throw new LiteRTImportException($"Tensor type {tensorType} is not supported"),
                _ => throw new LiteRTImportException($"Tensor type {tensorType} is not supported")
            };
        }

        public static ConstantTensor GetConstant(this Tensor tensor, Buffer buffer)
        {
            if (tensor.Sparsity.HasValue)
                throw new LiteRTImportException("Sentis does not support sparse tensors");
            var shape = new TensorShape(tensor.GetShapeArray());
            var data = shape.length == 0 ? Array.Empty<byte>() : buffer.GetDataArray();

            switch (tensor.Type)
            {
                case TensorType.FLOAT16:
                {
                    return ConstantTensor.FloatFromFloat16(shape, data);
                }
                case TensorType.FLOAT32:
                {
                    return new ConstantTensor(shape, DataType.Float, data);
                }
                case TensorType.FLOAT64:
                {
                    return ConstantTensor.FloatFromFloat64(shape, data);
                }
                case TensorType.BOOL:
                {
                    return ConstantTensor.IntFromBool(shape, data);
                }
                case TensorType.UINT8:
                {
                    return ConstantTensor.IntFromUint8(shape, data);
                }
                case TensorType.UINT16:
                {
                    return ConstantTensor.IntFromUint16(shape, data);
                }
                case TensorType.UINT32:
                {
                    return ConstantTensor.IntFromUint32(shape, data);
                }
                case TensorType.UINT64:
                {
                    return ConstantTensor.IntFromUint64(shape, data);
                }
                case TensorType.INT8:
                {
                    return ConstantTensor.IntFromInt8(shape, data);
                }
                case TensorType.INT16:
                {
                    return ConstantTensor.IntFromInt16(shape, data);
                }
                case TensorType.INT32:
                {
                    return new ConstantTensor(shape, DataType.Int, data);
                }
                case TensorType.INT64:
                {
                    return ConstantTensor.IntFromInt64(shape, data);
                }
                case TensorType.STRING:
                case TensorType.COMPLEX64:
                case TensorType.COMPLEX128:
                case TensorType.RESOURCE:
                case TensorType.VARIANT:
                case TensorType.INT4:
                case TensorType.BFLOAT16:
                default:
                    throw new LiteRTImportException($"Input constant tensor \"{tensor.Name}\": type {tensor.Type} is not supported");
            }
        }
    }
}
