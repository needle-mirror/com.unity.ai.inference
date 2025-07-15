using System;
using Unity.Collections;
using UnityEngine;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Options for the reduction operation to use in a scatter layer.
    /// </summary>
    enum ScatterReductionMode
    {
        /// <summary>
        /// Use no reduction.
        /// </summary>
        None,
        /// <summary>
        /// Use the addition operator when reducing.
        /// </summary>
        Add,
        /// <summary>
        /// Use the multiplication operator when reducing.
        /// </summary>
        Mul,
        /// <summary>
        /// Use the maximum operator when reducing.
        /// </summary>
        Max,
        /// <summary>
        /// Use the minimum operator when reducing.
        /// </summary>
        Min,
    }

    /// <summary>
    /// Represents a reduction which calculates indices.
    /// </summary>
    abstract class ArgReduce : Layer
    {
        public int axis;
        public bool keepdims;
        public bool selectLastIndex;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var shapeX = getPartialTensor(0).shape;
            if (!shapeX.hasRank)
            {
                setPartialTensor(0, new PartialTensor<int>());
                return;
            }

            var reducedShape = new DynamicTensorShape(shapeX);

            // reducing on a zero axis will result in a zero rather than a one
            if (shapeX[axis].isValue)
                reducedShape[axis] = shapeX[axis].value == 0 ? DynamicTensorDim.Zero : DynamicTensorDim.One;
            else
                reducedShape[axis] = DynamicTensorDim.Unknown;

            var shapeOut = !keepdims ? reducedShape.Squeeze(axis) : reducedShape;
            setPartialTensor(0, new PartialTensor<int>(shapeOut));
        }
    }

    /// <summary>
    /// Represents an `ArgMax` layer. This computes the indices of the maximum elements of the input tensor along a given axis.
    /// </summary>
    [Operator(category = "Indexing")]
    partial class ArgMax : ArgReduce
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shapeO = X.shape.Reduce(axis, keepdims);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.ArgMax(X as Tensor<int>, O, axis, selectLastIndex);
            else
                ctx.backend.ArgMax(X as Tensor<float>, O, axis, selectLastIndex);
        }
    }

    /// <summary>
    /// Represents an `ArgMin` layer. This computes the indices of the minimum elements of the input tensor along a given axis.
    /// </summary>
    [Operator(category = "Indexing")]
    partial class ArgMin : ArgReduce
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shapeO = X.shape.Reduce(axis, keepdims);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (O.shape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.ArgMin(X as Tensor<int>, O, axis, selectLastIndex);
            else
                ctx.backend.ArgMin(X as Tensor<float>, O, axis, selectLastIndex);
        }
    }

    /// <summary>
    /// Represents a `Gather` layer. This takes values from the input tensor indexed by the indices tensor along a given axis and concatenates them.
    /// </summary>
    [Operator(category = "Indexing")]
    [Inputs(names = new[] { "input", "indices" })]
    partial class Gather : Layer
    {
        public int axis;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var input = getPartialTensor(0);
            var dataType = input.dataType;
            var indices = getPartialTensor(1) as PartialTensor<int>;
            if (dataType == DataType.Int && axis == 0 && input.isPartiallyKnown && indices.isPartiallyKnown && input.shape.rank == 1 && indices.shape.rank <= 1)
            {
                var tensorOut = new PartialTensor<int>(indices.shape);
                var inputInt = input as PartialTensor<int>;
                for (var i = 0; i < indices.length; i++)
                {
                    var index = indices[i];
                    if (index.isValue && index.value < 0)
                        index = PartialTensorElement<int>.Value(index.value + input.length);
                    tensorOut[i] = inputInt[index];
                }

                setPartialTensor(0, tensorOut);
                return;
            }

            var shapeX = input.shape;
            var shapeIndices = indices.shape;
            if (!shapeX.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(dataType));
                return;
            }

            Logger.AssertIsTrue(shapeX.hasRank ? shapeX.rank >= 1 : true, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (!shapeIndices.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(dataType));
                return;
            }

            var axisX = shapeX.Axis(axis);

            var shapeOut = DynamicTensorShape.DynamicOfRank(shapeX.rank - 1 + shapeIndices.rank);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < axisX)
                    shapeOut[i] = shapeX[i];
                else if (i < axisX + shapeIndices.rank)
                    shapeOut[i] = shapeIndices[i - axisX];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Gather(X.shape, indices.shape, axis), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Gather(X, indices, O, axis);
        }
    }

    /// <summary>
    /// Represents a `GatherElements` layer. This takes values from the input tensor indexed by the `indices` tensor along a given axis.
    /// </summary>
    [Operator(category = "Indexing")]
    [Inputs(names = new[] { "input", "indices" })]
    partial class GatherElements : Layer
    {
        public int axis;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var shapeX = X.shape;
            var shapeIndices = getPartialTensor(1).shape;
            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            if (shapeX.hasRank)
            {
                shapeX.Axis(axis);
                shapeIndices.DeclareRank(shapeX.rank);
            }

            setPartialTensor(0, PartialTensor.Create(X.dataType, shapeIndices));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], indices.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GatherElements(X, indices, O, axis);
        }
    }

    /// <summary>
    /// Represents a `GatherND` layer. This takes slices of values from the batched input tensor indexed by the `indices` tensor.
    /// </summary>
    [Operator(category = "Indexing")]
    [Inputs(names = new[] { "input", "indices" })]
    partial class GatherND : Layer
    {
        public int batchDims;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var indices = getPartialTensor(1);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeIndices = indices.shape;
            // from https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= batchDims, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeX.rank);
            if (shapeIndices.hasRank)
                Logger.AssertIsTrue(shapeIndices.rank >= batchDims, "RankError: incorrect rank, expecting at least {0}, got {1}", batchDims, shapeIndices.rank);

            if (!shapeX.hasRank || !shapeIndices.hasRank || !shapeIndices[-1].isValue)
            {
                setPartialTensor(0, PartialTensor.Create(dataType));
                return;
            }

            Logger.AssertIsTrue(batchDims + shapeIndices[-1].value <= shapeX.rank, "GatherND.InputError: last indices dim too large");
            var shapeOut = DynamicTensorShape.DynamicOfRank(shapeX.rank + shapeIndices.rank - shapeIndices[-1].value - 1 - batchDims);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i < batchDims)
                    shapeOut[i] = DynamicTensorDim.MaxDefinedDim(shapeX[i], shapeIndices[i]);
                else if (i < shapeIndices.rank - 1)
                    shapeOut[i] = shapeIndices[i];
                else
                    shapeOut[i] = shapeX[i - shapeOut.rank];
            }

            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GatherND(X.shape, indices.shape, batchDims), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GatherND(X, indices, O, batchDims);
        }
    }

    /// <summary>
    /// Represents a `NonZero` layer. This returns the indices of the elements of the input tensor that are not zero.
    /// </summary>
    [Operator(category = "Indexing")]
    partial class NonZero : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var shapeX = X.shape;
            var shape = !shapeX.hasRank ? DynamicTensorShape.DynamicOfRank(2) : new DynamicTensorShape(DynamicTensorDim.Int(shapeX.rank), DynamicTensorDim.Unknown);
            setPartialTensor(0, new PartialTensor<int>(shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            // need to download, if gpucompute need to execute commandbuffer and flush.
            if (ctx.backend is GPUComputeBackend gpuBackend)
                gpuBackend.ExecuteCommandBufferAndClear();

            // pixel we don't know which dim to pin
            var outputBackendType = ctx.backend.backendType;
            if (outputBackendType == BackendType.GPUPixel)
                outputBackendType = BackendType.CPU;

            if (X is Tensor<int>)
            {
                // reduce(notequal(X, 0)) to get nbNonZeroIndices
                var arrayX = (X as Tensor<int>).DownloadToNativeArray();
                int nbNonZeroIndices = 0;
                for (int i = 0; i < X.shape.length; ++i)
                {
                    if (arrayX[i] != 0)
                        nbNonZeroIndices += 1;
                }

                // compact with condition mask?
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(X.shape.rank, nbNonZeroIndices), DataType.Int, outputBackendType) as Tensor<int>;
                if (O.shape.HasZeroDims())
                    return;
                var arrayO = new NativeArray<int>(O.shape.length, Allocator.Temp);

                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (arrayX[it.index] != 0)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            arrayO[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                        nonZeroIndicesIdx++;
                    }
                }
                O.dataOnBackend.Upload(arrayO, arrayO.Length);
            }
            else
            {
                // reduce(notequal(X, 0)) to get nbNonZeroIndices
                var arrayX = (X as Tensor<float>).DownloadToNativeArray();
                int nbNonZeroIndices = 0;
                for (int i = 0; i < X.shape.length; ++i)
                {
                    if (arrayX[i] != 0)
                        nbNonZeroIndices += 1;
                }

                // compact with condition mask?
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], new TensorShape(X.shape.rank, nbNonZeroIndices), DataType.Int, outputBackendType) as Tensor<int>;
                if (O.shape.HasZeroDims())
                    return;
                var arrayO = new NativeArray<int>(O.shape.length, Allocator.Temp);

                int nonZeroIndicesIdx = 0;
                for (var it = new TensorNDIterator(X.shape); it.HasNext(); it.MoveNext())
                {
                    if (arrayX[it.index] != 0)
                    {
                        for (int i = 0; i < X.shape.rank; i++)
                            arrayO[i * nbNonZeroIndices + nonZeroIndicesIdx] = it[i];
                        nonZeroIndicesIdx++;
                    }
                }
                O.dataOnBackend.Upload(arrayO, arrayO.Length);
            }
        }
    }

    /// <summary>
    /// Represents a `ScatterElements` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor along a given axis.
    ///
    /// `ScatterElements` updates the values depending on the reduction mode used.
    /// </summary>
    [Operator(category = "Indexing")]
    [Inputs(names = new[] { "input", "indices", "updates" })]
    partial class ScatterElements : Layer
    {
        public int axis;
        public ScatterReductionMode reduction;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeIndices = getPartialTensor(1).shape;
            var shapeUpdates = getPartialTensor(2).shape;

            if (!shapeX.hasRank && !shapeIndices.hasRank && !shapeUpdates.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(dataType));
                return;
            }

            if (!shapeX.hasRank && shapeIndices.hasRank)
                shapeX = DynamicTensorShape.DynamicOfRank(shapeIndices.rank);

            if (!shapeX.hasRank && shapeUpdates.hasRank)
                shapeX = DynamicTensorShape.DynamicOfRank(shapeUpdates.rank);

            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            shapeIndices.DeclareRank(shapeX.rank);
            shapeUpdates.DeclareRank(shapeX.rank);

            // throw error if axis incorrect
            shapeX.Axis(axis);

            // throw error if indices and updates don't match
            for (var i = 0; i < shapeIndices.rank; i++)
            {
                DynamicTensorDim.MaxDefinedDim(shapeIndices[i], shapeUpdates[i]);
            }

            setPartialTensor(0, PartialTensor.Create(dataType, shapeX));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            var updates = ctx.storage.GetTensor(inputs[2]);
            Logger.AssertIsTrue(indices.shape == updates.shape, "ScatterElements.InputError indices and updates must have same shape");
            if (indices.shape.HasZeroDims())
                ctx.backend.MemCopy(X, O);
            else
                ctx.backend.ScatterElements(X, indices, updates, O, axis, reduction);
        }
    }

    /// <summary>
    /// Represents a `ScatterND` layer. This copies the input tensor and updates values at indexes specified by the `indices` tensor with values specified by the `updates` tensor.
    ///
    /// `ScatterND` updates the values depending on the reduction mode used.
    /// </summary>
    [Operator(category = "Indexing")]
    [Inputs(names = new[] { "input", "indices", "updates" })]
    partial class ScatterND : Layer
    {
        public ScatterReductionMode reduction;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var dataType = X.dataType;
            var shapeX = X.shape;
            var shapeIndices = getPartialTensor(1).shape;
            var shapeUpdates = getPartialTensor(2).shape;

            if (shapeIndices.hasRank)
                Logger.AssertIsTrue(shapeIndices.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", shapeIndices.rank, 1);

            if (shapeIndices.hasRank && shapeUpdates.hasRank && shapeIndices[-1].isValue)
                shapeX.DeclareRank(shapeUpdates.rank - (shapeIndices.rank - shapeIndices[-1].value - 1));

            if (shapeX.hasRank)
                Logger.AssertIsTrue(shapeX.rank >= 1, "RankError: incorrect rank, expecting at least {0}, got {1}", 1, shapeX.rank);

            setPartialTensor(0, PartialTensor.Create(dataType, shapeX));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var indices = ctx.storage.GetTensor(inputs[1]) as Tensor<int>;
            if (indices.shape.HasZeroDims())
            {
                ctx.backend.MemCopy(X, O);
                return;
            }

            if (X is Tensor<int>)
                ctx.backend.ScatterND(X as Tensor<int>, ctx.storage.GetTensor(inputs[1]) as Tensor<int>, ctx.storage.GetTensor(inputs[2]) as Tensor<int>, O as Tensor<int>, reduction);
            else
                ctx.backend.ScatterND(X as Tensor<float>, ctx.storage.GetTensor(inputs[1]) as Tensor<int>, ctx.storage.GetTensor(inputs[2]) as Tensor<float>, O as Tensor<float>, reduction);
        }
    }

    /// <summary>
    /// Represents a `TopK` layer. This calculates the top-K largest or smallest elements of an input tensor along a given axis.
    ///
    /// This layer calculates both the values tensor of the top-K elements and the indices tensor of the top-K elements as outputs.
    /// </summary>
    [Operator(category = "Indexing")]
    [Inputs(names = new[] { "input", "k" }, inputCPURead = new[] { 1 })]
    [Outputs(names = new[] { "values", "indices" })]
    partial class TopK : Layer
    {
        public int axis;
        public bool largest;
        public bool sorted;

        internal override int OutputCount => 2;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var K = getPartialTensor(1) as PartialTensor<int>;
            var dataType = X.dataType;
            var shapeX = X.shape;
            if (!shapeX.hasRank)
            {
                setPartialTensor(1, new PartialTensor<int>());
                setPartialTensor(0, PartialTensor.Create(dataType));
                return;
            }

            var shapeK = K.shape;
            shapeK.DeclareRank(1);
            Logger.AssertIsFalse(shapeK[0] != 1, "TopK.InputError: k must be a single value");

            var shapeOut = new DynamicTensorShape(shapeX);

            var axisX = shapeX.Axis(axis);

            shapeOut[axisX] = (DynamicTensorDim)K[0];

            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
            setPartialTensor(1, new PartialTensor<int>(shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var k = ctx.storage.GetInt(inputs[1]);
            var outputShape = new TensorShape(X.shape);
            outputShape[axis] = k;

            var values = ctx.storage.AllocateTensorAndStore(outputs[0], outputShape, X.dataType, ctx.backend.backendType);
            var indices = ctx.storage.AllocateTensorAndStore(outputs[1], outputShape, DataType.Int, ctx.backend.backendType) as Tensor<int>;
            if (outputShape.HasZeroDims())
                return;
            if (X is Tensor<int>)
                ctx.backend.TopK(X as Tensor<int>, values as Tensor<int>, indices, k, axis, largest);
            else
                ctx.backend.TopK(X as Tensor<float>, values as Tensor<float>, indices, k, axis, largest);
        }
    }
}
