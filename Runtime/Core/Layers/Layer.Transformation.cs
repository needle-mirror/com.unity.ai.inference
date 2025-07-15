using System;
using UnityEngine;
using UnityEngine.Assertions;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Options for the padding values for `Pad`.
    /// </summary>
    enum PadMode
    {
        /// <summary>
        /// Use a constant value for the padded data.
        /// </summary>
        Constant,
        /// <summary>
        /// Use the reflection of the values of the input tensor mirrored on the first and last values along the axis. The edge values appear once in the output tensor.
        /// </summary>
        Reflect,
        /// <summary>
        /// Use the edge values of the input tensor.
        /// </summary>
        Edge,
        /// <summary>
        /// Use the reflection of the values of the input tensor mirrored half a step outside the first and last values along the axis. The edge values appear twice in the output tensor.
        /// </summary>
        Symmetric,
        /// <summary>
        /// Wrap the values of the input tensor like a torus for the padded data.
        /// </summary>
        Wrap,
    }

    /// <summary>
    /// Options for the scaling mode to use for `Resize`.
    /// </summary>
    enum ScaleMode
    {
        /// <summary>
        /// Use the size tensor directly for the shape of the output tensor.
        /// </summary>
        Sizes,
        /// <summary>
        /// Use the scales tensor to multiply the shape of the input tensor to calculate the shape of the output tensor.
        /// </summary>
        Scales
    }

    /// <summary>
    /// Options for the interpolation mode to use for `Resize`.
    /// </summary>
    enum InterpolationMode
    {
        /// <summary>
        /// Use the nearest element to the calculated coordinate. The exact behaviour depends on `nearestMode`.
        /// </summary>
        Nearest,
        /// <summary>
        /// Use a linear sampling of the surrounding elements to the calculated coordinate.
        /// </summary>
        Linear,
        /// <summary>
        /// Use a cubic sampling of the surrounding elements to the calculated coordinate.
        /// </summary>
        Cubic
    }

    /// <summary>
    /// Options for how to sample the nearest element in `Resize` when using `InterpolationMode.NearestMode`.
    /// </summary>
    enum NearestMode
    {
        /// <summary>
        /// Use rounding to the nearest integer coordinate. If the fractional part equals 0.5 then round down.
        /// </summary>
        RoundPreferFloor,
        /// <summary>
        /// Use rounding to the nearest integer coordinate. If the fractional part equals 0.5 then round up.
        /// </summary>
        RoundPreferCeil,
        /// <summary>
        /// Use rounding down to the next integer coordinate less than or equal to the input coordinate.
        /// </summary>
        Floor,
        /// <summary>
        /// Use rounding up to the next integer coordinate greater than or equal to the input coordinate.
        /// </summary>
        Ceil
    }

    /// <summary>
    /// Padding mode for outside grid values.
    /// </summary>
    enum PaddingMode
    {
        /// <summary>
        /// Use 0 for out-of-bound grid locations.
        /// </summary>
        Zeros,
        /// <summary>
        /// Use border value for out-of-bound grid locations.
        /// </summary>
        Border,
        /// <summary>
        /// Use values at locations reflected by the border for out-of-bound grid locations. Distant values are reflected multiple times until in bounds.
        /// </summary>
        Reflection
    }

    /// <summary>
    /// Options for how to transform between the coordinate in the output tensor and the coordinate in the input tensor in `Resize`.
    /// </summary>
    enum CoordTransformMode
    {
        /// <summary>
        /// Use shifting by half a pixel before and after scaling.
        /// </summary>
        HalfPixel,
        /// <summary>
        /// Use shifting by half a pixel before and after scaling if the output length is greater than 1, otherwise use 0.
        /// </summary>
        PytorchHalfPixel,
        /// <summary>
        /// Use scaling by `length - 1` so that corner pixels align.
        /// </summary>
        AlignCorners,
        /// <summary>
        /// Use direct scaling of coordinates by the scaling factor.
        /// </summary>
        Asymmetric,
    }

    /// <summary>
    /// Options for which part of the input matrix to retain in `Trilu`.
    /// </summary>
    enum TriluMode
    {
        /// <summary>
        /// Use retaining of the lower part of the input matrix.
        /// </summary>
        Lower = 0,
        /// <summary>
        /// Use retaining of the upper part of the input matrix.
        /// </summary>
        Upper = 1,
    }

    /// <summary>
    /// Options for the ordering of the elements in `DepthToSpace`.
    /// </summary>
    enum DepthToSpaceMode
    {
        /// <summary>
        /// Use depth, column, row ordering where the data is arranged (by * blocksize * channels) + (bx * channels) + c.
        /// </summary>
        DepthColumnRow,
        /// <summary>
        /// Use column, row, depth ordering where the data is arranged (c * blocksize * blocksize) + (by * blocksize) + bx.
        /// </summary>
        ColumnRowDepth,
    }

    /// <summary>
    /// Represents an element-wise `Cast` layer: f(x) = (float)x or f(x) = (int)x depending on the value of `toType`.
    /// </summary>
    [Operator(category = "Transformation")]
    partial class Cast : Layer
    {
        public DataType toType;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            setPartialTensor(0, PartialTensor.Create(toType, getPartialTensor(0).shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, toType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;

            if (X.dataType == O.dataType)
                ctx.backend.MemCopy(X, O);
            else if (X.dataType == DataType.Int && O.dataType == DataType.Float)
                ctx.backend.Cast(X as Tensor<int>, O as Tensor<float>);
            else if (X.dataType == DataType.Float && O.dataType == DataType.Int)
                ctx.backend.Cast(X as Tensor<float>, O as Tensor<int>);
            else if (X.dataType == DataType.Short && O.dataType == DataType.Float)
                ctx.backend.Cast(X as Tensor<short>, O as Tensor<float>);
            else
                throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Represents an element-wise `CastLike` layer: f(x) = (float)x or f(x) = (int)x depending on the data type of the targetType tensor.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "targetType" }, inputNoDataDependency = new[] { 1 })]
    partial class CastLike : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var toType = getPartialTensor(1).dataType;
            setPartialTensor(0, PartialTensor.Create(toType, getPartialTensor(0).shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var dataType = ctx.storage.GetDataType(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;

            if (X.dataType == dataType)
                ctx.backend.MemCopy(X, O);
            else if (X.dataType == DataType.Int && dataType == DataType.Float)
                ctx.backend.Cast(X as Tensor<int>, O as Tensor<float>);
            else if (X.dataType == DataType.Float && dataType == DataType.Int)
                ctx.backend.Cast(X as Tensor<float>, O as Tensor<int>);
            else if (X.dataType == DataType.Short && dataType == DataType.Float)
                ctx.backend.Cast(X as Tensor<short>, O as Tensor<float>);
            else
                throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Represents a `Concat` concatenation layer. The layer computes the output tensor by concatenating the input tensors along a given axis.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(isVariadic = true)]
    partial class Concat : Layer
    {
        public int axis;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            Logger.AssertIsTrue(inputs.Length > 0, "Concat.InputError: can't broadcast shapes array of size 0");
            var inputTensors = new PartialTensor[inputs.Length];
            for (var i = 0; i < inputs.Length; i++)
                inputTensors[i] = getPartialTensor(i);

            var dataType = inputTensors[0].dataType;

            var rank = DynamicTensorDim.Unknown;
            foreach (var tensorInput in inputTensors)
            {
                if (tensorInput.shape.hasRank)
                    rank = DynamicTensorDim.MaxDefinedDim(rank, DynamicTensorDim.Int(tensorInput.shape.rank));
            }

            if (rank.isUnknown)
            {
                setPartialTensor(0, PartialTensor.Create(dataType, DynamicTensorShape.DynamicRank));
                return;
            }

            foreach (var tensorInput in inputTensors)
                tensorInput.shape.DeclareRank(rank.value);

            var shapeOut = DynamicTensorShape.DynamicOfRank(rank.value);
            var axisOut = shapeOut.Axis(axis);

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (i == axisOut)
                {
                    shapeOut[i] = DynamicTensorDim.Zero;
                    foreach (var tensorInput in inputTensors)
                    {
                        shapeOut[i] += tensorInput.shape[i];
                    }
                }
                else
                {
                    shapeOut[i] = DynamicTensorDim.Unknown;
                    foreach (var tensorInput in inputTensors)
                    {
                        shapeOut[i] = DynamicTensorDim.MaxDefinedDim(shapeOut[i], tensorInput.shape[i]);
                    }
                }
            }

            var tensorOut = PartialTensor.Create(dataType, shapeOut);

            if (shapeOut.rank != 1 || !tensorOut.isPartiallyKnown)
            {
                setPartialTensor(0, tensorOut);
                return;
            }

            var index = 0;
            foreach (var X in inputTensors)
            {
                for (var i = 0; i < X.length; i++)
                {
                    tensorOut.CopyElement(index++, X, i);
                }
            }

            setPartialTensor(0, tensorOut);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var shapeO = ctx.storage.GetTensorShape(inputs[0]);
            for (var i = 1; i < inputs.Length; i++)
            {
                var shape = ctx.storage.GetTensorShape(inputs[i]);
                shapeO = shapeO.Concat(shape, axis);
            }
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeO, ctx.storage.GetDataType(inputs[0]), ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            // this is necessary for not propagating NaN values
            if (ctx.backend.backendType == BackendType.GPUPixel)
                ctx.backend.MemClear(O);
            var start = 0;
            for (var i = 0; i < inputs.Length; i++)
            {
                var X = ctx.storage.GetTensor(inputs[i]);
                var length = X.shape[axis];
                if (length == 0)
                    continue;
                ctx.backend.SliceSet(X, O, axis, start, 1);
                start += length;
            }
        }
    }

    /// <summary>
    /// Represents a `DepthToSpace` layer. The layer computes the output tensor by permuting data from depth into blocks of spatial data.
    /// </summary>
    [Operator(category = "Transformation")]
    partial class DepthToSpace : Layer
    {
        public int blocksize;
        public DepthToSpaceMode mode;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var shapeX = X.shape;
            shapeX.DeclareRank(4);
            setPartialTensor(0, PartialTensor.Create(X.dataType, new DynamicTensorShape(shapeX[0], shapeX[1] / (blocksize * blocksize), shapeX[2] * blocksize, shapeX[3] * blocksize)));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.DepthToSpace(X.shape, blocksize), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.DepthToSpace(X, O, blocksize, mode);
        }
    }

    /// <summary>
    /// Represents an `Expand` layer. The layer computes the output tensor by broadcasting the input tensor into a given shape.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "shape" }, inputCPURead = new[] { 1 })]
    partial class Expand : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var shape = DynamicTensorShape.FromPartialTensor(getPartialTensor(1) as PartialTensor<int>);
            setPartialTensor(0, PartialTensor.Create(X.dataType, shape.Broadcast(X.shape)));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shape = new TensorShape(ctx.storage.GetInts(inputs[1]));
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Broadcast(shape), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Expand(X, O);
        }
    }

    /// <summary>
    /// Represents a `Flatten` layer. The layer computes the output tensor by reshaping the input tensor into a 2D matrix according to the given axis.
    /// </summary>
    [Operator(category = "Transformation")]
    partial class Flatten : Layer
    {
        public int axis;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var shapeX = X.shape;
            if (!shapeX.hasRank)
            {
                if (axis == 0)
                    setPartialTensor(0, X.Reshape(new DynamicTensorShape(DynamicTensorDim.One, shapeX.Length())));
                else
                    setPartialTensor(0, X.Reshape(DynamicTensorShape.DynamicOfRank(2)));
                return;
            }

            var axisX = axis >= 0 ? axis : shapeX.rank + axis;

            var shapeOut = DynamicTensorShape.Ones(2);
            for (var i = 0; i < axisX; i++)
            {
                shapeOut[0] *= shapeX[i];
            }
            for (var i = axisX; i < shapeX.rank; i++)
            {
                shapeOut[1] *= shapeX[i];
            }

            setPartialTensor(0, X.Reshape(shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shape = X.shape.Flatten(axis);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O);
        }
    }

    /// <summary>
    /// Represents a `GridSample` layer. The layer computes the output tensor by sampling the input tensor with coordinates given by the grid tensor.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "grid" })]
    partial class GridSample : Layer
    {
        public InterpolationMode mode;
        public PaddingMode paddingMode;
        public bool alignCorners;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var grid = getPartialTensor(1);

            var outShape = DynamicTensorShape.DynamicRank;

            if (X.shape.hasRank)
                outShape.DeclareRank(X.shape.rank);
            if (grid.shape.hasRank)
                outShape.DeclareRank(grid.shape.rank);

            for (var i = 0; i < (outShape.hasRank ? outShape.rank : 0); i++)
            {
                outShape[i] = i switch
                {
                    0 => DynamicTensorDim.MaxDefinedDim(X.shape[0], grid.shape[0]),
                    1 => X.shape[i],
                    _ => grid.shape[i - 1]
                };
            }

            setPartialTensor(0, PartialTensor.Create(X.dataType, outShape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var grid = ctx.storage.GetTensor(inputs[1]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.GridSample(X.shape, grid.shape), DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.GridSample(X, grid, O, mode, paddingMode, alignCorners);
        }
    }

    /// <summary>
    /// Represents an `Identity` layer. The output tensor is a copy of the input tensor.
    /// </summary>
    [Operator(category = "Transformation")]
    partial class Identity : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            setPartialTensor(0, getPartialTensor(0));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.MemCopy(X, O);
        }
    }

    /// <summary>
    /// Represents a `MoveDim` layer. The layer computes the output tensor by moving the dimensions of input at the positions in source to the positions in destination.
    ///
    /// Other dimensions of input that are not explicitly moved remain in their original order and appear at the positions not specified in destination.
    /// </summary>
    [Operator(category = "Transformation")]
    partial class MoveDim : Layer
    {
        public int[] source;
        public int[] destination;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var shapeX = getPartialTensor(0).shape;

            if (!shapeX.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(getPartialTensor(0).dataType));
                return;
            }

            var shapeOut = DynamicTensorShape.DynamicOfRank(shapeX.rank);

            // move given dims
            uint srcAxesBitMask = 0;
            uint dstAxesBitMask = 0;
            for (var i = 0; i < source.Length; i++)
            {
                var srcAxis = shapeX.Axis(source[i]);
                var dstAxis = shapeX.Axis(destination[i]);
                Logger.AssertIsTrue(((srcAxesBitMask >> srcAxis) & 1U) == 0, "MoveDim.ValueError: source dims may not repeat");
                Logger.AssertIsTrue(((dstAxesBitMask >> dstAxis) & 1U) == 0, "MoveDim.ValueError: destination dims may not repeat");
                srcAxesBitMask |= 1U << srcAxis;
                dstAxesBitMask |= 1U << dstAxis;
                shapeOut[dstAxis] = shapeX[srcAxis];
            }

            // fill remaining dims in order
            for (int srcAxis = 0, dstAxis = 0; srcAxis < shapeX.rank; srcAxis++)
            {
                if (((srcAxesBitMask >> srcAxis) & 1U) != 0)
                    continue;
                while (((dstAxesBitMask >> dstAxis) & 1U) != 0)
                    dstAxis++;
                srcAxesBitMask |= 1U << srcAxis;
                dstAxesBitMask |= 1U << dstAxis;
                shapeOut[dstAxis] = shapeX[srcAxis];
                dstAxis++;
            }

            setPartialTensor(0, PartialTensor.Create(getPartialTensor(0).dataType, shapeOut));
        }

        internal override unsafe void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);

            Span<int> permutations = stackalloc int[X.shape.rank];
            ShapeInference.MoveDim(X.shape, source, destination, ref permutations);

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Transpose(permutations), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Transpose(X, O, permutations);
        }
    }

    /// <summary>
    /// Represents a `Narrow` layer. The layer calculates the output tensor by slicing the input tensor along a given dim with a given start and length.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "dim", "start", "length" }, inputCPURead = new[] { 1, 2, 3 })]
    partial class Narrow : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var dim = getPartialTensor(1) as PartialTensor<int>;
            var start = getPartialTensor(2) as PartialTensor<int>;
            var length = getPartialTensor(3) as PartialTensor<int>;
            if (!dim.IsStatic())
            {
                setPartialTensor(0, PartialTensor.Create(X.dataType, DynamicTensorShape.DynamicOfRank(X.shape.rank)));
                return;
            }

            var dimValue = dim[0].value;

            var outShape = X.shape;
            outShape[dimValue] = DynamicTensorDim.Unknown;

            if (start.IsStatic() && length.IsStatic() && X.shape[dimValue].isValue)
            {
                var dimSize = X.shape[dimValue].value;
                var startValue = (start[0].value + dimSize) % dimSize;
                var end = Mathf.Min(startValue + length[0].value, dimSize);
                var lengthValue = end - startValue;
                outShape[dimValue] = DynamicTensorDim.FromInt(lengthValue);

                if (X.isPartiallyKnown)
                {
                    var tensorOut = PartialTensor.Create(X.dataType, outShape);
                    for (var i = 0; i < lengthValue; i++)
                        tensorOut.CopyElement(i, X, startValue + i);
                    setPartialTensor(0, tensorOut);
                    return;
                }
            }

            setPartialTensor(0, PartialTensor.Create(X.dataType, outShape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var dim = ctx.storage.GetInt(inputs[1]);
            var start = ctx.storage.GetInt(inputs[2]);
            var length = ctx.storage.GetInt(inputs[3]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.MoveDim(X.shape, ref dim, ref start, ref length), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Split(X, O, dim, start);
        }
    }

    /// <summary>
    /// Represents a `Pad` layer. The layer calculates the output tensor by adding padding to the input tensor according to the given padding values and mode.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "data", "pads", "constantValue", "axes" }, inputCPURead = new[] { 1, 2, 3 })]
    partial class Pad : Layer
    {
        public PadMode padMode;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var pads = getPartialTensor(1) as PartialTensor<int>;
            var shapeX = X.shape;
            var shapePads = pads.shape;
            if (shapePads.hasRank)
            {
                Logger.AssertIsTrue(shapePads.rank == 1, "Pad.ValueError: pads must be rank 1");
                Logger.AssertIsTrue(!shapePads[0].isValue || shapePads[0].value % 2 == 0, "Pad.ValueError: length of pads must divide by 2");
            }

            var axes = getPartialTensor(3) as PartialTensor<int>;

            if (axes == null)
            {
                shapeX.DeclareRank(shapePads[0] / 2);
                axes = shapeX.hasRank ? PartialTensor.Range(0, shapeX.rank) : new PartialTensor<int>(DynamicTensorShape.DynamicOfRank(1));
            }

            if (!axes.isPartiallyKnown)
            {
                setPartialTensor(0, PartialTensor.Create(X.dataType, DynamicTensorShape.DynamicOfRankLike(shapeX)));
                return;
            }

            Logger.AssertIsTrue(!shapePads[0].isValue || shapePads[0].value == axes.length * 2, "Pad.ValueError: length of pads must be twice the length of the axes");

            var shapeOut = new DynamicTensorShape(shapeX);

            for (var i = 0; i < axes.length; i++)
            {
                if (!axes[i].isValue)
                    continue;
                var axis = axes[i].value;
                var dimPad = pads[i] + pads[i + axes.length];
                shapeOut[axis] = (DynamicTensorDim)((PartialTensorElement<int>)shapeX[axis] + dimPad);
            }

            setPartialTensor(0, PartialTensor.Create(X.dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var pad = ctx.storage.GetInts(inputs[1]);
            var axes = ctx.storage.GetInts(inputs[3], null);

            Span<int> pads = stackalloc int[2 * X.shape.rank];
            if (axes != null)
            {
                for (var i = 0; i < axes.Length; i++)
                {
                    var axis = X.shape.Axis(axes[i]);
                    pads[axis] = pad[i];
                    pads[axis + X.shape.rank] = pad[i + axes.Length];
                }
            }
            else
            {
                pad.CopyTo(pads);
            }

            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Pad(pads), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (padMode != PadMode.Constant)
            {
                Assert.IsFalse(X.shape.HasZeroDims(), "ValueError: zero dimensions input for Pad operator is not supported");
                if (X.dataType == DataType.Float)
                    ctx.backend.Pad(X as Tensor<float>, O as Tensor<float>, pads, padMode, 0);
                else
                    ctx.backend.Pad(X as Tensor<int>, O as Tensor<int>, pads, padMode, 0);
                return;
            }

            if (X.dataType == DataType.Float)
            {
                var constantValue = ctx.storage.GetFloat(inputs[2]);
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<float>, constantValue);
                else
                    ctx.backend.Pad(X as Tensor<float>, O as Tensor<float>, pads, padMode, constantValue);
            }
            else
            {
                var constantValue = ctx.storage.GetInt(inputs[2]);
                if (X.shape.HasZeroDims())
                    ctx.backend.MemSet(O as Tensor<int>, constantValue);
                else
                    ctx.backend.Pad(X as Tensor<int>, O as Tensor<int>, pads, padMode, constantValue);
            }
        }
    }

    /// <summary>
    /// Represents a `Reshape` layer. The layer calculates the output tensor by copying the data from the input tensor and using a given shape. The data from the input tensor is unchanged.
    ///
    /// Only one of the elements of the shape can be -1. The layer infers the size of this dimension from the remaining dimensions and the length of the input tensor.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "shape" }, inputCPURead = new[] { 1 })]
    partial class Reshape : Layer
    {
        public bool allowZero;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var shape = getPartialTensor(1) as PartialTensor<int>;
            var shapeX = X.shape;
            shape.shape.DeclareRank(1);

            if (!shape.isPartiallyKnown)
            {
                if (shape.shape[0].isValue)
                    setPartialTensor(0, X.Reshape(DynamicTensorShape.DynamicOfRank(shape.shape[0].value)));
                else
                    setPartialTensor(0, X.Reshape(DynamicTensorShape.DynamicRank));
                return;
            }

            var shapeOut = DynamicTensorShape.DynamicOfRank(shape.length);

            var containsMinusOne = false;

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (shape[i].Equals(-1))
                    containsMinusOne = true;
            }

            for (var i = 0; i < shapeOut.rank; i++)
            {
                if (shape[i].isUnknown)
                    continue;

                var dim = (DynamicTensorDim)shape[i];
                if (shape[i].isParam)
                {
                    if (allowZero || (shapeX.hasRank && i >= shapeX.rank) || shapeX[i] == dim)
                        shapeOut[i] = dim;
                    else if (containsMinusOne)
                    {
                        for (var j = 0; j < shapeX.rank; j++)
                        {
                            if (shapeX[j] == dim)
                            {
                                shapeOut[i] = dim;
                                break;
                            }
                        }
                    }
                    continue;
                }

                if (shape[i].value > 0)
                    shapeOut[i] = dim;
                else if (shape[i].value == 0)
                    shapeOut[i] = allowZero ? DynamicTensorDim.Zero : shapeX[i];
            }

            setPartialTensor(0, X.Reshape(shapeOut, !containsMinusOne));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var shape = X.shape.Reshape(ctx.storage.GetInts(inputs[1]), allowZero);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O);
        }
    }

    /// <summary>
    /// Represents a `Resize` layer. The layer calculates the output tensor by resampling the input tensor along the spatial dimensions to a given shape.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "scalesOrSizes" }, inputCPURead = new[] { 1 })]
    partial class Resize : Layer
    {
        public ScaleMode scaleMode;
        public CoordTransformMode coordTransformMode;
        public InterpolationMode mode;
        public NearestMode nearestMode;
        public int[] axes;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var dataType = getPartialTensor(0).dataType;
            var shapeX = getPartialTensor(0).shape;

            var sizesOrScales = getPartialTensor(1);
            sizesOrScales.shape.DeclareRank(1);
            if (axes == null)
                shapeX.DeclareRank(sizesOrScales.shape[0]);
            var shapeOut = new DynamicTensorShape(shapeX);
            if (shapeOut.hasRank)
            {
                if (axes == null)
                {
                    for (var i = 0; i < shapeOut.rank; i++)
                        shapeOut[i] = scaleMode == ScaleMode.Sizes ? (DynamicTensorDim)sizesOrScales.Get<int>(i) : shapeX[i].Resize(sizesOrScales.Get<float>(i));
                }
                else
                {
                    for (var i = 0; i < axes.Length; i++)
                    {
                        var axis = shapeOut.Axis(axes[i]);
                        shapeOut[axis] = scaleMode == ScaleMode.Sizes ? (DynamicTensorDim)sizesOrScales.Get<int>(i) : shapeX[axis].Resize(sizesOrScales.Get<float>(i));
                    }
                }
            }

            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            Span<float> s = stackalloc float[X.shape.rank];
            for (var i = 0; i < s.Length; i++)
                s[i] = 1f;

            if (scaleMode == ScaleMode.Sizes)
            {
                var sizes = ctx.storage.GetInts(inputs[1]);

                if (axes != null)
                {
                    for (var i = 0; i < axes.Length; i++)
                    {
                        var axis = X.shape.Axis(axes[i]);
                        s[axis] = sizes[i] / (float)X.shape[axis];
                    }
                }
                else
                {
                    for (var i = 0; i < X.shape.rank; i++)
                        s[i] = sizes[i] / (float)X.shape[i];
                }

                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Resize(X.shape, s), DataType.Float, ctx.backend.backendType) as Tensor<float>;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Resize(X, O, s, mode, nearestMode, coordTransformMode);
            }
            else
            {
                var scales = ctx.storage.GetFloats(inputs[1]);
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.Resize(X.shape, scales), DataType.Float, ctx.backend.backendType) as Tensor<float>;
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Resize(X, O, scales, mode, nearestMode, coordTransformMode);
            }
        }
    }

    /// <summary>
    /// Represents a `Select` layer. The layer calculates the output tensor by slicing the input tensor along a given dim with a given index, the sliced dim is removed from the output.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "dim", "selectIndex" }, inputCPURead = new[] { 1, 2 })]
    partial class Select : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var dim = getPartialTensor(1) as PartialTensor<int>;
            var selectIndex = getPartialTensor(2) as PartialTensor<int>;
            var outShape = X.shape;
            if (!dim.IsStatic())
            {
                outShape = X.shape.hasRank ? DynamicTensorShape.DynamicOfRank(X.shape.rank - 1) : DynamicTensorShape.DynamicRank;
                setPartialTensor(0, PartialTensor.Create(X.dataType, outShape));
                return;
            }

            var axis = dim[0].value;
            outShape[axis] = DynamicTensorDim.One;
            outShape = outShape.Squeeze(axis);
            var tensorOut = PartialTensor.Create(X.dataType, outShape);

            if (axis == 0 && X.isPartiallyKnown && selectIndex.IsStatic())
            {
                var index = selectIndex[0].value;
                index = index < 0 ? index + X.shape.Length().value : index;
                tensorOut.CopyElement(0, X, index);
            }

            setPartialTensor(0, tensorOut);
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var dim = ctx.storage.GetInt(inputs[1]);
            var selectIndex = ctx.storage.GetInt(inputs[2]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Reduce(dim, false), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var unsqueezed = ctx.storage.AllocateTensor(X.shape.Reduce(dim, true), X.dataType, ctx.backend.backendType);
            dim = X.shape.Axis(dim);
            selectIndex = selectIndex < 0 ? selectIndex + X.shape[dim] : selectIndex;
            ctx.backend.Split(X, unsqueezed, dim, selectIndex);
            ctx.backend.Reshape(unsqueezed, O);
            ctx.storage.Dispose(unsqueezed);
        }
    }

    /// <summary>
    /// Represents a `Slice` layer. The layer calculates the output tensor by slicing the input tensor along given axes with given starts, ends and steps.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "starts", "ends", "axes", "steps" }, inputCPURead = new[] { 1, 2, 3, 4 })]
    partial class Slice : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var data = getPartialTensor(0);
            if (!data.shape.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(data.dataType));
                return;
            }

            var starts = getPartialTensor(1) as PartialTensor<int>;
            var ends = getPartialTensor(2) as PartialTensor<int>;
            var axes = getPartialTensor(3) as PartialTensor<int> ?? PartialTensor.Range(0, data.shape.rank);
            var steps = getPartialTensor(4) as PartialTensor<int> ?? PartialTensor.Ones(starts.shape);

            if (data.isPartiallyKnown && data.shape.rank == 1 && starts[0].isValue && ends[0].isValue && steps[0].isValue)
            {
                var dim = data.shape[0].value;
                var start = starts[0].value;
                var end = ends[0].value;
                var step = steps[0].value;

                var clampAdjustDirection = step < 0 ? -1 : 0;

                start = start < 0 ? dim + start : start;
                start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

                end = end < 0 ? dim + end : end;
                end = Mathf.Clamp(end, clampAdjustDirection, dim);

                var length = (int)Math.Ceiling((end - start) / (double)step);
                length = Mathf.Max(length, 0);

                var tensorOut = PartialTensor.Create(data.dataType, new DynamicTensorShape(length));

                for (var i = 0; i < length; i++)
                {
                    tensorOut.CopyElement(i, data, start + i * step);
                }

                setPartialTensor(0, tensorOut);
                return;
            }

            if (!axes.isPartiallyKnown)
            {
                setPartialTensor(0, PartialTensor.Create(data.dataType, DynamicTensorShape.DynamicOfRank(data.shape.rank)));
                return;
            }

            var shapeOut = new DynamicTensorShape(data.shape);

            for (var i = 0; i < axes.length; i++)
            {
                var axisElement = axes[i];
                if (!axisElement.isValue)
                {
                    shapeOut = DynamicTensorShape.DynamicOfRank(data.shape.rank);
                    continue;
                }
                var axis = shapeOut.Axis(axisElement.value);
                shapeOut[axis] = data.shape[axis].Slice(starts[i], ends[i], steps[i]);
            }

            setPartialTensor(0, PartialTensor.Create(data.dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var starts = ctx.storage.GetInts(inputs[1]);
            var ends = ctx.storage.GetInts(inputs[2]);
            var axes = ctx.storage.GetInts(inputs[3], null);
            var steps = ctx.storage.GetInts(inputs[4], null);
            var numAxes = starts.Length;
            Span<int> startsSpan = stackalloc int[numAxes];
            Span<int> endsSpan = stackalloc int[numAxes];
            Span<int> axesSpan = stackalloc int[numAxes];
            Span<int> stepsSpan = stackalloc int[numAxes];
            ShapeInference.Slice(X.shape, starts, ends, axes, steps, ref startsSpan, ref endsSpan, ref axesSpan, ref stepsSpan);
            var shapeOut = X.shape.Slice(startsSpan, endsSpan, axesSpan, stepsSpan);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shapeOut, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Slice(X, O, startsSpan, axesSpan, stepsSpan);
        }
    }

    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "values", "starts", "ends", "axes", "steps" }, inputCPURead = new[] { 2, 3, 4, 5 })]
    partial class SliceSet : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            setPartialTensor(0, PartialTensor.Create(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var values = ctx.storage.GetTensor(inputs[1]);
            var starts = ctx.storage.GetInts(inputs[2]);
            var ends = ctx.storage.GetInts(inputs[3]);
            var axes = ctx.storage.GetInts(inputs[4], null);
            var steps = ctx.storage.GetInts(inputs[5], null);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            var numAxes = starts.Length;
            Span<int> startsSpan = stackalloc int[numAxes];
            Span<int> endsSpan = stackalloc int[numAxes];
            Span<int> axesSpan = stackalloc int[numAxes];
            Span<int> stepsSpan = stackalloc int[numAxes];
            for (var i = 0; i < numAxes; i++)
            {
                var axis = axes == null ? i : X.shape.Axis(axes[i]);
                var start = starts[i];
                var end = ends[i];
                var step = steps == null ? 1 : steps[i];

                stepsSpan[i] = step;
                axesSpan[i] = axis;

                var dim = X.shape[axis];
                var clampAdjustDirection = step < 0 ? -1 : 0;

                start = start < 0 ? dim + start : start;
                start = Mathf.Clamp(start, 0, dim + clampAdjustDirection);

                end = end < 0 ? dim + end : end;
                end = Mathf.Clamp(end, clampAdjustDirection, dim);

                startsSpan[i] = dim == 0 ? 0 : start;
                endsSpan[i] = dim == 0 ? 0 : end;
            }
            var slicedShape = X.shape.Slice(startsSpan, endsSpan, axesSpan, stepsSpan);
            Logger.AssertIsTrue(slicedShape.Broadcast(values.shape) == slicedShape, "SliceSet.InputError: values shape must be broadcastable to sliced shape, {0} {1}", values.shape, slicedShape);
            if (slicedShape != values.shape)
            {
                // broadcast values
                var broadcastValues = ctx.storage.AllocateTensor(slicedShape, values.dataType, ctx.backend.backendType);
                ctx.backend.Expand(values, broadcastValues);
                ctx.backend.SliceSet(X, broadcastValues, O, startsSpan, axesSpan, stepsSpan);
                ctx.storage.Dispose(broadcastValues);
            }
            else
            {
                ctx.backend.SliceSet(X, values, O, startsSpan, axesSpan, stepsSpan);
            }
        }
    }

    /// <summary>
    /// Represents a `SpaceToDepth` layer. The layer computes the output tensor by permuting data from blocks of spatial data into depth.
    /// </summary>
    [Operator(category = "Transformation")]
    partial class SpaceToDepth : Layer
    {
        public int blocksize;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var shapeX = getPartialTensor(0).shape;
            shapeX.DeclareRank(4);
            setPartialTensor(0, PartialTensor.Create(getPartialTensor(0).dataType, new DynamicTensorShape(shapeX[0], shapeX[1] * (blocksize * blocksize), shapeX[2] / blocksize, shapeX[3] / blocksize)));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], ShapeInference.SpaceToDepth(X.shape, blocksize), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.SpaceToDepth(X, O, blocksize);
        }
    }

    /// <summary>
    /// Represents a `Split` layer. The layer computes the output tensors by splitting the input tensor along a single given axis.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "split" }, inputCPURead = new[] { 1 })]
    [Outputs(isVariadic = true)]
    partial class Split : Layer
    {
        public int axis;
        public int numOutputs;

        internal override int OutputCount => numOutputs;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var partialSplit = getPartialTensor(1) as PartialTensor<int>;
            if (partialSplit == null)
            {
                partialSplit = new PartialTensor<int>(new DynamicTensorShape(numOutputs));

                var dim = X.shape[axis];
                if (dim.isParam && numOutputs == 1)
                {
                    partialSplit[0] = PartialTensorElement<int>.Param(dim.param);
                }
                else if (dim.isValue)
                {
                    Logger.AssertIsTrue(numOutputs >= 1, "Split.InputError: numOutputs must be positive if split tensor is null");
                    var splitLength = Mathf.CeilToInt(dim.value / (float)numOutputs);
                    for (var i = 0; i < numOutputs - 1; i++)
                    {
                        partialSplit[i] = PartialTensorElement<int>.Value(splitLength);
                    }

                    // final split length is the (possible smaller) remainder along the axis
                    var lastSplitLength = dim.value - (splitLength * (numOutputs - 1));
                    Logger.AssertIsTrue(lastSplitLength >= 0, "Split.InputError: split axis too small for numOutputs");
                    partialSplit[numOutputs - 1] = PartialTensorElement<int>.Value(lastSplitLength);
                }
            }

            for (var i = 0; i < outputs.Length; i++)
            {
                var outputShape = new DynamicTensorShape(X.shape);
                outputShape[axis] = (DynamicTensorDim)partialSplit[i];
                setPartialTensor(i, PartialTensor.Create(X.dataType, outputShape));
            }
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);

            var dim = X.shape[axis];
            var equalSplitLength = 0;
            var split = ctx.storage.GetInts(inputs[1], null);
            if (split == null)
            {
                // if splits are not given calculate even split length
                equalSplitLength = (int)Math.Ceiling(dim / (double)numOutputs);
            }
            var start = 0;
            for (var i = 0; i < outputs.Length; i++)
            {
                var end = start + (split != null ? split[i] : equalSplitLength);
                end = Math.Min(end, dim);
                var O = ctx.storage.AllocateTensorAndStore(outputs[i], X.shape.Split(axis, start, end), X.dataType, ctx.backend.backendType);
                if (!O.shape.HasZeroDims())
                    ctx.backend.Split(X, O, X.shape.Axis(axis), start);
                start = end;
            }
        }
    }

    /// <summary>
    /// Represents a `Squeeze` layer. The layer computes the output tensor by reshaping the input tensor by removing dimensions of size 1.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "axes" }, inputCPURead = new[] { 1 })]
    partial class Squeeze : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var axes = getPartialTensor(1) as PartialTensor<int>;
            if (axes != null)
            {
                if (!axes.isPartiallyKnown)
                    setPartialTensor(0, X.Reshape(DynamicTensorShape.DynamicRank));
                else if (!axes.IsStatic())
                    setPartialTensor(0, X.Reshape(DynamicTensorShape.DynamicOfRank(X.shape.rank - axes.length)));
                else
                    setPartialTensor(0, X.Reshape(X.shape.Squeeze(axes)));
                return;
            }

            setPartialTensor(0, X.Reshape(X.shape.Squeeze()));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1], null);
            var shape = axes != null ? X.shape.Squeeze(axes) : X.shape.Squeeze();
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O); // TODO<tensordata> refcount tensordata
        }
    }

    /// <summary>
    /// Represents a `Tile` layer. The layer computes the output tensor by repeating the input layer a given number of times along each axis.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "repeats" }, inputCPURead = new[] { 1 })]
    partial class Tile : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var dataType = getPartialTensor(0).dataType;
            var shapeX = getPartialTensor(0).shape;
            var repeats = getPartialTensor(1) as PartialTensor<int>;
            repeats.shape.DeclareRank(1);

            if (!repeats.isPartiallyKnown)
            {
                if (repeats.shape[0].isValue && !shapeX.hasRank)
                    shapeX = DynamicTensorShape.DynamicOfRank(repeats.shape[0].value);
                if (shapeX.hasRank)
                    Logger.AssertIsFalse(repeats.shape[0] != shapeX.rank, "Tile.InputError: repeats value must be equal to input rank");
                setPartialTensor(0, PartialTensor.Create(dataType, DynamicTensorShape.DynamicOfRankLike(shapeX)));
                return;
            }

            shapeX.DeclareRank(repeats.length);

            var shapeOut = new DynamicTensorShape(shapeX);
            for (var i = 0; i < shapeOut.rank; i++)
            {
                shapeOut[i] *= (DynamicTensorDim)repeats[i];
            }
            setPartialTensor(0, PartialTensor.Create(dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var repeats = ctx.storage.GetInts(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Tile(repeats), X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Tile(X, O, repeats);
        }
    }

    /// <summary>
    /// Represents a `Transpose` layer. The layer computes the output tensor by permuting the axes and data of the input tensor according to the given permutations.
    /// </summary>
    [Operator(category = "Transformation")]
    partial class Transpose : Layer
    {
        public int[] permutations;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var shapeX = getPartialTensor(0).shape;
            if (permutations != null)
                shapeX.DeclareRank(permutations.Length);

            if (!shapeX.hasRank)
            {
                setPartialTensor(0, PartialTensor.Create(getPartialTensor(0).dataType));
                return;
            }

            var shapeOut = DynamicTensorShape.DynamicOfRank(shapeX.rank);

            if (permutations == null || permutations.Length == 0)
            {
                // reverse axes
                for (var i = 0; i < shapeX.rank; i++)
                {
                    shapeOut[i] = shapeX[shapeX.rank - 1 - i];
                }
            }
            else
            {
                uint axesBitMask = 0;
                for (var i = 0; i < permutations.Length; i++)
                {
                    var axis = shapeX.Axis(permutations[i]);
                    Logger.AssertIsTrue(((axesBitMask >> axis) & 1U) == 0, "Transpose.ValueError: permutation must be a permutation of the axis (0, rank-1)");
                    axesBitMask |= 1U << axis;
                    shapeOut[i] = shapeX[axis];
                }
            }

            setPartialTensor(0, PartialTensor.Create(getPartialTensor(0).dataType, shapeOut));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            if (permutations == null)
            {
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Transpose(), X.dataType, ctx.backend.backendType);
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Transpose(X, O);
                return;
            }
            else
            {
                var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape.Transpose(permutations), X.dataType, ctx.backend.backendType);
                if (O.shape.HasZeroDims())
                    return;
                ctx.backend.Transpose(X, O, permutations);
                return;
            }
        }
    }

    /// <summary>
    /// Represents a `Trilu` layer. The layer computes the output tensor by retaining the upper or lower triangular values from an input matrix or matrix batch and setting the other values to zero.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "k" }, inputCPURead = new[] { 1 })]
    partial class Trilu : Layer
    {
        public TriluMode mode;

        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            setPartialTensor(0, PartialTensor.Create(X.dataType, X.shape));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var k = ctx.storage.GetInt(inputs[1]);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            if (mode == TriluMode.Upper)
                ctx.backend.Triu(X, O, k);
            else
                ctx.backend.Tril(X, O, k);
        }
    }

    /// <summary>
    /// Represents an `Unsqueeze` layer. The layer computes the output tensor by reshaping the input tensor by adding dimensions of size 1 at the given axes.
    /// </summary>
    [Operator(category = "Transformation")]
    [Inputs(names = new[] { "input", "axes" }, inputCPURead = new[] { 1 })]
    partial class Unsqueeze : Layer
    {
        internal override void InferPartial(Func<int, PartialTensor> getPartialTensor, Action<int, PartialTensor> setPartialTensor)
        {
            var X = getPartialTensor(0);
            var shape = getPartialTensor(1) as PartialTensor<int>;
            setPartialTensor(0, X.Reshape(X.shape.Unsqueeze(shape)));
        }

        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]);
            var axes = ctx.storage.GetInts(inputs[1]);
            var shape = X.shape.Unsqueeze(axes);
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], shape, X.dataType, ctx.backend.backendType);
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Reshape(X, O); // TODO<tensordata> refcount tensordata
        }
    }
}
