using System;
using UnityEngine;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the input tensors concatenated along a dimension.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <param name="dim">The dimension along which to concatenate.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Concat(FunctionalTensor[] tensors, int dim = 0)
        {
            return FromLayer(new Layers.Concat(-1, new int[tensors.Length], dim), tensors);
        }

        /// <summary>
        /// Returns the input tensor gathered along a dimension with indices.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to gather.</param>
        /// <param name="index">The indices tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Gather(this FunctionalTensor input, int dim, FunctionalTensor index)
        {
            DeclareType(DataType.Int, index);
            return FromLayer(new Layers.GatherElements(-1, -1, -1, dim), new[] { input, index });
        }

        /// <summary>
        /// Returns the input tensor indexed along a dimension with entries in a 1D index tensor.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to select.</param>
        /// <param name="index">The indices tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor IndexSelect(this FunctionalTensor input, int dim, FunctionalTensor index)
        {
            DeclareType(DataType.Int, index);
            return FromLayer(new Layers.Gather(-1, -1, -1, dim), new[] { input, index });
        }

        /// <summary>
        /// Returns the input tensor with a dimension moved from source to destination.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="source">The dimension in the input tensor to move.</param>
        /// <param name="destination">The moved dimension in the output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MoveDim(this FunctionalTensor input, int source, int destination)
        {
            return MoveDim(input, new[] { source }, new[] { destination });
        }

        /// <summary>
        /// Returns the input tensor with multiple dimensions moved from source to destination.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="source">The dimensions in the input tensor to move.</param>
        /// <param name="destination">The moved dimensions in the output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MoveDim(this FunctionalTensor input, int[] source, int[] destination)
        {
            return FromLayer(new Layers.MoveDim(-1, -1, source, destination), input);
        }

        /// <summary>
        /// Returns the input tensor narrowed along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to narrow.</param>
        /// <param name="start">The start index along the dimension.</param>
        /// <param name="length">The number of elements along the dimension.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Narrow(this FunctionalTensor input, int dim, int start, int length)
        {
            return FromLayer(new Layers.Narrow(-1, -1, -1, -1, -1), new[] { input, Constant(dim), Constant(start), Constant(length) });
        }

        /// <summary>
        /// Returns the input tensor narrowed along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to narrow.</param>
        /// <param name="start">The functional start index along the dimension.</param>
        /// <param name="length">The functional number of elements along the dimension.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Narrow(this FunctionalTensor input, int dim, FunctionalTensor start, FunctionalTensor length)
        {
            DeclareType(DataType.Int, start, length);
            return FromLayer(new Layers.Narrow(-1, -1, -1, -1, -1), new[] { input, Constant(dim), start, length });
        }

        /// <summary>
        /// Returns the indices of the input tensor with values not equal to zero.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor NonZero(FunctionalTensor input)
        {
            // TODO support asTuple
            return Transpose(FromLayer(new Layers.NonZero(-1, -1), input), 0, 1);
        }

        /// <summary>
        /// Returns the input tensor padded with size determined by the pad array and values determined by the mode.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="pad">The padding lower and upper sizes starting from the final dimension (pad_w_lower, pad_w_upper, pad_h_lower, pad_h_upper, ...), not all dimensions need to be padded.</param>
        /// <param name="mode">The mode to use for sampling values, should be `constant`, `reflect`, `replicate` or `circular`, for constant padding with non zero values use one of the other `Pad` methods.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pad(this FunctionalTensor input, int[] pad, string mode)
        {
            var padMode = mode switch
            {
                "constant" => Layers.PadMode.Constant,
                "reflect" => Layers.PadMode.Reflect,
                "replicate" => Layers.PadMode.Edge,
                "circular" => Layers.PadMode.Wrap,
                _ => Layers.PadMode.Constant
            };
            var axes = new int[pad.Length / 2];
            var pads = new int[pad.Length];
            for (var i = 0; i < axes.Length; i++)
            {
                axes[i] = -i - 1;
                pads[i] = pad[2 * i];
                pads[i + axes.Length] = pad[2 * i + 1];
            }
            return FromLayer(new Layers.Pad(-1, -1, -1, -1, -1, padMode), new[] { input, Constant(pads), null, Constant(axes) });
        }

        /// <summary>
        /// Returns the input tensor padded with size determined by the pad array and a constant value.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="pad">The padding lower and upper sizes starting from the final dimension (pad_w_lower, pad_w_upper, pad_h_lower, pad_h_upper, ...), not all dimensions need to be padded.</param>
        /// <param name="value">The constant value to use for padding.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pad(this FunctionalTensor input, int[] pad, int value)
        {
            if (input.dataType == DataType.Float)
                return Pad(input, pad, (float)value);
            var axes = new int[pad.Length / 2];
            var pads = new int[pad.Length];
            for (var i = 0; i < axes.Length; i++)
            {
                axes[i] = -i - 1;
                pads[i] = pad[2 * i];
                pads[i + axes.Length] = pad[2 * i + 1];
            }
            return FromLayer(new Layers.Pad(-1, -1, -1, -1, -1, Layers.PadMode.Constant), new[] { input, Constant(pads), Constant(value), Constant(axes) });
        }

        /// <summary>
        /// Returns the input tensor padded with size determined by the pad array and a constant value.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="pad">The padding lower and upper sizes starting from the final dimension (pad_w_lower, pad_w_upper, pad_h_lower, pad_h_upper, ...), not all dimensions need to be padded.</param>
        /// <param name="value">The constant value to use for padding.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Pad(this FunctionalTensor input, int[] pad, float value)
        {
            DeclareType(DataType.Float, input);
            var axes = new int[pad.Length / 2];
            var pads = new int[pad.Length];
            for (var i = 0; i < axes.Length; i++)
            {
                axes[i] = -i - 1;
                pads[i] = pad[2 * i];
                pads[i + axes.Length] = pad[2 * i + 1];
            }
            return FromLayer(new Layers.Pad(-1, -1, -1, -1, -1, Layers.PadMode.Constant), new[] { input, Constant(pads), Constant(value), Constant(axes) });
        }

        /// <summary>
        /// Returns the input tensor with permuted dimensions.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dims">The dimensions of the input tensor to use in the permuted output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Permute(this FunctionalTensor input, int[] dims)
        {
            return FromLayer(new Layers.Transpose(-1, -1, dims), input);
        }

        /// <summary>
        /// Returns the input tensor elements reshaped.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="shape">The shape of the output tensor. A negative value is inferred from the others.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Reshape(this FunctionalTensor input, int[] shape)
        {
            return FromLayer(new Layers.Reshape(-1, -1, -1), new[] { input, Constant(shape) });
        }

        /// <summary>
        /// Returns the input tensor sliced along a dimension at an index.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to select.</param>
        /// <param name="index">The index along the dimension to select.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Select(this FunctionalTensor input, int dim, int index)
        {
            return FromLayer(new Layers.Select(-1, -1, -1, -1), new[] { input, Constant(dim), Constant(index) });
        }

        /// <summary>
        /// Returns the input tensor sliced along a dimension at an index.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to select.</param>
        /// <param name="index">The functional index along the dimension to select.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Select(this FunctionalTensor input, int dim, FunctionalTensor index)
        {
            DeclareType(DataType.Int, index);
            return FromLayer(new Layers.Select(-1, -1, -1, -1), new[] { input, Constant(dim), index });
        }

        /// <summary>
        /// Returns a copy of the input with the elements replaced by those from source given by the index along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="index">The index tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Scatter(FunctionalTensor input, int dim, FunctionalTensor index, FunctionalTensor src)
        {
            // TODO add reduction
            DeclareType(DataType.Int, index);
            return FromLayer(new Layers.ScatterElements(-1, -1, -1, -1, dim, Layers.ScatterReductionMode.None), new[] { input, index, src });
        }

        /// <summary>
        /// Returns a copy of the input with the elements replaced by those from source at a dimension and index.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="index">The index at which to scatter along the dimension.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor SelectScatter(FunctionalTensor input, FunctionalTensor src, int dim, int index)
        {
            return FromLayer(new Layers.SliceSet(-1, -1, -1, -1, -1, -1, -1), new[] { input, Unsqueeze(src, dim), Constant(new[] { index }), Constant(new[] { index + 1 }), Constant(new[] { dim }), null });
        }

        /// <summary>
        /// Returns a copy of the input with the elements replaced by those from source along a dimension with start, end and step.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="start">The index of the first element to replace along the dimension.</param>
        /// <param name="end">The end index of the scatter along the dimension.</param>
        /// <param name="step">The step between the indices along the dimension.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor SliceScatter(FunctionalTensor input, FunctionalTensor src, int dim = 0, int start = 0, int end = int.MaxValue, int step = 1)
        {
            return FromLayer(new Layers.SliceSet(-1, -1, -1, -1, -1, -1, -1), new[] { input, src, Constant(new[] { start }), Constant(new[] { end }), Constant(new[] { dim }), Constant(new[] { step }) });
        }

        /// <summary>
        /// Returns a copy of the input with the elements updated by adding by those from source given by the index along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension along which to scatter.</param>
        /// <param name="index">The index tensor.</param>
        /// <param name="src">The source tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ScatterAdd(FunctionalTensor input, int dim, FunctionalTensor index, FunctionalTensor src)
        {
            return FromLayer(new Layers.ScatterElements(-1, -1, -1, -1, dim, Layers.ScatterReductionMode.Add), new[] { input, index, src });
        }

        /// <summary>
        /// Returns an array of tensors by splitting the input into sections along a dimension.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="sections">The length of each section along the dimension.</param>
        /// <param name="dim">The dimension along which to split.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor[] Split(this FunctionalTensor input, int[] sections, int dim = 0)
        {
            var dataTypes = new DataType[sections.Length];
            for (var i = 0; i < dataTypes.Length; i++)
                dataTypes[i] = input.dataType;
            return FromLayerMultiOutput(new Layers.Split(new int[sections.Length], -1, -1, dim, sections.Length), new[] { input, Constant(sections) });
        }

        /// <summary>
        /// Returns the input tensor with all dimensions of size 1 removed.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Squeeze(this FunctionalTensor input)
        {
            return FromLayer(new Layers.Squeeze(-1, -1, -1), new[] { input, null });
        }

        /// <summary>
        /// Returns the input tensor with all specified dimensions of size 1 removed.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimensions of size 1 to remove.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Squeeze(this FunctionalTensor input, int[] dim)
        {
            return FromLayer(new Layers.Squeeze(-1, -1, -1), new[] { input, Constant(dim) });
        }

        /// <summary>
        /// Returns the input tensors concatenated along a new dimension.
        /// </summary>
        /// <param name="tensors">The input tensors.</param>
        /// <param name="dim">The dimension along which to stack.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Stack(FunctionalTensor[] tensors, int dim = 0)
        {
            // TODO add properly
            var unsqueezedTensors = new FunctionalTensor[tensors.Length];
            for (var i = 0; i < unsqueezedTensors.Length; i++)
                unsqueezedTensors[i] = Unsqueeze(tensors[i], dim);
            return Concat(unsqueezedTensors, dim);
        }

        /// <summary>
        /// Returns a tensor with the elements of input at indices.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="index">The index tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Take(this FunctionalTensor input, FunctionalTensor index)
        {
            return Gather(Ravel(input), 0, index);
        }

        /// <summary>
        /// Returns the input tensor repeated on the dims.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dims">The number of times to repeat the input tensor along each dim.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Tile(this FunctionalTensor input, int[] dims)
        {
            // TODO deal with cases where dims.length != input.shape.rank
            return FromLayer(new Layers.Tile(-1, -1, -1), new[] { input, Constant(dims) });
        }

        /// <summary>
        /// Returns the input tensor with two dimensions swapped.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim0">The first dimension to swap.</param>
        /// <param name="dim1">The second dimension to swap.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Transpose(this FunctionalTensor input, int dim0, int dim1)
        {
            return MoveDim(input, new[] { dim0, dim1 }, new[] { dim1, dim0 });
        }

        /// <summary>
        /// Returns the input tensor with a new dimension of size 1 inserted.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="dim">The dimension at which to insert a size 1 dimension in the output tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Unsqueeze(this FunctionalTensor input, int dim)
        {
            return FromLayer(new Layers.Unsqueeze(-1, -1, -1), new[] { input, Constant(new[] { dim }) });
        }

        /// <summary>
        /// Returns condition ? input : other element-wise.
        /// </summary>
        /// <param name="condition">The condition tensor.</param>
        /// <param name="input">The first input tensor.</param>
        /// <param name="other">The second input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Where(FunctionalTensor condition, FunctionalTensor input, FunctionalTensor other)
        {
            DeclareType(DataType.Int, condition);
            return FromLayer(new Layers.Where(-1, -1, -1, -1), new[] { condition, input, other });
        }
    }
}
