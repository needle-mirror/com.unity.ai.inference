using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the result of a 1D convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilation">The dilation value of each spatial dimension of the filter.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Conv1D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
        {
            // TODO add auto padding
            DeclareRank(input, 3);
            DeclareRank(weight, 3);
            if (bias != null)
                DeclareRank(bias, 1);
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            var strides = new[] { stride };
            var pads = new[] { padding, padding };
            var dilations = new[] { dilation };
            return FromLayer(new Layers.Conv(Layers.AutoPad.NotSet, dilations, groups, pads, strides, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 2D convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilation">The dilation value of each spatial dimension of the filter.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Conv2D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
        {
            // TODO add auto padding
            DeclareRank(input, 4);
            DeclareRank(weight, 4);
            if (bias != null)
                DeclareRank(bias, 1);
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            var strides = new[] { stride, stride };
            var pads = new[] { padding, padding, padding, padding };
            var dilations = new[] { dilation, dilation };
            return FromLayer(new Layers.Conv(Layers.AutoPad.NotSet, dilations, groups, pads, strides, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 2D convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilation">The dilation value of each spatial dimension of the filter.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Conv2D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, (int, int) stride, (int, int) padding, (int, int) dilation, int groups = 1)
        {
            // TODO add auto padding
            DeclareRank(input, 4);
            DeclareRank(weight, 4);
            if (bias != null)
                DeclareRank(bias, 1);
            var strideArray = new[] { stride.Item1, stride.Item2 };
            var paddingArray = new[] { padding.Item1, padding.Item2, padding.Item1, padding.Item2 };
            var dilationArray = new[] { dilation.Item1, dilation.Item2 };
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            return FromLayer(new Layers.Conv(Layers.AutoPad.NotSet, dilationArray, groups, paddingArray, strideArray, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 3D convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilation">The dilation value of each spatial dimension of the filter.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Conv3D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, int stride = 1, int padding = 0, int dilation = 1, int groups = 1)
        {
            // TODO add auto padding
            DeclareRank(input, 5);
            DeclareRank(weight, 5);
            if (bias != null)
                DeclareRank(bias, 1);
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            var strides = new[] { stride, stride, stride };
            var pads = new[] { padding, padding, padding, padding, padding, padding };
            var dilations = new[] { dilation, dilation, dilation };
            return FromLayer(new Layers.Conv(Layers.AutoPad.NotSet, dilations, groups, pads, strides, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 3D convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="dilation">The dilation value of each spatial dimension of the filter.</param>
        /// <param name="groups">The number of groups that input channels and output channels are divided into.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor Conv3D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, (int, int, int) stride, (int, int, int) padding, (int, int, int) dilation, int groups = 1)
        {
            // TODO add auto padding
            DeclareRank(input, 5);
            DeclareRank(weight, 5);
            if (bias != null)
                DeclareRank(bias, 1);
            var strideArray = new[] { stride.Item1, stride.Item2, stride.Item3 };
            var paddingArray = new[] { padding.Item1, padding.Item2, padding.Item3, padding.Item1, padding.Item2, padding.Item3 };
            var dilationArray = new[] { dilation.Item1, dilation.Item2, dilation.Item3 };
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            return FromLayer(new Layers.Conv(Layers.AutoPad.NotSet, dilationArray, groups, paddingArray, strideArray, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 1D transposed convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ConvTranspose1D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, int stride = 1, int padding = 0, int outputPadding = 0)
        {
            // TODO add auto padding
            // TODO support groups, dilation
            DeclareRank(input, 3);
            DeclareRank(weight, 3);
            if (bias != null)
                DeclareRank(bias, 1);
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            var dilations = new[] { 1 };
            var strides = new[] { stride };
            var pads = new[] { padding, padding };
            var outputPaddings = new[] { outputPadding };
            return FromLayer(new Layers.ConvTranspose(Layers.AutoPad.NotSet, dilations, 1, outputPaddings, pads, strides, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 2D transposed convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ConvTranspose2D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, int stride = 1, int padding = 0, int outputPadding = 0)
        {
            // TODO add auto padding
            // TODO support groups, dilation
            DeclareRank(input, 4);
            DeclareRank(weight, 4);
            if (bias != null)
                DeclareRank(bias, 1);
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            var dilations = new[] { 1, 1 };
            var strides = new[] { stride, stride };
            var pads = new[] { padding, padding, padding, padding };
            var outputPaddings = new[] { outputPadding, outputPadding };
            return FromLayer(new Layers.ConvTranspose(Layers.AutoPad.NotSet, dilations, 1, outputPaddings, pads, strides, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 2D transposed convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ConvTranspose2D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, (int, int) stride, (int, int) padding, (int, int) outputPadding)
        {
            // TODO add auto padding
            // TODO support groups, dilation
            DeclareRank(input, 4);
            DeclareRank(weight, 4);
            if (bias != null)
                DeclareRank(bias, 1);
            var dilations = new[] { 1, 1 };
            var strideArray = new[] { stride.Item1, stride.Item2 };
            var paddingArray = new[] { padding.Item1, padding.Item2, padding.Item1, padding.Item2 };
            var outputPaddingArray = new[] { outputPadding.Item1, outputPadding.Item2 };
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            return FromLayer(new Layers.ConvTranspose(Layers.AutoPad.NotSet, dilations, 1, outputPaddingArray, paddingArray, strideArray, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 3D transposed convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ConvTranspose3D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, int stride = 1, int padding = 0, int outputPadding = 0)
        {
            // TODO add auto padding
            // TODO support groups, dilation
            DeclareRank(input, 5);
            DeclareRank(weight, 5);
            if (bias != null)
                DeclareRank(bias, 1);
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            var dilations = new[] { 1, 1, 1 };
            var strides = new[] { stride, stride, stride };
            var pads = new[] { padding, padding, padding, padding, padding, padding };
            var outputPaddings = new[] { outputPadding, outputPadding, outputPadding };
            return FromLayer(new Layers.ConvTranspose(Layers.AutoPad.NotSet, dilations, 1, outputPaddings, pads, strides, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }

        /// <summary>
        /// Returns the result of a 3D transposed convolution of the input with the weight and bias tensors.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="weight">The weight tensor.</param>
        /// <param name="bias">The optional bias tensor.</param>
        /// <param name="stride">The stride value for each spatial dimension of the filter.</param>
        /// <param name="padding">The lower and upper padding values for each spatial dimension of the filter.</param>
        /// <param name="outputPadding">The output padding value for each spatial dimension in the filter.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor ConvTranspose3D(FunctionalTensor input, FunctionalTensor weight, FunctionalTensor bias, (int, int, int) stride, (int, int, int) padding, (int, int, int) outputPadding)
        {
            // TODO add auto padding
            // TODO support groups, dilation
            DeclareRank(input, 5);
            DeclareRank(weight, 5);
            if (bias != null)
                DeclareRank(bias, 1);
            var dilations = new[] { 1, 1, 1 };
            var strideArray = new[] { stride.Item1, stride.Item2, stride.Item3 };
            var paddingArray = new[] { padding.Item1, padding.Item2, padding.Item3, padding.Item1, padding.Item2, padding.Item3 };
            var outputPaddingArray = new[] { outputPadding.Item1, outputPadding.Item2, outputPadding.Item3 };
            input = input.Float();
            weight = weight.Float();
            bias = bias?.Float();
            return FromLayer(new Layers.ConvTranspose(Layers.AutoPad.NotSet, dilations, 1, outputPaddingArray, paddingArray, strideArray, null, Layers.FusableActivation.None), new[] { input, weight, bias });
        }
    }
}
