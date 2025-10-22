using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns the result of a 1D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool1D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 3);
            input = input.Float();
            var s = stride ?? kernelSize;
            var kernelArray = new[] { kernelSize };
            var strideArray = new[] { s };
            var paddingArray = new[] { padding, padding };
            return FunctionalLayer.AveragePool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 2D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool2D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 4);
            input = input.Float();
            var s = stride ?? kernelSize;
            var kernelArray = new[] { kernelSize, kernelSize };
            var strideArray = new[] { s, s };
            var paddingArray = new[] { padding, padding, padding, padding };
            return FunctionalLayer.AveragePool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 2D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool2D(FunctionalTensor input, (int, int) kernelSize, (int, int)? stride = null, (int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 4);
            input = input.Float();
            var kernelArray = new[] { kernelSize.Item1, kernelSize.Item2 };
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0 };
            return FunctionalLayer.AveragePool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 3D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool3D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 5);
            input = input.Float();
            var s = stride ?? kernelSize;
            var kernelArray = new[] { kernelSize, kernelSize, kernelSize };
            var strideArray = new[] { s, s, s };
            var paddingArray = new[] { padding, padding, padding, padding, padding, padding };
            return FunctionalLayer.AveragePool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 3D average pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor AvgPool3D(FunctionalTensor input, (int, int, int) kernelSize, (int, int, int)? stride = null, (int, int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 5);
            input = input.Float();
            var kernelArray = new[] { kernelSize.Item1, kernelSize.Item2, kernelSize.Item3 };
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2, stride?.Item3 ?? kernelSize.Item3 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0 };
            return FunctionalLayer.AveragePool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 1D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool1D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 3);
            input = input.Float();
            var s = stride ?? kernelSize;
            var kernelArray = new[] { kernelSize };
            var strideArray = new[] { s };
            var paddingArray = new[] { padding, padding };
            return FunctionalLayer.MaxPool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 2D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool2D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 4);
            input = input.Float();
            var s = stride ?? kernelSize;
            var kernelArray = new[] { kernelSize, kernelSize };
            var strideArray = new[] { s, s };
            var paddingArray = new[] { padding, padding, padding, padding };
            return FunctionalLayer.MaxPool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 2D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool2D(FunctionalTensor input, (int, int) kernelSize, (int, int)? stride = null, (int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 4);
            input = input.Float();
            var kernelArray = new[] { kernelSize.Item1, kernelSize.Item2 };
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0 };
            return FunctionalLayer.MaxPool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 3D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool3D(FunctionalTensor input, int kernelSize, int? stride = null, int padding = 0)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 5);
            input = input.Float();
            var s = stride ?? kernelSize;
            var kernelArray = new[] { kernelSize, kernelSize, kernelSize };
            var strideArray = new[] { s, s, s };
            var paddingArray = new[] { padding, padding, padding, padding, padding, padding };
            return FunctionalLayer.MaxPool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }

        /// <summary>
        /// Returns the result of a 3D maximum pooling of the input.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="kernelSize">The size of the kernel.</param>
        /// <param name="stride">The optional stride of the pooling. The default value is the kernel size.</param>
        /// <param name="padding">The amount of padding on the spatial dimensions of the input.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor MaxPool3D(FunctionalTensor input, (int, int, int) kernelSize, (int, int, int)? stride = null, (int, int, int)? padding = null)
        {
            // TODO add auto padding, ceil_mode, count_include_pad
            DeclareRank(input, 5);
            input = input.Float();
            var strideArray = new[] { stride?.Item1 ?? kernelSize.Item1, stride?.Item2 ?? kernelSize.Item2, stride?.Item3 ?? kernelSize.Item3 };
            var paddingArray = new[] { padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0, padding?.Item1 ?? 0, padding?.Item2 ?? 0, padding?.Item3 ?? 0 };
            var kernelArray = new[] { kernelSize.Item1, kernelSize.Item2, kernelSize.Item3 };
            return FunctionalLayer.MaxPool(input, kernelArray, strideArray, paddingArray, Layers.AutoPad.NotSet);
        }
    }
}
