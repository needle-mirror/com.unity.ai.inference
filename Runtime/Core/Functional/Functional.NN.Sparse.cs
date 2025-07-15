using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Returns a one hot tensor with given index values that has zeros everywhere except where the index of last dimension matches the corresponding value of the input tensor, in which case it will be 1.
        /// </summary>
        /// <param name="tensor">The index tensor.</param>
        /// <param name="numClasses">Total number of classes. If set to -1, the number of classes will be inferred as one greater than the largest class value in the input tensor.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor OneHot(FunctionalTensor tensor, int numClasses = -1)
        {
            FunctionalTensor depthTensor;
            if (numClasses == -1)
                depthTensor = ReduceMax(tensor, 0) + 1;
            else
                depthTensor = Constant(numClasses);
            return FromLayer(new Layers.OneHot(-1, true), new[] { tensor, depthTensor, Constant(new[] { 0, 1 }) });
        }
    }
}
