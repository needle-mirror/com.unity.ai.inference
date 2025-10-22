using System;

namespace Unity.InferenceEngine
{
    public static partial class Functional
    {
        /// <summary>
        /// Computes the STFT of the input signal.
        /// </summary>
        /// <param name="input">The input signal tensor.</param>
        /// <param name="hop_length">The stride (in the signal) between two frames to be processed.</param>
        /// <param name="window">The optional window tensor that modulates a signal frame.</param>
        /// <param name="n_fft">The size of a single frame. If window is specified, has to be the same as the window length.</param>
        /// <param name="onesided">Returns only half of the DFT frequency results (real signals have a symmetric DFT spectrum).</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor STFT(FunctionalTensor input, int hop_length, FunctionalTensor window, int n_fft, bool onesided)
        {
            DeclareRank(input, 3);
            if (window != null)
                DeclareRank(window, 1);
            input = input.Float();
            window = window?.Float();
            return FunctionalLayer.STFT(input, Constant(hop_length), window, Constant(n_fft), windowedDFTMatrix: null, onesided);
        }

        /// <summary>
        /// Returns a Blackman window of shape [windowLength].
        /// </summary>
        /// <param name="windowLength">The size of the window.</param>
        /// <param name="periodic">If True, returns a window to be used as periodic function. If False, return a symmetric window.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor BlackmanWindow(int windowLength, bool periodic)
        {
            return FunctionalLayer.BlackmanWindow(Constant(windowLength), periodic);
        }

        /// <summary>
        /// Returns a Hamming window of shape [windowLength] with α = 0.54347826087 and β = 0.45652173913.
        /// </summary>
        /// <param name="windowLength">The size of the window.</param>
        /// <param name="periodic">If True, returns a window to be used as periodic function. If False, return a symmetric window.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor HammingWindow(int windowLength, bool periodic)
        {
            return FunctionalLayer.HammingWindow(Constant(windowLength), periodic);
        }

        /// <summary>
        /// Returns a Hann window of shape [windowLength].
        /// </summary>
        /// <param name="windowLength">The size of the window.</param>
        /// <param name="periodic">If True, returns a window to be used as periodic function. If False, return a symmetric window.</param>
        /// <returns>The output tensor.</returns>
        public static FunctionalTensor HannWindow(int windowLength, bool periodic)
        {
            return FunctionalLayer.HannWindow(Constant(windowLength), periodic);
        }
    }
}
