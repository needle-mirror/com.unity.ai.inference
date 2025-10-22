using Unity.InferenceEngine.Tokenization.PreTokenizers;

namespace Unity.InferenceEngine.Tokenization
{
    /// <summary>
    /// Options for how to deal with the delimiter when splitting the input string.
    /// See <see cref="RegexSplitPreTokenizer"/>
    /// </summary>
    /// <seealso cref="SplitPreTokenizer"/>
    public enum SplitDelimiterBehavior
    {
        /// <summary>
        /// The delimiter is not included in the output tokens at all.
        /// </summary>
        Removed,

        /// <summary>
        /// The delimiter is kept as a separate token.
        /// </summary>
        Isolated,

        /// <summary>
        /// The delimiter is appended to the previous token.
        /// </summary>
        MergedWithPrevious,

        /// <summary>
        /// The delimiter is prepended to the next token.
        /// </summary>
        MergedWithNext,

        /// <summary>
        /// Variation of <see cref="Isolated"/>, but merges the contiguous delimiters.
        /// </summary>
        Contiguous,
    }
}
