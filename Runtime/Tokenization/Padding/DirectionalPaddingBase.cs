using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Unity.InferenceEngine.Tokenization.Padding
{
    /// <summary>
    /// Base type for directional padding processor.
    /// </summary>
    public abstract class DirectionalPaddingBase : IPadding
    {
        readonly Pool<List<int>> m_ListOfIntPool = new(() => new(), list => list.Clear());
        readonly Pool<List<Token>> m_ListOfTokenPool = new(() => new(), list => list.Clear());

        readonly IPaddingSizeProvider m_SizeProvider;

        /// <summary>
        /// The token to use to fill the final sequence with.
        /// </summary>
        protected readonly Token PadToken;

        /// <summary>
        /// Initializes a new instance of the <see cref="DirectionalPaddingBase" /> type.
        /// </summary>
        /// <param name="paddingSizeProvider">
        /// When applying the padding, this object provide the final size of the padded sequence.
        /// </param>
        /// <param name="padToken">
        /// The token to use to pad a sequence of token.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// <paramref name="paddingSizeProvider" /> cannot be null.
        /// </exception>
        protected DirectionalPaddingBase(
            [NotNull] IPaddingSizeProvider paddingSizeProvider,
            Token padToken)
        {
            PadToken = padToken;
            m_SizeProvider = paddingSizeProvider
                ?? throw new ArgumentNullException(nameof(paddingSizeProvider));
        }

        /// <summary>
        /// Apply the padding to sequences of tokens and add the result to the
        /// <paramref name="output" />.
        /// </summary>
        /// <param name="sequences">
        /// The collection of sequences of tokens to pad.
        /// </param>
        /// <param name="output">
        /// The target container of padded sequences.
        /// </param>
        public void Pad(IReadOnlyList<IReadOnlyList<Token>> sequences,
            Output<IEnumerable<Token>> output)
        {
            if (sequences == null)
                throw new ArgumentNullException(nameof(sequences));

            int padSize;
            using (m_ListOfIntPool.Get(out var sizes))
            {
                for (int i = 0, _ = sequences.Count; i < _; i++)
                    sizes.Add(sequences[i].Count);

                padSize = m_SizeProvider.GetPaddingSize(sizes);
            }

            using var tokensHandle = m_ListOfTokenPool.Get(out var paddedTokens);

            for (int i = 0, limitI = sequences.Count; i < limitI; i++)
            {
                var tokens = sequences[i];

                if (tokens.Count == padSize)
                    for (int j = 0, limitJ = tokens.Count; j < limitJ; j++)
                        paddedTokens.Add(tokens[j].SetAttention(true));

                else
                    PadSequence(tokens, padSize, paddedTokens.AsOutput());

                output.Add(paddedTokens);
                paddedTokens.Clear();
            }
        }

        /// <summary>
        /// Pads the <paramref name="input" /> sequence of tokens to reach the
        /// <paramref name="padSize" />.
        /// </summary>
        /// <param name="input">
        /// The sequence of tokens to pad.
        /// </param>
        /// <param name="padSize">
        /// The target size.
        /// </param>
        /// <param name="output">
        /// Recipient of padded tokens.
        /// </param>
        protected abstract void PadSequence([NotNull] IReadOnlyList<Token> input, int padSize,
            Output<Token> output);
    }
}
