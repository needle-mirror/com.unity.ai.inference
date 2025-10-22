using System.Collections.Generic;
using JetBrains.Annotations;
using UnityEngine.Assertions;

namespace Unity.InferenceEngine.Tokenization.Padding
{
    /// <summary>
    /// Pads the sequences of tokens by adding tokens to the left.
    /// </summary>
    public class LeftPadding : DirectionalPaddingBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LeftPadding" /> type.
        /// </summary>
        /// <param name="paddingSizeProvider">
        /// Computes the target length of the padded sequences.
        /// </param>
        /// <param name="padToken">
        /// The token to use to pad a sequence of token.
        /// </param>
        public LeftPadding(
            [NotNull] IPaddingSizeProvider paddingSizeProvider,
            Token padToken) : base(paddingSizeProvider, padToken)
        {
        }

        /// <inheritdoc />
        protected override void PadSequence(IReadOnlyList<Token> tokens, int padSize,
            Output<Token> output)
        {
            Assert.IsNotNull(tokens);

            for (int i = 0, limit = padSize - tokens.Count; i < limit; i++)
                output.Add(PadToken.SetAttention(false).SetSpecial(true));

            for (int i = 0, _ = tokens.Count; i < _; i++)
                output.Add(tokens[i].SetAttention(true));
        }
    }
}
