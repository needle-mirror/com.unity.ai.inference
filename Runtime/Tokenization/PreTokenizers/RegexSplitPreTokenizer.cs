using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using JetBrains.Annotations;
using Unity.InferenceEngine.Tokenization.SplitDelimiterBehaviors;
using UnityEngine.Assertions;

namespace Unity.InferenceEngine.Tokenization.PreTokenizers
{
    /// <summary>
    /// Splits the input based on a regular expression.
    /// </summary>
    public class RegexSplitPreTokenizer : IPreTokenizer
    {
        readonly Pool<List<(Range offsets, bool isContent)>> m_ListOfRangesPool =
            new(() => new(), list => list.Clear());

        Regex m_Regex;
        ISplitDelimiterBehavior m_Behavior;

        /// <summary>
        /// Initializes a new instance of the <see cref="RegexSplitPreTokenizer"/> type.
        /// </summary>
        /// <param name="pattern">
        /// The regular expression pattern to use for splitting the input.
        /// </param>
        /// <param name="behavior">
        /// Indicates how to handle splits and patterns.
        /// <see cref="SplitDelimiterBehavior"/>
        /// </param>
        /// <param name="invert">
        /// Whether of not to invert the pattern.
        /// Not yet implemented.
        /// </param>
        /// <exception cref="ArgumentNullException">
        /// Thrown when <paramref name="pattern"/> is null.
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">
        /// Thrown when <paramref name="behavior"/> is not a valid <see cref="SplitDelimiterBehavior"/> value.
        /// </exception>
        public RegexSplitPreTokenizer([NotNull] string pattern, SplitDelimiterBehavior behavior,
            bool invert = false)
        {
            if (pattern == null)
                throw new ArgumentNullException(nameof(pattern));

            var regex = new Regex(pattern);

            ISplitDelimiterBehavior behaviorImpl = behavior switch
            {
                SplitDelimiterBehavior.Removed => SplitDelimiterRemove.Instance,
                SplitDelimiterBehavior.Isolated => SplitDelimiterIsolate.Instance,
                SplitDelimiterBehavior.MergedWithPrevious => SplitDelimiterMergeWithPrevious
                    .Instance,
                SplitDelimiterBehavior.MergedWithNext => SplitDelimiterMergeWithNext.Instance,
                SplitDelimiterBehavior.Contiguous => SplitDelimiterContiguous.Instance,
                _ => throw new ArgumentOutOfRangeException(nameof(behavior), behavior, null)
            };

            Init(regex, behaviorImpl);
        }

        /// <inheritdoc cref="RegexSplitPreTokenizer"/>
        internal RegexSplitPreTokenizer(Regex pattern, ISplitDelimiterBehavior behavior) =>
            Init(pattern, behavior);

        void Init(Regex pattern, ISplitDelimiterBehavior behavior)
        {
            m_Regex = pattern;
            m_Behavior = behavior;
        }

        /// <inheritdoc />
        public void PreTokenize(SubString input, Output<SubString> output)
        {
            if(input.IsNull)
                throw new ArgumentNullException(nameof(input));

            using var _ = m_ListOfRangesPool.Get(out var splits);

            var copy = input.ToString();
            var matches = m_Regex.Matches(copy);

            var expectedOffset = 0;
            for (var i = 0; i < matches.Count; i++)
            {
                var g = matches[i].Groups[0];
                var offsets = new Range(g.Index, g.Index + g.Length);

                var (offset, length) = offsets.GetOffsetAndLength(input.Length);
                if (offset > expectedOffset)
                    splits.Add((expectedOffset .. offset, false));

                splits.Add((offsets, true));
                expectedOffset = offset + length;
            }

            if (expectedOffset < input.Length)
                splits.Add((expectedOffset .. input.Length, false));

            m_Behavior.Apply(input, splits, output);
        }
    }
}
