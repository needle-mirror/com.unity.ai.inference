using Newtonsoft.Json.Linq;
using Unity.InferenceEngine.Tokenization.Truncators;

namespace Unity.InferenceEngine.Tokenization.Parsers.HuggingFace.Truncation
{
    [HfTruncation("LongestFirst")]
    class LongestFirstTruncationBuilder : IComponentBuilder<ITruncator>
    {
        public ITruncator Build(JToken parameters, HuggingFaceParser parser)
        {
            var directionData = parameters["direction"];
            var direction = directionData is not null ? directionData.Value<string>() : "Right";

            IRangeGenerator rangeGenerator = direction switch
            {
                "Right" => new RightDirectionRangeGenerator(),
                "Left" => new LeftDirectionRangeGenerator(),
                _ => throw new($"Unknown direction: {direction}")
            };

            var maxLength = parameters.GetInteger("max_length");
            var stride = parameters.GetIntegerOptional("stride", 0);

            return new LongestFirstTruncator(rangeGenerator, maxLength, stride);
        }
    }
}
