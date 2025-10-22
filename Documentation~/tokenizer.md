# Tokenization

Use the built-in [`tokenizer`](xref:Unity.InferenceEngine.Tokenization.ITokenizer) to convert text into numerical tokens that can be used as input for models that process text.

## Optional

The tokenizer is optional for Sentis. You can provide inputs from other sources if you prefer.

The tokenizer is designed for compatibility with the Hugging Face `tokenizers` Python library. To configure it, use the `tokenizer.json` file available in most Hugging Face model repositories.

## Tokenization workflow

A tokenizer processes text through several steps. Not all steps are required for every model:

### Normalization

Transforms the input string, such as replacing characters or applying Unicode normalization. This step outputs a new `string`. See [`normalizers`](xref:Unity.InferenceEngine.Tokenization.Normalizers.INormalizer).

### Pre-tokenization

Splits the normalized `string` into smaller parts for token conversion. See [`pre-tokenizers`](xref:Unity.InferenceEngine.Tokenization.PreTokenizers.IPreTokenizer).

### Models (token-to-ID conversion)

Maps each substring to a unique `integer` ID. See [`models`](xref:Unity.InferenceEngine.Tokenization.PreTokenizers.IPreTokenizer).

### Truncation

Enforces maximum input length by splitting or trimming token sequences. See [`truncation`](xref:Unity.InferenceEngine.Tokenization.Truncators.ITruncator).

### Padding

Adds tokens to ensure sequences have a fixed length when required by the model. See [`padding`](xref:Unity.InferenceEngine.Tokenization.Padding.IPadding).

### Post Processors

Adds special tokens, such as separators or markers, to prepare the sequence for the model. See [`post processors`](xref:Unity.InferenceEngine.Tokenization.PostProcessors.IPostProcessor).

### Decoders

Converts token IDs back into text after inference. Decoding is separate from the encoding steps and is only used when interpreting model outputs. See [`decoders`](xref:Unity.InferenceEngine.Tokenization.Decoders.IDecoder).

## Creating a tokenizer

At minimum, tokenization requires token-to-ID conversion. Most text-based models also require additional steps such as, normalization, pre-tokenization, or padding.

The following sample implementation is included with the package and available in the Unity Package Manager.```

### Encode input

After initialization, the tokenizer converts text inputs into sequences of IDs that you can pass to Sentis.

### Decode output

For text-based models, use the same tokenizer to decode the generated IDs back into readable text.

## Sample code

```cs
using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using Unity.InferenceEngine.Tokenization;
using Unity.InferenceEngine.Tokenization.Decoders;
using Unity.InferenceEngine.Tokenization.Mappers;
using Unity.InferenceEngine.Tokenization.Normalizers;
using Unity.InferenceEngine.Tokenization.Padding;
using Unity.InferenceEngine.Tokenization.PostProcessors;
using Unity.InferenceEngine.Tokenization.PostProcessors.Templating;
using Unity.InferenceEngine.Tokenization.PreTokenizers;
using Unity.InferenceEngine.Tokenization.Truncators;
using UnityEngine;

class TokenizerSample : MonoBehaviour
{
    static Tensor<int> Encode(ITokenizer tokenizer, string input)
    {
        // Generates the sequence
        var encoding = tokenizer.Encode(input);

        // Then you can use the encoding to generate your tensors.

        // Gets this ids
        // Other masks or available, like:
        // - attention
        // - type ids
        // - special mask.
        int[] ids = encoding.GetIds().ToArray();

        // Create a 3D tensor shape
        TensorShape shape = new TensorShape(1, 1, ids.Length);

        // Create a new tensor from the array
        return new Tensor<int>(shape, ids);
    }

    static string Decode(ITokenizer tokenizer, Tensor<int> tensor)
    {
        var ids = tensor.DownloadToArray();
        return tokenizer.Decode(ids);
    }

    static Dictionary<string, int> BuildVocabulary()
    {
        // This stub method returns a legitimate string to id mapping for the tokenizer.
        // It is usually built from a large configuration JSON file.
        return new Dictionary<string, int>();
    }

    static TokenConfiguration[] GetAddedTokens()
    {
        // This stub method returns a legitimate collection of token configuration.
        // Token configuration is the Hugging Face equivalent of added token.
        return Array.Empty<TokenConfiguration>();
    }

    /// This sample initializes a tokenizer based on All MiniLM L6 v2.
    public ITokenizer CreateTokenizer()
    {
        var vocabulary = BuildVocabulary();
        var addedTokens = GetAddedTokens();

        // Central step of the tokenizer
        var mapper = new WordPieceMapper(vocabulary, "[UNK]", "##", 100);


        // Preliminary steps of the tokenization:
        // - normalization (transforms the input string)
        // - pre-tokenization (splits the input string)

        var normalizer = new BertNormalizer(
            cleanText: true,
            handleCjkChars: true,
            stripAccents: null,
            lowerCase: true);

        var preTokenizer = new BertPreTokenizer();


        // Final steps of tokenization:
        // - truncation (splits the token sequences)
        // - post-processing (decorates the token sequences)
        // - padding (adds tokens to match a sequence size).

        var truncator = new LongestFirstTruncator(new RightDirectionRangeGenerator(), 128, 0);

        var clsId = addedTokens.Where(tc => tc.Value == "[CLS]").Select(tc => tc.Id).FirstOrDefault();
        var sepId = addedTokens.Where(tc => tc.Value == "[SEP]").Select(tc => tc.Id).FirstOrDefault();
        var padId = addedTokens.Where(tc => tc.Value == "[PAD]").Select(tc => tc.Id).FirstOrDefault();

        var postProcessor = new TemplatePostProcessor(
          new(Template.Parse("[CLS]:0 $A:0 [SEP]:0")),
          new(Template.Parse("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1")),
          new (string, int)[] { ("[CLS]", clsId), ("[SEP]", sepId) });

        var padding = new RightPadding(
          new FixedPaddingSizeProvider(128),
          new Token(padId, "[PAD]"));


        // Decoding.

        var decoder = new WordPieceDecoder("##", true);


        // Creates the tokenizer from all the components
        // initialized above.

        return new Tokenizer(
            mapper,
            normalizer: normalizer,
            preTokenizer: preTokenizer,
            truncator: truncator,
            postProcessor: postProcessor,
            paddingProcessor: padding,
            decoder: decoder,
            addedVocabulary: addedTokens);
    }
}
```
