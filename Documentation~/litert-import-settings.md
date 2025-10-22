# Import settings for LiteRT models

You can import machine learning (ML) models in the `.tflite` format. These settings control how Unity processes the model before it's processed at runtime.

## Signatures

Sentis supports [signatures](https://ai.google.dev/edge/litert/models/signatures) for LiteRT models. Signatures specify the names, ordering and tensors to use as inputs and outputs for a runtime model

## Configure import settings

Use the **Model Asset Import Settings** window to change the import settings for the model.

To change the signature of the model, follow these steps:

1. Open the **Model Asset Import Settings** from the **Project** window.
2. Set a signature from the dropdown.
3. Select **Apply**.

The updated value will reflect in the **Inspector** for the Sentis model.

When you serialize the model to a `.sentis` file, the assigned inputs and outputs are saved. However, you can’t modify a signature after serialization.

## Additional resources

- [Import a model](import-a-model-file.md)
- [Supported models](supported-models.md)
- [Export a LiteRT file from a machine learning framework](export-convert-litert.md)
- [Sentis models](models-concept.md)
