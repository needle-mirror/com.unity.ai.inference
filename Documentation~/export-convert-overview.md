# Export a model from a machine learning framework

Sentis currently supports model files in the following formats:
- ONNX (Open Neural Network Exchange)
- LiteRT (formerly TensorFlow Lite)
- .sentis (Sentis serialized format)

If your model is not in one of these formats, you must convert it.

Use the following table to determine the appropriate export workflow based on your machine learning framework.

|Machine learning framework|Export workflow|
|-|-|
|PyTorch|[Export and convert to ONNX](export-convert-onnx.md)|
|TensorFlow, Keras, Tensorflow.js|[Export and convert to LiteRT](export-convert-litert.md)<br>[Export and convert to ONNX](export-convert-onnx.md)|

## Additional resources

- [Profile a model](profile-a-model.md)
- [Export an ONNX file from a machine learning framework](export-convert-onnx.md)
- [Export a LiteRT file from a machine learning framework](export-convert-litert.md)
