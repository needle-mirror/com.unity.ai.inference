# Import settings for ONNX models

You can import machine learning (ML) models in the `.onnx` format with **Model Asset Import Settings**. These settings control how Unity processes the model before it's processed at runtime.

## Dynamic input shape support

Inference Engine supports ONNX models with dynamic input dimensions. This means that the model is compatible with input tensors of different shapes that match a given pattern. Dynamic dimensions in ONNX models are named, such as `batch_size` or `sequence_length`.

For example, an input with shape (batch_size, 3, 256, 256) can accept an input tensors with different batch sizes, such as:

* (1, 3, 256, 256)
* (2, 3, 256, 256)
* (4, 3, 256, 256), and other values for the batch dimension.

If you know the value of a dynamic dimension in advance, for example, if the batch dimension will always be `1`, you can set it to a static value. This enables Inference Engine to optimize the model for better inference speed and memory efficiency.

## Configure import settings

Use the **Model Asset Import Settings** window to change the import settings for the model.

To set static values for dynamic dimensions, follow these steps:

1. Open the **Model Asset Import Settings** from the **Project** window.
2. Set a static value for any dynamic input dimensions.
3. Select **Apply**.

The updated value will reflect in the **Inspector** for the Inference Engine model.

When you serialize the model to a `.sentis` file, the assigned static values are saved. However, you can’t modify input dimensions after serialization.

## Dynamic input dimensions

The following table describes the properties available for dynamic input dimensions:

| **Property** | ****Description** |
| ------------ | ----------------- |
| `name`       | The name of the dynamic dimension. This is automatically populated from the ONNX model, such as `batch_size` or `sequence_length`. You can't modify this field. |
| `value`      | The value assigned to a dynamic input dimension. A value of `-1` keeps the dimension dynamic, while any non-negative value (≥ `0`) sets it to a static size. |

For more information on the shape of your model inputs, refer to [Model inputs](models-concept.md#model-inputs).

## Additional resources

- [Import a model](import-a-model-file.md)
- [Supported models](supported-models.md)
- [Export an ONNX file from a machine learning framework](export-convert-onnx.md)
- [Understand models in Inference Engine](models-concept.md)