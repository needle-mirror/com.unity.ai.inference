# Export and convert a file to ONNX

Inference Engine imports files in the Open Neural Network Exchange (ONNX) format or the `.sentis` serialized format. If your model is in a different format, convert it to ONNX before you import it into Unity.

The following sections describe how to export and convert models to the ONNX format.

## Export an ONNX file from a machine learning framework

Most machine learning frameworks let you to export models in ONNX format.

To export files in ONNX format from common machine learning frameworks, refer to the following documentation:

- [Exporting a model from PyTorch to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) on the PyTorch website.
- [Convert TensorFlow, Keras, Tensorflow.js and Tflite models to ONNX](https://github.com/onnx/tensorflow-onnx) on the ONNX GitHub repository.

> [!NOTE]
> To ensure compatibility with Inference Engine, set the ONNX opset version to `15` during export. For more information about ONNX compatibility, refer to [import a model file](import-a-model-file.md).

## Convert TensorFlow files to ONNX

TensorFlow uses two primary file types for saving models: SavedModel and Checkpoints.

The following sections explain each file type and how to convert them to the ONNX format.

### Model files

TensorFlow saves models in SavedModel files, which contain a complete TensorFlow program, including trained parameters and computation. SavedModels have the `.pb` file extension. For more information on SavedModels, refer to [Using the SavedModel format](https://www.tensorflow.org/guide/saved_model) (TensorFlow documentation).

To generate an ONNX file from a SavedModel, use the [tf2onnx](https://github.com/onnx/tensorflow-onnx) tool. This is a command line tool that works best with full path names.

### Checkpoints

[Checkpoints](https://www.tensorflow.org/guide/checkpoint) (TensorFlow documentation) contain only the parameters of the model.

Checkpoints in TensorFlow can consist up of two file formats:

- A file to store the graph, with the extension `.ckpt.meta`.
- A file to store the weights, with the extension `.ckpt`.

If you have both the graph and weight file types, use the [tf2onnx](https://github.com/onnx/tensorflow-onnx) tool to create an ONNX file.

If you only have the `.ckpt` file, find the Python code that constructs the model and loads in the weights. After that, proceed to [export the model to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

## Convert PyTorch files to ONNX

The following sections explain how to convert PyTorch files to the ONNX format.

### Model files

PyTorch model files usually have the `.pt` file extension.

To export a model file to ONNX, refer to the links in the following instructions:

1. [Load the model](https://pytorch.org/tutorials/beginner/saving_loading_models.html) in Python.
2. [Export the model](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) as an ONNX file. When you export your model, it's recommended to use Opset `15` or higher.

If your `.pt` file doesn't contain the model graph, find the Python code that constructs the model and loads in the weights. After that, [export the model to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

###  Checkpoints

You can create [Checkpoints](https://pytorch.org/docs/stable/checkpoint.html) in PyTorch to save the state of your model at any instance of time. Checkpoint files are usually denoted with the `.tar` or `.pth` extension.

To convert a checkpoint file to ONNX, find the Python code which constructs the model and loads in the weights. After that, proceed to [export the model to ONNX](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html).

## Additional resources

- [Open Neural Network Exchange](https://onnx.ai/)
- [Supported ONNX operators](supported-operators.md)
- [Profile a model](profile-a-model.md)
- [Convert TensorFlow, Keras, Tensorflow.js and Tflite models to ONNX](https://github.com/onnx/tensorflow-onnx)