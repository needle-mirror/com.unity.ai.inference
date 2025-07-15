using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Google.Protobuf;
using Onnx;
using Unity.Mathematics;

[assembly: InternalsVisibleTo("Unity.InferenceEngine.EditorTests")]
[assembly: InternalsVisibleTo("Unity.InferenceEngine.ONNX.Editor")]
[assembly: InternalsVisibleTo("Unity.InferenceEngine.Editor")]

namespace Unity.InferenceEngine.ONNX.Editor
{
    /// <summary>
    /// Represents a converter from an ONNX model to Inference Engine format.
    /// </summary>
    class ONNXModelConverter
    {
        // Configuration
        string m_DirectoryPath;
        string m_FilePath;

        /// <summary>
        /// Occurs when the metadata of the ONNX model is loaded.
        /// </summary>
        /// <remarks>
        /// This event is triggered during the conversion of an ONNX model to Inference Engine format, when
        /// <see cref="Convert"/> is called. The event handler receives an argument of type
        /// <see cref="ONNXModelMetadata"/> containing metadata loaded from ONNX model.
        /// </remarks>
        public event Action<ONNXModelMetadata> MetadataLoaded;

        internal Dictionary<string, int> DynamicDimConfigs = new();

        /// <summary>
        /// Converts an ONNX model to an Inference Engine `Model` object.
        /// </summary>
        /// <returns>The converted Inference Engine model.</returns>
        public Model Convert()
        {
            using var readStream = new FileStream(m_FilePath, FileMode.Open, FileAccess.Read);
            using var inputStream = new CodedInputStream(readStream);

            var onnxModel = new ModelProto();
            onnxModel.MergeFrom(inputStream);

            Model model = null;
            model = ConvertOnnxModel(onnxModel);

#if UNITY_EDITOR && UNITY_2023_2_OR_NEWER && ENABLE_CLOUD_SERVICES_ANALYTICS
            var data = new InferenceEngineAnalytics.Data()
            {
                allOperators = model.layers.Select(l => l.opName).Distinct().ToArray(),
                importWarningSeverity = Warnings.Select(w => (int)w.MessageSeverity).ToArray(),
                importWarningMessages = Warnings.Select(w => w.Message).ToArray(),
                modelLayerCount = model.layers.Count,
            };
            InferenceEngineAnalytics.SendEvent(data);
#endif

            return model;
        }

        /// <summary>
        /// Initializes and returns an instance of `ONNXModelConverter`.
        /// </summary>
        /// <param name="filePath">The path of the asset to convert.</param>
        public ONNXModelConverter(string filePath)
        {
            m_FilePath = filePath;
            m_DirectoryPath = Path.GetDirectoryName(m_FilePath);
        }

        // import helpers
        int m_IndexCount;
        Dictionary<string, int> m_NameToIndex;

        // Returns layer index given name or add layer to count
        int NameToIndex(string name)
        {
            if (string.IsNullOrEmpty(name))
                return -1;
            return m_NameToIndex[name];
        }

        int AppendName(string name)
        {
            if (string.IsNullOrEmpty(name))
                return -1;
            var index = m_IndexCount;
            m_NameToIndex[name] = index;
            m_IndexCount++;
            return index;
        }

        int AppendNewLayer()
        {
            var index = m_IndexCount;
            m_IndexCount++;
            return index;
        }

        void OnNode(Model model, long defaultOpsetVersion, ONNXNodeWrapper node)
        {
            var opType = node.OperatorType;
            if (opType == "Constant")
            {
                if (node.HasAttribute("value"))
                {
                    var constant = ONNXConstantsLoader.LoadConstant(node.GetRequiredTensor("value"), m_DirectoryPath);
                    constant.index = AppendName(node.Name);
                    model.AddConstant(constant);
                }
                else if (node.HasAttribute("value_float"))
                {
                    var value = node.GetRequiredFloat("value_float");
                    var constant = new Constant(AppendName(node.Name), new TensorShape(), new[] { value });
                    model.AddConstant(constant);
                }
                else if (node.HasAttribute("value_floats"))
                {
                    var values = node.GetRequiredFloatArray("value_floats");
                    var constant = new Constant(AppendName(node.Name), new TensorShape(values.Length), values);
                    model.AddConstant(constant);
                }
                else if (node.HasAttribute("value_int"))
                {
                    var value = node.GetRequiredInt("value_int");
                    var constant = new Constant(AppendName(node.Name), new TensorShape(), new[] { value });
                    model.AddConstant(constant);
                }
                else if (node.HasAttribute("value_ints"))
                {
                    var values = node.GetRequiredIntArray("value_ints");
                    var constant = new Constant(AppendName(node.Name), new TensorShape(values.Length), values);
                    model.AddConstant(constant);
                }
                else
                {
                    node.UnsupportedAttribute("sparse_value");
                    node.UnsupportedAttribute("value_string");
                    node.UnsupportedAttribute("value_strings");
                    Warn(WarningType.Error, $"<b>{opType}</b>: Required attribute `<b>value</b>`, `<b>value_int(s)</b>` or `<b>value_float(s)</b>`");
                    Debug.LogError(Warnings.Last().Message);
                }
            }
            // Layer.Activation
            else if (opType == "Celu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                model.AddLayer(new Layers.Celu(alpha).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Elu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                model.AddLayer(new Layers.Elu(alpha).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Erf")
            {
                model.AddLayer(new Layers.Erf().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Gelu")
            {
                var approximate = node.GetOptionalString("approximate", "none");
                if (approximate.Equals("tanh"))
                    model.AddLayer(new Layers.GeluFast().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
                else
                    model.AddLayer(new Layers.Gelu().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Hardmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                model.AddLayer(new Layers.Hardmax(axis).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "HardSigmoid")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.2f);
                var beta = node.GetOptionalFloat("beta", 0.5f);
                model.AddLayer(new Layers.HardSigmoid(alpha, beta).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "HardSwish")
            {
                model.AddLayer(new Layers.HardSwish().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "LeakyRelu")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.01f);
                model.AddLayer(new Layers.LeakyRelu(alpha).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Mish")
            {
                model.AddLayer(new Layers.Mish().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "PRelu")
            {
                model.AddLayer(new Layers.PRelu().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Relu")
            {
                model.AddLayer(new Layers.Relu().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Selu")
            {
                var alpha = node.GetOptionalFloat("alpha", defaultOpsetVersion < 6 ? 1.6732f : 1.67326319f);
                var gamma = node.GetOptionalFloat("gamma", defaultOpsetVersion < 6 ? 1.0507f : 1.05070102f);
                model.AddLayer(new Layers.Selu(alpha, gamma).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Sigmoid")
            {
                model.AddLayer(new Layers.Sigmoid().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Softplus")
            {
                model.AddLayer(new Layers.Softplus().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Softsign")
            {
                model.AddLayer(new Layers.Softsign().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Tanh")
            {
                model.AddLayer(new Layers.Tanh().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ThresholdedRelu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                model.AddLayer(new Layers.ThresholdedRelu(alpha).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.ActivationNonLinear
            else if (opType == "LogSoftmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                model.AddLayer(new Layers.LogSoftmax(axis).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Softmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                model.AddLayer(new Layers.Softmax(axis).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Convolution
            else if (opType == "Conv")
            {
                // Conv-1, Conv-11

                var autoPad = node.AutoPadMode();
                var dilations = node.GetOptionalIntArray("dilations", new[] { 1, 1, 1, 1, 1, 1 });
                var group = node.GetOptionalInt("group", 1);
                var pads = node.GetOptionalIntArray("pads", new int[12]);
                var strides = node.GetOptionalIntArray("strides", new[] { 1, 1, 1, 1, 1, 1 });
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);

                model.AddLayer(new Layers.Conv(autoPad, dilations, group, pads, strides, kernelShape, Layers.FusableActivation.None).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.OptionalInput(2))).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ConvTranspose")
            {
                // ConvTranspose-1, ConvTranspose-11

                node.UnsupportedAttribute("output_shape", "null");

                var outputPadding = node.GetOptionalIntArray("output_padding", new[] { 0, 0, 0, 0, 0, 0 });
                var autoPad = node.AutoPadMode();
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);
                var dilations = node.GetOptionalIntArray("dilations", new[] { 1, 1, 1, 1, 1, 1 });
                var group = node.GetOptionalInt("group", 1);
                var pads = node.GetOptionalIntArray("pads", new int[12]);
                var strides = node.GetOptionalIntArray("strides", new[] { 1, 1, 1, 1, 1, 1 });

                model.AddLayer(new Layers.ConvTranspose(autoPad, dilations, group, outputPadding, pads, strides, kernelShape, Layers.FusableActivation.None).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.OptionalInput(2))).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Dimension
            else if (opType == "Shape")
            {
                // Shape-1, Shape-13, Shape-15
                var start = node.GetOptionalInt("start", 0);
                var end = node.GetOptionalInt("end", TensorShape.maxRank);
                model.AddLayer(new Layers.Shape(start, end).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Size")
            {
                // Size-1, Size-13
                model.AddLayer(new Layers.Size().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Generator
            else if (opType == "ConstantOfShape")
            {
                UnityEngine.Debug.Assert(node.InputCount > 0);

                if (!node.HasAttribute("value"))
                {
                    model.AddLayer(new Layers.ConstantOfShape(DataType.Float, 0.0f, 0).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
                    return;
                }

                var constant = ONNXConstantsLoader.LoadConstant(node.GetRequiredTensor("value"), m_DirectoryPath);
                if (constant.dataType == DataType.Int)
                {
                    var value = constant.weights.Get<int>(0);
                    model.AddLayer(new Layers.ConstantOfShape(DataType.Int, 0f, value).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    var value = constant.weights.Get<float>(0);
                    model.AddLayer(new Layers.ConstantOfShape(DataType.Float, value, 0).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
                }
                constant.weights.Dispose();
            }
            else if (opType == "Range")
            {
                model.AddLayer(new Layers.Range().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "OneHot")
            {
                // OneHot-9, OneHot-11
                var axis = node.GetOptionalInt("axis", -1);
                var allowNegativeIndexes = true;
                model.AddLayer(new Layers.OneHot(axis, allowNegativeIndexes).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Indexing
            else if (opType == "ArgMax")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                model.AddLayer(new Layers.ArgMax(axis, keepdims, selectLastIndex).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ArgMin")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                model.AddLayer(new Layers.ArgMin(axis, keepdims, selectLastIndex).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Gather")
            {
                var axis = node.GetOptionalInt("axis", 0);
                model.AddLayer(new Layers.Gather(axis).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "GatherElements")
            {
                var axis = node.GetOptionalInt("axis", 0);
                model.AddLayer(new Layers.GatherElements(axis).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "GatherND")
            {
                var batchDims = node.GetOptionalInt("batch_dims", 0);
                model.AddLayer(new Layers.GatherND(batchDims).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "NonZero")
            {
                model.AddLayer(new Layers.NonZero().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Scatter")
            {
                // Scatter-9 maps to ScatterElements
                var axis = node.GetOptionalInt("axis", 0);
                model.AddLayer(new Layers.ScatterElements(axis, Layers.ScatterReductionMode.None).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ScatterElements")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var reduction = node.GetOptionalString("reduction", "none") switch
                {
                    "none" => Layers.ScatterReductionMode.None,
                    "add" => Layers.ScatterReductionMode.Add,
                    "mul" => Layers.ScatterReductionMode.Mul,
                    "max" => Layers.ScatterReductionMode.Max,
                    "min" => Layers.ScatterReductionMode.Min,
                    _ => throw new ArgumentOutOfRangeException()
                };

                model.AddLayer(new Layers.ScatterElements(axis, reduction).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ScatterND")
            {
                var reduction = node.GetOptionalString("reduction", "none") switch
                {
                    "none" => Layers.ScatterReductionMode.None,
                    "add" => Layers.ScatterReductionMode.Add,
                    "mul" => Layers.ScatterReductionMode.Mul,
                    "max" => Layers.ScatterReductionMode.Max,
                    "min" => Layers.ScatterReductionMode.Min,
                    _ => throw new ArgumentOutOfRangeException()
                };

                model.AddLayer(new Layers.ScatterND(reduction).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "TopK")
            {
                var axis = node.GetOptionalInt("axis", -1);
                var largest = node.GetOptionalInt("largest", 1) == 1;
                var sorted = node.GetOptionalInt("sorted", 1) == 1;
                if (defaultOpsetVersion < 10)
                {
                    // TopK-1
                    var k = node.GetRequiredInt("k");
                    var kConstant = new Constant(AppendNewLayer(), new TensorShape(1), new[] { k });
                    model.AddConstant(kConstant);
                    model.AddLayer(new Layers.TopK(axis, largest, sorted).SetInputs(NameToIndex(node.Input0), kConstant.index).SetOutputs(AppendName(node.Output0), AppendName(node.Output1)));
                }
                else
                {
                    // TopK-10, TopK-11
                    model.AddLayer(new Layers.TopK(axis, largest, sorted).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Output0), AppendName(node.Output1)));
                }
            }
            // Layer.Logical
            else if (opType == "And")
            {
                model.AddLayer(new Layers.And().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Compress")
            {
                var hasAxis = node.HasAttribute("axis");
                var axis = node.GetOptionalInt("axis", 0);
                model.AddLayer(new Layers.Compress(hasAxis, axis).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Equal")
            {
                model.AddLayer(new Layers.Equal().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Greater")
            {
                model.AddLayer(new Layers.Greater().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "GreaterOrEqual")
            {
                model.AddLayer(new Layers.GreaterOrEqual().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "IsInf")
            {
                var detectNegative = node.GetOptionalInt("detect_negative", 1) != 0;
                var detectPositive = node.GetOptionalInt("detect_positive", 1) != 0;
                model.AddLayer(new Layers.IsInf(detectNegative, detectPositive).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "IsNaN")
            {
                model.AddLayer(new Layers.IsNaN().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Less")
            {
                model.AddLayer(new Layers.Less().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "LessOrEqual")
            {
                model.AddLayer(new Layers.LessOrEqual().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Not")
            {
                model.AddLayer(new Layers.Not().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Or")
            {
                model.AddLayer(new Layers.Or().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Xor")
            {
                model.AddLayer(new Layers.Xor().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Where")
            {
                model.AddLayer(new Layers.Where().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Math
            else if (opType == "Abs")
            {
                model.AddLayer(new Layers.Abs().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Add")
            {
                model.AddLayer(new Layers.Add().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Ceil")
            {
                model.AddLayer(new Layers.Ceil().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Clip")
            {
                if (defaultOpsetVersion < 11)
                {
                    // Clip-1, Clip-6
                    var min = node.GetOptionalFloat("min", float.MinValue);
                    var minConstant = new Constant(AppendNewLayer(), new TensorShape(), new[] { min });
                    model.AddConstant(minConstant);
                    var max = node.GetOptionalFloat("max", float.MaxValue);
                    var maxConstant = new Constant(AppendNewLayer(), new TensorShape(), new[] { max });
                    model.AddConstant(maxConstant);
                    model.AddLayer(new Layers.Clip().SetInputs(NameToIndex(node.Input0), minConstant.index, maxConstant.index).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    // Clip-11, Clip-12, Clip-13 or Clip-1, Clip-6 with no min or max
                    model.AddLayer(new Layers.Clip().SetInputs(NameToIndex(node.Input0), NameToIndex(node.OptionalInput(1)), NameToIndex(node.OptionalInput(2))).SetOutputs(AppendName(node.Name)));
                }
            }
            else if (opType == "CumSum")
            {
                var reverse = node.GetOptionalInt("reverse", 0) == 1;
                var exclusive = node.GetOptionalInt("exclusive", 0) == 1;
                model.AddLayer(new Layers.CumSum(reverse, exclusive).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Div")
            {
                model.AddLayer(new Layers.Div().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Einsum")
            {
                var inputs = new int[node.InputCount];
                for (var i = 0; i < node.InputCount; i++)
                    inputs[i] = NameToIndex(node.RequiredInput(i));
                model.AddLayer(new Layers.Einsum(node.GetRequiredString("equation")).SetInputs(inputs).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Exp")
            {
                model.AddLayer(new Layers.Exp().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Floor")
            {
                model.AddLayer(new Layers.Floor().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Gemm")
            {
                var transposeA = node.GetOptionalInt("transA", 0) == 1;
                var transposeB = node.GetOptionalInt("transB", 0) == 1;

                var alpha = node.GetOptionalFloat("alpha", 1.0f);
                var scalarMadA = AppendNewLayer();
                model.AddLayer(new Layers.ScalarMad(DataType.Float, alpha, 0, 0, 0).SetInputs(NameToIndex(node.Input0)).SetOutputs(scalarMadA));

                var hasC = node.InputCount == 3 && !string.IsNullOrEmpty(node.Inputs[2]);
                if (hasC)
                {
                    var matMulIndex = AppendNewLayer();
                    model.AddLayer(new Layers.MatMul2D(transposeA, transposeB).SetInputs(scalarMadA, NameToIndex(node.Input1)).SetOutputs(matMulIndex));
                    var beta = node.GetOptionalFloat("beta", 1.0f);
                    var scalarMadC = AppendNewLayer();
                    model.AddLayer(new Layers.ScalarMad(DataType.Float, beta, 0, 0, 0).SetInputs(NameToIndex(node.Input2)).SetOutputs(scalarMadC));
                    model.AddLayer(new Layers.Add().SetInputs(matMulIndex, scalarMadC).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    model.AddLayer(new Layers.MatMul2D(transposeA, transposeB).SetInputs(scalarMadA, NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
                }
            }
            else if (opType == "Log")
            {
                model.AddLayer(new Layers.Log().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "MatMul")
            {
                model.AddLayer(new Layers.MatMul().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Max")
            {
                var prevIndex = NameToIndex(node.RequiredInput(0));
                for (var i = 1; i < node.InputCount - 1; i++)
                {
                    var currentIndex = AppendNewLayer();
                    model.AddLayer(new Layers.Max().SetInputs(NameToIndex(node.RequiredInput(i)), prevIndex).SetOutputs(currentIndex));
                    prevIndex = currentIndex;
                }
                model.AddLayer(new Layers.Max().SetInputs(NameToIndex(node.RequiredInput(node.InputCount - 1)), prevIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Mean")
            {
                var prevIndex = AppendNewLayer();
                model.AddLayer(new Layers.Add().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(prevIndex));
                for (var i = 2; i < node.InputCount; i++)
                {
                    var currentIndex = AppendNewLayer();
                    model.AddLayer(new Layers.Add().SetInputs(NameToIndex(node.RequiredInput(i)), prevIndex).SetOutputs(currentIndex));
                    prevIndex = currentIndex;
                }
                model.AddLayer(new Layers.ScalarMad(DataType.Float, 1.0f / node.InputCount, 0, 0, 0).SetInputs(prevIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Min")
            {
                var prevIndex = NameToIndex(node.RequiredInput(0));
                for (var i = 1; i < node.InputCount - 1; i++)
                {
                    var currentIndex = AppendNewLayer();
                    model.AddLayer(new Layers.Min().SetInputs(NameToIndex(node.RequiredInput(i)), prevIndex).SetOutputs(currentIndex));
                    prevIndex = currentIndex;
                }
                model.AddLayer(new Layers.Min().SetInputs(NameToIndex(node.RequiredInput(node.InputCount - 1)), prevIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Mod")
            {
                var fmod = node.GetOptionalInt("fmod", 0) != 0;
                model.AddLayer(new Layers.Mod(fmod).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Mul")
            {
                model.AddLayer(new Layers.Mul().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Neg")
            {
                model.AddLayer(new Layers.Neg().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Pow")
            {
                // Pow-1, Pow-7, Pow-12, Pow-13
                model.AddLayer(new Layers.Pow().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Reciprocal")
            {
                model.AddLayer(new Layers.Reciprocal().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Round")
            {
                model.AddLayer(new Layers.Round().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Shrink")
            {
                var bias = node.GetOptionalFloat("bias", 0f);
                var lambd = node.GetOptionalFloat("lambd", 0.5f);
                model.AddLayer(new Layers.Shrink(bias, lambd).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Sign")
            {
                model.AddLayer(new Layers.Sign().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Sqrt")
            {
                model.AddLayer(new Layers.Sqrt().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Sub")
            {
                model.AddLayer(new Layers.Sub().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Sum")
            {
                var add0Index = AppendNewLayer();
                model.AddLayer(new Layers.Add().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(add0Index));
                for (var i = 2; i < node.InputCount; i++)
                {
                    var addIIndex = AppendNewLayer();
                    model.AddLayer(new Layers.Add().SetInputs(NameToIndex(node.RequiredInput(i)), add0Index).SetOutputs(addIIndex));
                    add0Index = addIIndex;
                }
                model.AddLayer(new Layers.Identity().SetInputs(add0Index).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Normalization
            else if (opType == "BatchNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                model.AddLayer(new Layers.BatchNormalization(epsilon).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2), NameToIndex(node.Input3), NameToIndex(node.Input4)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "InstanceNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                model.AddLayer(new Layers.InstanceNormalization(epsilon).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "LayerNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                node.UnsupportedAttribute("axis", -1);
                model.AddLayer(new Layers.LayerNormalization(epsilon).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.OptionalInput(2))).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "LRN")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.0001f);
                var beta = node.GetOptionalFloat("beta", 0.75f);
                var bias = node.GetOptionalFloat("bias", 1.0f);
                var size = node.GetRequiredInt("size");
                model.AddLayer(new Layers.LRN(alpha, beta, bias, size).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.ObjectDetection
            else if (opType == "NonMaxSuppression")
            {
                var centerPointBox = (node.GetOptionalInt("center_point_box", 0) == 0) ? Layers.CenterPointBox.Corners : Layers.CenterPointBox.Center;
                model.AddLayer(new Layers.NonMaxSuppression(centerPointBox).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.OptionalInput(2)), NameToIndex(node.OptionalInput(3)), NameToIndex(node.OptionalInput(4))).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "RoiAlign")
            {
                Layers.RoiCoordinateTransformationMode coordinateTransformMode;
                if (defaultOpsetVersion < 16)
                {
                    coordinateTransformMode = Layers.RoiCoordinateTransformationMode.OutputHalfPixel;
                }
                else
                {
                    coordinateTransformMode = node.GetOptionalString("coordinate_transformation_mode", "half_pixel") switch
                    {
                        "output_half_pixel" => Layers.RoiCoordinateTransformationMode.OutputHalfPixel,
                        "half_pixel" => Layers.RoiCoordinateTransformationMode.HalfPixel,
                        _ => throw new ArgumentOutOfRangeException()
                    };
                }

                var mode = node.GetOptionalString("mode", "avg") switch
                {
                    "avg" => Layers.RoiPoolingMode.Avg,
                    "max" => Layers.RoiPoolingMode.Max,
                    _ => throw new ArgumentOutOfRangeException()
                };
                var outputHeight = node.GetOptionalInt("output_height", 1);
                var outputWidth = node.GetOptionalInt("output_width", 1);
                var samplingRatio = node.GetOptionalInt("sampling_ratio", 0);
                var spatialScale = node.GetOptionalFloat("spatial_scale", 1.0f);

                model.AddLayer(new Layers.RoiAlign(mode, outputHeight, outputWidth, samplingRatio, spatialScale, coordinateTransformMode).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Pooling
            else if (opType == "AveragePool")
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);
                node.UnsupportedAttribute("count_include_pad", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", new int[2 * kernelShape.Length]);
                var strides = node.GetOptionalIntArray("strides", null);

                if (strides == null)
                {
                    strides = new int[kernelShape.Length];
                    for (var i = 0; i < strides.Length; i++)
                        strides[i] = 1;
                }

                model.AddLayer(new Layers.AveragePool(kernelShape, strides, pads, autopad).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "GlobalAveragePool")
            {
                model.AddLayer(new Layers.GlobalAveragePool().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "GlobalMaxPool")
            {
                model.AddLayer(new Layers.GlobalMaxPool().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "MaxPool")
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", new int[2 * kernelShape.Length]);
                var strides = node.GetOptionalIntArray("strides", null);

                if (strides == null)
                {
                    strides = new int[kernelShape.Length];
                    for (var i = 0; i < strides.Length; i++)
                        strides[i] = 1;
                }

                model.AddLayer(new Layers.MaxPool(kernelShape, strides, pads, autopad).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Random
            else if (opType == "Bernoulli")
            {
                var dataType = node.GetDataType(defaultValue: DataType.Float);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                model.AddLayer(new Layers.Bernoulli(dataType, hasSeed, seed).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Multinomial")
            {
                node.IgnoredAttribute("dtype", "dtype can only be int32 or int64 which both map to Tensor<int>");
                var samples = node.GetOptionalInt("sample_size", 1);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                model.AddLayer(new Layers.Multinomial(samples, hasSeed, seed).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "RandomNormal")
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                var shape = node.GetRequiredIntArray("shape");
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                model.AddLayer(new Layers.RandomNormal(mean, scale, shape, hasSeed, seed).SetInputs().SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "RandomNormalLike")
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                model.AddLayer(new Layers.RandomNormalLike(mean, scale, hasSeed, seed).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "RandomUniform")
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                var shape = node.GetRequiredIntArray("shape");
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                model.AddLayer(new Layers.RandomUniform(low, high, shape, hasSeed, seed).SetInputs().SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "RandomUniformLike")
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                model.AddLayer(new Layers.RandomUniformLike(low, high, hasSeed, seed).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Recurrent
            else if (opType == "LSTM")
            {
                var hiddenSize = node.GetRequiredInt("hidden_size");
                var direction = node.Direction();
                var activationsNode = node.Activations();
                var activationAlphaNode = node.GetOptionalFloatArray("activation_alpha", null);
                var activationBetaNode = node.GetOptionalFloatArray("activation_beta", null);

                var numDirections = direction == Layers.RnnDirection.Bidirectional ? 2 : 1;

                var activations = new Layers.RnnActivation[3 * numDirections];
                var activationAlpha = new float[3 * numDirections];
                var activationBeta = new float[3 * numDirections];
                for (var i = 0; i < 3 * numDirections; i++)
                {
                    activations[i] = i % 3 == 0 ? Layers.RnnActivation.Sigmoid : Layers.RnnActivation.Tanh;
                    if (activationsNode != null && i < activationsNode.Length)
                        activations[i] = activationsNode[i];
                    switch (activations[i])
                    {
                        case Layers.RnnActivation.Affine:
                            activationAlpha[i] = 1.0f;
                            break;
                        case Layers.RnnActivation.LeakyRelu:
                            activationAlpha[i] = 0.01f;
                            break;
                        case Layers.RnnActivation.ThresholdedRelu:
                            activationAlpha[i] = 1.0f;
                            break;
                        case Layers.RnnActivation.ScaledTanh:
                            activationAlpha[i] = 1.0f;
                            activationBeta[i] = 1.0f;
                            break;
                        case Layers.RnnActivation.HardSigmoid:
                            activationAlpha[i] = 0.2f;
                            activationBeta[i] = 0.5f;
                            break;
                        case Layers.RnnActivation.Elu:
                            activationAlpha[i] = 1.0f;
                            break;
                    }

                    if (activationAlphaNode != null && i < activationAlphaNode.Length)
                        activationAlpha[i] = activationAlphaNode[i];
                    if (activationBetaNode != null && i < activationBetaNode.Length)
                        activationBeta[i] = activationBetaNode[i];
                }

                var clip = node.GetOptionalFloat("clip", float.MaxValue);
                var inputForget = node.GetOptionalInt("input_forget", 0) != 0;
                var layout = node.Layout();

                var lstm = new Layers.LSTM(hiddenSize, direction, activations, activationAlpha, activationBeta, clip, inputForget, layout).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2), NameToIndex(node.OptionalInput(3)), NameToIndex(node.OptionalInput(4)), NameToIndex(node.OptionalInput(5)), NameToIndex(node.OptionalInput(6)), NameToIndex(node.OptionalInput(7))).SetOutputs(AppendName(node.Output0), AppendName(node.OptionalOutput(1)), AppendName(node.OptionalOutput(2)));
                model.AddLayer(lstm);
            }
            // Layer.Reduction
            else if (opType == "ReduceL1")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceL1(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceL2")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceL2(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceLogSum")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceLogSum(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceLogSumExp")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceLogSumExp(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceMax")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceMax(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceMean")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceMean(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceMin")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceMin(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceProd")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceProd(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceSum")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 13)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceSum(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ReduceSumSquare")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axesIndex = -1;
                if (defaultOpsetVersion < 18)
                {
                    var axes = node.GetOptionalIntArray("axes", null);
                    if (axes != null)
                    {
                        axesIndex = AppendNewLayer();
                        var axesConstant = new Constant(axesIndex, new TensorShape(axes.Length), axes);
                        model.AddConstant(axesConstant);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axesIndex = NameToIndex(node.OptionalInput(1));
                }

                model.AddLayer(new Layers.ReduceSumSquare(keepDims, noopWithEmptyAxes).SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
            }
            // Layer.Transformation
            else if (opType == "Cast")
            {
                var toOnnxType = (TensorProto.Types.DataType)node.GetRequiredInt("to");
                var toDataType = ONNXNodeWrapper.DataTypeFromOnnxDataType(toOnnxType, OnUnsupported: () =>
                {
                    Warn(WarningType.Error, $"Unsupported tensor dataType: {toOnnxType}.");
                    Debug.LogError(Warnings.Last().Message);
                });
                model.AddLayer(new Layers.Cast(toDataType).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "CastLike")
            {
                model.AddLayer(new Layers.CastLike().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Concat")
            {
                var axis = node.GetRequiredInt("axis");
                var inputs = new int[node.InputCount];
                for (var i = 0; i < node.InputCount; i++)
                    inputs[i] = NameToIndex(node.RequiredInput(i));
                model.AddLayer(new Layers.Concat(axis).SetInputs(inputs).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "DepthToSpace")
            {
                var modeType = node.GetOptionalString("mode", "DCR");
                var mode = modeType == "DCR" ? Layers.DepthToSpaceMode.DepthColumnRow : Layers.DepthToSpaceMode.ColumnRowDepth;
                var blocksize = node.GetRequiredInt("blocksize");
                model.AddLayer(new Layers.DepthToSpace(blocksize, mode).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Expand")
            {
                // Expand-8, Expand-13
                model.AddLayer(new Layers.Expand().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Flatten")
            {
                var axis = node.GetOptionalInt("axis", 1);
                model.AddLayer(new Layers.Flatten(axis).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "GridSample")
            {
                var mode = node.GetOptionalString("mode", "linear") switch
                {
                    "bilinear" => Layers.InterpolationMode.Linear,
                    "linear" => Layers.InterpolationMode.Linear,
                    "cubic" => Layers.InterpolationMode.Cubic,
                    "bicubic" => Layers.InterpolationMode.Cubic,
                    "nearest" => Layers.InterpolationMode.Nearest,
                    _ => throw new ArgumentOutOfRangeException()
                };
                var paddingMode = node.GetOptionalString("padding_mode", "zeros") switch
                {
                    "zeros" => Layers.PaddingMode.Zeros,
                    "border" => Layers.PaddingMode.Border,
                    "reflection" => Layers.PaddingMode.Reflection,
                    _ => throw new ArgumentOutOfRangeException()
                };
                var alignCorners = node.GetOptionalInt("align_corners", 0) == 1;
                model.AddLayer(new Layers.GridSample(mode, paddingMode, alignCorners).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Dropout")
            {
                model.AddLayer(new Layers.Identity().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Identity")
            {
                model.AddLayer(new Layers.Identity().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Pad")
            {
                var mode = node.PadMode();
                if (defaultOpsetVersion < 11)
                {
                    // Pad-1 or Pad-2
                    var pads = node.GetRequiredIntArray(node.HasAttribute("pads") ? "pads" : "paddings");
                    var padsIndex = AppendNewLayer();
                    model.AddConstant(new Constant(padsIndex, new TensorShape(pads.Length), pads));

                    var value = node.GetOptionalFloat("value", 0f);
                    var valueIndex = AppendNewLayer();
                    model.AddConstant(new Constant(valueIndex, new TensorShape(), new[] { value }));

                    model.AddLayer(new Layers.Pad(mode).SetInputs(NameToIndex(node.Input0), padsIndex, valueIndex, -1).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    // Pad-11, Pad-13, Pad-18
                    var constantIndex = node.InputCount > 2 ? NameToIndex(node.Inputs[2]) : -1;
                    var axesIndex = node.InputCount > 3 ? NameToIndex(node.Inputs[3]) : -1;
                    model.AddLayer(new Layers.Pad(mode).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), constantIndex, axesIndex).SetOutputs(AppendName(node.Name)));
                }
            }
            else if (opType == "Reshape")
            {
                if (defaultOpsetVersion < 5)
                {
                    // Reshape-1
                    var shape = node.GetRequiredIntArray("shape");
                    var shapeIndex = AppendNewLayer();
                    model.AddConstant(new Constant(shapeIndex, new TensorShape(shape.Length), shape));
                    model.AddLayer(new Layers.Reshape(false).SetInputs(NameToIndex(node.Input0), shapeIndex).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    // Reshape-5, Reshape-13, Reshape-14
                    var allowZero = node.GetOptionalInt("allowzero", 0) != 0;
                    model.AddLayer(new Layers.Reshape(allowZero).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
                }
            }
            else if (opType == "Resize")
            {
                var mode = node.InterpolationMode();
                var axes = node.GetOptionalIntArray("axes", null);
                if (defaultOpsetVersion < 11)
                {
                    // Resize-10
                    model.AddLayer(new Layers.Resize(Layers.ScaleMode.Scales, Layers.CoordTransformMode.Asymmetric, mode, Layers.NearestMode.Floor, axes).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    node.UnsupportedAttribute("cubic_coeff_a", -0.75f);
                    node.UnsupportedAttribute("exclude_outside", 0);
                    node.UnsupportedAttribute("extrapolation_value", 0);
                    var coordinateTransformMode = node.CoordinateTransformMode();
                    var nearestMode = node.NearestMode();
                    if (node.InputCount == 3 || string.IsNullOrEmpty(node.Inputs[3]))
                    {
                        // Resize-11, Resize-13, Resize-18 with scales
                        model.AddLayer(new Layers.Resize(Layers.ScaleMode.Scales, coordinateTransformMode, mode, nearestMode, axes).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input2)).SetOutputs(AppendName(node.Name)));
                    }
                    else if (node.InputCount == 4)
                    {
                        // Resize-11, Resize-13, Resize-18 with sizes
                        model.AddLayer(new Layers.Resize(Layers.ScaleMode.Sizes, coordinateTransformMode, mode, nearestMode, axes).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input3)).SetOutputs(AppendName(node.Name)));
                    }
                }
            }
            else if (opType == "Slice")
            {
                if (defaultOpsetVersion < 10)
                {
                    // Slice-1
                    var starts = node.GetRequiredIntArray("starts");
                    var startsIndex = AppendNewLayer();
                    model.AddConstant(new Constant(startsIndex, new TensorShape(starts.Length), starts));

                    var ends = node.GetRequiredIntArray("ends");
                    var endsIndex = AppendNewLayer();
                    model.AddConstant(new Constant(endsIndex, new TensorShape(ends.Length), ends));

                    if (node.HasAttribute("axes"))
                    {
                        var axes = node.GetRequiredIntArray("axes");
                        var axesIndex = AppendNewLayer();
                        model.AddConstant(new Constant(axesIndex, new TensorShape(axes.Length), axes));
                        model.AddLayer(new Layers.Slice().SetInputs(NameToIndex(node.Input0), startsIndex, endsIndex, axesIndex, -1).SetOutputs(AppendName(node.Name)));
                    }
                    else
                    {
                        model.AddLayer(new Layers.Slice().SetInputs(NameToIndex(node.Input0), startsIndex, endsIndex, -1, -1).SetOutputs(AppendName(node.Name)));
                    }
                }
                else
                {
                    // Slice-10, Slice-11, Slice-13
                    model.AddLayer(new Layers.Slice().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1), NameToIndex(node.Input2), NameToIndex(node.OptionalInput(3)), NameToIndex(node.OptionalInput(4))).SetOutputs(AppendName(node.Name)));
                }
            }
            else if (opType == "SpaceToDepth")
            {
                var blocksize = node.GetRequiredInt("blocksize");
                model.AddLayer(new Layers.SpaceToDepth(blocksize).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Split")
            {
                var axis = node.GetOptionalInt("axis", 0);
                int[] outputs = new int[node.OutputCount];
                for (var i = 0; i < node.OutputCount; i++)
                    outputs[i] = AppendName(node.RequiredOutput(i));
                if (node.HasAttribute("split"))
                {
                    // Split-1, Split-2, Split-11 with "split" attribute
                    var split = node.GetRequiredIntArray("split");
                    var splitIndex = AppendNewLayer();
                    model.AddConstant(new Constant(splitIndex, new TensorShape(split.Length), split));
                    model.AddLayer(new Layers.Split(axis, node.OutputCount).SetInputs(NameToIndex(node.Input0), splitIndex).SetOutputs(outputs));
                }
                else if (!string.IsNullOrEmpty(node.OptionalInput(1)))
                {
                    // Split-1, Split-2, Split-11, Split-13, Split-18 with split tensor
                    model.AddLayer(new Layers.Split(axis, node.OutputCount).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(outputs));
                }
                else
                {
                    // Split-1, Split-2, Split-11, Split-13, Split-18 with num_outputs
                    var numOutputs = node.GetOptionalInt("num_outputs", node.Outputs.Length);
                    model.AddLayer(new Layers.Split(axis, numOutputs).SetInputs(NameToIndex(node.Input0), -1).SetOutputs(outputs));
                }
            }
            else if (opType == "Squeeze")
            {
                if (defaultOpsetVersion < 13 && node.HasAttribute("axes"))
                {
                    // Squeeze-1, Squeeze-11 with given axes
                    var axes = node.GetRequiredIntArray("axes");
                    var axesIndex = AppendNewLayer();

                    model.AddConstant(new Constant(axesIndex, new TensorShape(axes.Length), axes));
                    model.AddLayer(new Layers.Squeeze().SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    // Squeeze-13 or Squeeze-1, Squeeze-11 without given axes
                    model.AddLayer(new Layers.Squeeze().SetInputs(NameToIndex(node.Input0), NameToIndex(node.OptionalInput(1))).SetOutputs(AppendName(node.Name)));
                }
            }
            else if (opType == "Tile")
            {
                model.AddLayer(new Layers.Tile().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Transpose")
            {
                var permutations = node.GetOptionalIntArray("perm", null);
                model.AddLayer(new Layers.Transpose(permutations).SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Trilu")
            {
                var upper = node.GetOptionalInt("upper", 1);
                model.AddLayer(new Layers.Trilu((Layers.TriluMode)upper).SetInputs(NameToIndex(node.Input0), NameToIndex(node.OptionalInput(1))).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Upsample")
            {
                var coordinateTransformMode = Layers.CoordTransformMode.Asymmetric;
                var mode = node.InterpolationMode();
                var nearestMode = Layers.NearestMode.Floor;
                if (defaultOpsetVersion < 9)
                {
                    // Upsample-7
                    var scales = node.GetRequiredFloatArray("scales");
                    var scalesIndex = AppendNewLayer();

                    model.AddConstant(new Constant(scalesIndex, new TensorShape(scales.Length), scales));
                    model.AddLayer(new Layers.Resize(Layers.ScaleMode.Scales, coordinateTransformMode, mode, nearestMode, null).SetInputs(NameToIndex(node.Input0), scalesIndex).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    // Upsample-9
                    model.AddLayer(new Layers.Resize(Layers.ScaleMode.Scales, coordinateTransformMode, mode, nearestMode, null).SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
                }
            }
            else if (opType == "Unsqueeze")
            {
                if (defaultOpsetVersion < 13)
                {
                    // Unsqueeze-1, Unsqueeze-11
                    var axes = node.GetRequiredIntArray("axes");
                    var axesIndex = AppendNewLayer();

                    model.AddConstant(new Constant(axesIndex, new TensorShape(axes.Length), axes));
                    model.AddLayer(new Layers.Unsqueeze().SetInputs(NameToIndex(node.Input0), axesIndex).SetOutputs(AppendName(node.Name)));
                }
                else
                {
                    // Unsqueeze-13
                    model.AddLayer(new Layers.Unsqueeze().SetInputs(NameToIndex(node.Input0), NameToIndex(node.Input1)).SetOutputs(AppendName(node.Name)));
                }
            }
            // Layer.Trigonometric
            else if (opType == "Acos")
            {
                model.AddLayer(new Layers.Acos().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Acosh")
            {
                model.AddLayer(new Layers.Acosh().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Asin")
            {
                model.AddLayer(new Layers.Asin().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Asinh")
            {
                model.AddLayer(new Layers.Asinh().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Atan")
            {
                model.AddLayer(new Layers.Atan().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Atanh")
            {
                model.AddLayer(new Layers.Atanh().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Cos")
            {
                model.AddLayer(new Layers.Cos().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Cosh")
            {
                model.AddLayer(new Layers.Cosh().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Sin")
            {
                model.AddLayer(new Layers.Sin().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Sinh")
            {
                model.AddLayer(new Layers.Sinh().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "Tan")
            {
                model.AddLayer(new Layers.Tan().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            // Non standard ONNX
            else if (opType == "Swish")
            {
                model.AddLayer(new Layers.Swish().SetInputs(NameToIndex(node.Input0)).SetOutputs(AppendName(node.Name)));
            }
            else if (opType == "ImageScaler")
            {
                var attrBias = node.GetRequiredFloatArray("bias");
                var maxElements = attrBias.Length;
                var attrScale = Enumerable.Repeat(node.GetOptionalFloat("scale", 1.0f), maxElements).ToArray();

                var scaleIndex = AppendNewLayer();
                var biasIndex = AppendNewLayer();
                model.AddConstant(new Constant(scaleIndex, new TensorShape(maxElements), attrScale));
                model.AddConstant(new Constant(biasIndex, new TensorShape(maxElements), attrBias));
                model.AddLayer(new Layers.ScaleBias().SetInputs(NameToIndex(node.Input0), scaleIndex, biasIndex).SetOutputs(AppendName(node.Name)));
            }
            else
            {
                Warn(WarningType.Error, $"{opType} not supported");
                Debug.LogError(Warnings.Last().Message);
            }
        }

        // NOTE: It's questionable whether we should be doing this since the ONNX specification requires the graph to be
        // topologically sorted, but at least one network encountered that was exported from keras2onnx v1.7.0 produced
        // an incorrectly sorted graph. related example: https://github.com/onnx/keras-onnx/issues/184
        static List<NodeProto> SortTopologically(ModelProto onnxModel)
        {
            GraphProto onnxGraph = onnxModel.Graph;
            HashSet<string> encounteredNodes = new HashSet<string>();
            foreach (var i in onnxGraph.Input)
                encounteredNodes.Add(i.Name);
            foreach (var i in onnxGraph.Initializer)
                encounteredNodes.Add(i.Name);

            var sortedGraph = new List<NodeProto>();
            bool graphInSortedOrder = true;
            foreach (NodeProto node in onnxGraph.Node)
            {
                foreach (var input in node.Input)
                    graphInSortedOrder &= encounteredNodes.Contains(input);

                if (!graphInSortedOrder)
                    break;

                foreach (var output in node.Output)
                    encounteredNodes.Add(output);
                sortedGraph.Add(node);
            }

            if (graphInSortedOrder)
                return sortedGraph;

            sortedGraph.Clear();
            var nodesToSort = new Queue<NodeProto>();
            foreach (NodeProto node in onnxGraph.Node)
            {
                nodesToSort.Enqueue(node);
            }

            var requeueNodes = new Queue<NodeProto>();
            while (nodesToSort.Count > 0)
            {
                NodeProto node = nodesToSort.Dequeue();

                var allInputsExist = true;
                foreach (string input in node.Input)
                {
                    if (string.IsNullOrEmpty(input))
                        continue;

                    if (!sortedGraph.Exists(n => n.Output.Any(o => o == input))
                        && !onnxGraph.Input.Any(i => i.Name == input)
                        && !onnxGraph.Initializer.Any(i => i.Name == input))
                    {
                        allInputsExist = false;
                        break;
                    }
                }

                if (!allInputsExist)
                {
                    if (nodesToSort.Count != 0)
                    {
                        // Mark for re-processing again when (potentially) all inputs have been processed
                        // We use a separate list, so we don't continually spin on nodes that are missing inputs
                        if (!requeueNodes.Contains(node))
                            requeueNodes.Enqueue(node);
                        continue;
                    }

                    // Something must've gone wrong
                    throw new OnnxImportException($"Missing inputs to node {node.Name}, but there are no nodes to process.");
                }

                if (!sortedGraph.Contains(node))
                    sortedGraph.Add(node);

                // Now that we have at least processed a single new node, let's requeue
                while (requeueNodes.Count > 0)
                    nodesToSort.Enqueue(requeueNodes.Dequeue());
            }

            return sortedGraph;
        }

        Model ConvertOnnxModel(ModelProto onnxModel)
        {
            var model = new Model();
            m_IndexCount = 0;
            m_NameToIndex = new Dictionary<string, int>();

            long defaultOpsetVersion = 15;

            // Parse producer meta data
            foreach (var opsetSetIdProto in onnxModel.OpsetImport)
            {
                if (string.IsNullOrEmpty(opsetSetIdProto.Domain))
                    defaultOpsetVersion = opsetSetIdProto.Version;
            }
            model.ProducerName = onnxModel.ProducerName;
            if (!string.IsNullOrEmpty(onnxModel.ProducerVersion))
                model.ProducerName += $" v{onnxModel.ProducerVersion}";

            // Convert graph inputs & outputs
            var initializersByName = onnxModel.Graph.Initializer.ToDictionary(i => i.Name, i => true);
            var namedDims = new List<string>();
            foreach (var input in onnxModel.Graph.Input)
            {
                // skip input tensors that have initializer data, they are constant tensors not global inputs
                // also skip nodes that should be trimmed
                if (initializersByName.ContainsKey(input.Name))
                    continue;

                var onnxShape = input.Type.TensorType.Shape;
                var inputShape = DynamicTensorShape.DynamicOfRank(onnxShape.Dim.Count);

                for (var i = 0; i < inputShape.rank; i++)
                {
                    var dim = onnxShape.Dim[i];
                    switch (dim.ValueCase)
                    {
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.None:
                            inputShape[i] = DynamicTensorDim.Unknown;
                            break;
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam:
                            var index = namedDims.IndexOf(dim.DimParam);
                            if (index < 0)
                            {
                                index = namedDims.Count;
                                namedDims.Add(dim.DimParam);
                            }
                            inputShape[i] = DynamicTensorDim.Param((byte)index);
                            if (DynamicDimConfigs.TryGetValue(dim.DimParam, out var dimInt))
                                inputShape[i] = DynamicTensorDim.Int(dimInt);
                            break;
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue:
                            if (dim.DimValue < 0)
                                Warn(WarningType.Warning, "Tensor shape has negative index, treating as unknown dimension");
                            else
                                inputShape[i] = DynamicTensorDim.Int(dim.DimValue > int.MaxValue ? int.MaxValue : (int)dim.DimValue);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                }

                var inputDataType = ONNXNodeWrapper.DataTypeFromOnnxDataType((TensorProto.Types.DataType)input.Type.TensorType.ElemType);

                model.AddInput(input.Name, AppendName(input.Name), inputDataType, inputShape);
            }

            var weightsStream = new Dictionary<string, FileStream>();
            // Read constants from initializer list
            foreach (TensorProto initializer in onnxModel.Graph.Initializer)
            {
                if (initializer.DataLocation == TensorProto.Types.DataLocation.External)
                {
                    string name = initializer.ExternalData.Single(x => x.Key == "location").Value;
                    if (!weightsStream.ContainsKey(name))
                    {
                        string filePath = Path.Combine(m_DirectoryPath, name);
                        if (File.Exists(filePath))
                            weightsStream.Add(name, File.OpenRead(Path.Combine(m_DirectoryPath, name)));
                        else
                        {
                            Warn(WarningType.Error, $"External Weights file not found! Expecting: {filePath}");
                            return null;
                        }
                    }
                    var stream = weightsStream[name];
                    var constant = ONNXConstantsLoader.LoadConstant(initializer, stream);
                    constant.index = AppendName(initializer.Name);
                    model.AddConstant(constant);
                }
                else
                {
                    var constant = ONNXConstantsLoader.LoadConstant(initializer);
                    constant.index = AppendName(initializer.Name);
                    model.AddConstant(constant);
                }
            }
            foreach (var stream in weightsStream.Values)
                stream.Dispose();

            // Nodes are supposed to be sorted, but this isn't always the case
            var sortedGraph = SortTopologically(onnxModel);

            // Convert graph nodes
            foreach (NodeProto onnxNode in sortedGraph)
            {
                var node = new ONNXNodeWrapper(onnxNode);

                try
                {
                    OnNode(model, defaultOpsetVersion, node);
                }
                catch (Exception e)
                {
                    Warn(WarningType.Error, e.Message);
                    throw new OnnxImportException(Warnings.Last().Message);
                }
            }

            // delete unused outputs
            foreach (ValueInfoProto o in onnxModel.Graph.Output)
            {
                if (m_NameToIndex.TryGetValue(o.Name, out var index))
                    model.AddOutput(o.Name, index);
            }

            // add symbolic names to model
            model.symbolicDimNames = namedDims.ToArray();

            // strip :0 at the end of string name for TF import
            model = TrimTensorflowNames(model);

            // validate imported model
            if (!Warnings.Any(w => w.MessageSeverity == WarningType.Error))
            {
                model = ModelValidator.ValidateModel(model);
            }
            if (!Warnings.Any(w => w.MessageSeverity == WarningType.Error))
            {
                ModelOptimizer.OptimizeModel(ref model);
            }

            // Invoke metadata handlers
            var propDict = new Dictionary<string, string>();
            foreach (var prop in onnxModel.MetadataProps)
            {
                propDict[prop.Key] = prop.Value;
            }

            MetadataLoaded?.Invoke(new ONNXModelMetadata
            {
                DocString = onnxModel.DocString,
                Domain = onnxModel.Domain,
                IRVersion = onnxModel.IrVersion,
                MetadataProps = propDict,
                ProducerName = onnxModel.ProducerName,
                ProducerVersion = onnxModel.ProducerVersion,
                ModelVersion = onnxModel.ModelVersion,
            });

            return model;
        }

        static Model TrimTensorflowNames(Model model)
        {
            model.inputs   = model.inputs.Select(i => {
                i.name = TrimTensorflowName(i.name);
                return i;
            }).ToList();

            model.outputs   = model.outputs.Select(o => {
                o.name = TrimTensorflowName(o.name);
                return o;
            }).ToList();

            return model;
        }

        static string TrimTensorflowName(string name)
        {
            if (name.EndsWith(":0"))
                return name.Remove(name.Length-2);
            return name;
        }

        // Logging helpers
        void Warn(WarningType severity, string message)
        {
            Warnings.Add(new ImporterWarning(message, severity));
        }

        /// <summary>
        /// The warnings from the model importer.
        /// </summary>
        public List<ImporterWarning> Warnings { get; } = new List<ImporterWarning>();

        /// <summary>
        /// Represents types of warning from the model importer.
        /// </summary>
        internal enum WarningType
        {
            /// <summary>
            /// No error.
            /// </summary>
            None = 0,

            /// <summary>
            /// Information. Execution should run without errors.
            /// </summary>
            Info = 1,

            /// <summary>
            /// Warning. Execution should run, but may have issues with precision or speed.
            /// </summary>
            Warning = 2,

            /// <summary>
            /// Error. Execution won't run.
            /// </summary>
            Error = 3
        }

        /// <summary>
        /// Represents the data structure for a warning from the model importer.
        /// </summary>
        internal class ImporterWarning
        {
            /// <summary>
            /// A message.
            /// </summary>
            public string Message { get; }

            /// <summary>
            /// The severity of a warning.
            /// </summary>
            public WarningType MessageSeverity { get; }

            /// <summary>
            /// Initializes and returns an instance of `ImporterWarning`.
            /// </summary>
            /// <param name="severity">The severity of the warning as a `WarningType`</param>
            /// <param name="msg">The message text of the warning</param>
            public ImporterWarning(string msg, WarningType severity)
            {
                Message = msg;
                MessageSeverity = severity;
            }
        }
    }

    /// <summary>
    /// Represents an exception during the import of an ONNX model.
    /// </summary>
    class OnnxImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `OnnxImportException`.
        /// </summary>
        /// <param name="message">message</param>
        public OnnxImportException(string message) : base(message) { }
    }

    /// <summary>
    /// Represents an exception during the import of a ONNX layer.
    /// </summary>
    class OnnxLayerImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `ONNXLayerImportException`.
        /// </summary>
        /// <param name="message">message</param>
        public OnnxLayerImportException(string message) : base(message) { }
    }
}
