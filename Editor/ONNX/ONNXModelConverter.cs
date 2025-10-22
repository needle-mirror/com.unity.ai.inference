using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Google.Protobuf;
using Unity.InferenceEngine.Graph;
using Unity.Mathematics;

namespace Unity.InferenceEngine.Editor.Onnx
{
    /// <summary>
    /// Represents a converter from an ONNX model to Sentis format.
    /// </summary>
    class ONNXModelConverter : ModelConverterBase
    {
        /// <summary>
        /// Occurs when the metadata of the ONNX model is loaded.
        /// </summary>
        /// <remarks>
        /// This event is triggered during the conversion of an ONNX model to Sentis format, when
        /// <see cref="Convert"/> is called. The event handler receives an argument of type
        /// <see cref="ONNXModelMetadata"/> containing metadata loaded from ONNX model.
        /// </remarks>
        public event Action<ONNXModelMetadata> MetadataLoaded;

        internal Dictionary<string, int> DynamicDimConfigs = new();

        /// <summary>
        /// Converts an ONNX model to a Sentis `Model` object.
        /// </summary>
        /// <returns>The converted Sentis model.</returns>
        public override Model Convert()
        {
            using var readStream = new FileStream(m_FilePath, FileMode.Open, FileAccess.Read);
            using var inputStream = new CodedInputStream(readStream);

            var onnxModel = new ModelProto();
            onnxModel.MergeFrom(inputStream);

            var model = ConvertOnnxModel(onnxModel);

#if UNITY_EDITOR && UNITY_2023_2_OR_NEWER && ENABLE_CLOUD_SERVICES_ANALYTICS
            var data = new SentisAnalytics.Data()
            {
                allOperators = model.layers.Select(l => l.opName).Distinct().ToArray(),
                importWarningSeverity = Warnings.Select(w => (int)w.MessageSeverity).ToArray(),
                importWarningMessages = Warnings.Select(w => w.Message).ToArray(),
                modelLayerCount = model.layers.Count,
            };
            SentisAnalytics.SendEvent(data);
#endif

            return model;
        }

        /// <summary>
        /// Initializes and returns an instance of `ONNXModelConverter`.
        /// </summary>
        /// <param name="filePath">The path of the asset to convert.</param>
        public ONNXModelConverter(string filePath)
            : base(filePath) { }

        void OnNode(GraphModule gm, Dictionary<string, Node> tensors, long defaultOpsetVersion, ONNXNodeWrapper node)
        {
            Node GetInput(int index)
            {
                if (index >= node.InputCount || string.IsNullOrEmpty(node.Inputs[index]))
                    return null;
                return tensors[node.Inputs[index]];
            }

            Node[] GetInputs()
            {
                var inputs = new Node[node.InputCount];
                for (var i = 0; i < node.InputCount; i++)
                    inputs[i] = GetInput(i);
                return inputs;
            }

            void SetOutput(Node output, int index = 0)
            {
                if (index >= node.OutputCount || string.IsNullOrEmpty(node.Outputs[index]))
                    return;
                tensors[node.Outputs[index]] = output;
            }

            void SetOutputs(Node[] outputs)
            {
                for (var i = 0; i < outputs.Length; i++)
                    SetOutput(outputs[i], i);
            }

            var opType = node.OperatorType;
            if (opType == "Constant")
            {
                if (node.HasAttribute("value"))
                {
                    var constantTensor = ONNXConstantsLoader.LoadConstant(node.GetRequiredTensor("value"), m_DirectoryPath);
                    var constantNode = gm.Constant(constantTensor);
                    SetOutput(constantNode);
                }
                else if (node.HasAttribute("value_float"))
                {
                    var value = node.GetRequiredFloat("value_float");
                    var constantNode = gm.Constant(value);
                    SetOutput(constantNode);
                }
                else if (node.HasAttribute("value_floats"))
                {
                    var values = node.GetRequiredFloatArray("value_floats");
                    var constant = gm.Constant(values);
                    SetOutput(constant);
                }
                else if (node.HasAttribute("value_int"))
                {
                    var value = node.GetRequiredInt("value_int");
                    var constant = gm.Constant(value);
                    SetOutput(constant);
                }
                else if (node.HasAttribute("value_ints"))
                {
                    var values = node.GetRequiredIntArray("value_ints");
                    var constant = gm.Constant(values);
                    SetOutput(constant);
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
                SetOutput(gm.Celu(GetInput(0), alpha));
            }
            else if (opType == "Elu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                SetOutput(gm.Elu(GetInput(0), alpha));
            }
            else if (opType == "Erf")
            {
                SetOutput(gm.Erf(GetInput(0)));
            }
            else if (opType == "Gelu")
            {
                var approximate = node.GetOptionalString("approximate", "none");
                if (approximate.Equals("tanh"))
                    SetOutput(gm.GeluFast(GetInput(0)));
                else
                    SetOutput(gm.Gelu(GetInput(0)));
            }
            else if (opType == "Hardmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                SetOutput(gm.Hardmax(GetInput(0), axis));
            }
            else if (opType == "HardSigmoid")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.2f);
                var beta = node.GetOptionalFloat("beta", 0.5f);
                SetOutput(gm.HardSigmoid(GetInput(0), alpha, beta));
            }
            else if (opType == "HardSwish")
            {
                SetOutput(gm.HardSwish(GetInput(0)));
            }
            else if (opType == "LeakyRelu")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.01f);
                SetOutput(gm.LeakyRelu(GetInput(0), alpha));
            }
            else if (opType == "Mish")
            {
                SetOutput(gm.Mish(GetInput(0)));
            }
            else if (opType == "PRelu")
            {
                SetOutput(gm.PRelu(GetInput(0), GetInput(1)));
            }
            else if (opType == "Relu")
            {
                SetOutput(gm.Relu(GetInput(0)));
            }
            else if (opType == "Selu")
            {
                var alpha = node.GetOptionalFloat("alpha", defaultOpsetVersion < 6 ? 1.6732f : 1.67326319f);
                var gamma = node.GetOptionalFloat("gamma", defaultOpsetVersion < 6 ? 1.0507f : 1.05070102f);
                SetOutput(gm.Selu(GetInput(0), alpha, gamma));
            }
            else if (opType == "Sigmoid")
            {
                SetOutput(gm.Sigmoid(GetInput(0)));
            }
            else if (opType == "Softplus")
            {
                SetOutput(gm.Softplus(GetInput(0)));
            }
            else if (opType == "Softsign")
            {
                SetOutput(gm.Softsign(GetInput(0)));
            }
            else if (opType == "Tanh")
            {
                SetOutput(gm.Tanh(GetInput(0)));
            }
            else if (opType == "ThresholdedRelu")
            {
                var alpha = node.GetOptionalFloat("alpha", 1f);
                SetOutput(gm.ThresholdedRelu(GetInput(0), alpha));
            }
            // Layer.ActivationNonLinear
            else if (opType == "LogSoftmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                SetOutput(gm.LogSoftmax(GetInput(0), axis));
            }
            else if (opType == "Softmax")
            {
                var axis = node.GetOptionalInt("axis", -1);
                SetOutput(gm.Softmax(GetInput(0), axis));
            }
            // Layer.Convolution
            else if (opType == "Conv")
            {
                // Conv-1, Conv-11

                var autoPadString = node.GetOptionalString("auto_pad", "NOTSET");
                var autoPad = autoPadString switch
                {
                    "NOTSET" => Layers.AutoPad.NotSet,
                    "VALID" => Layers.AutoPad.Valid,
                    "SAME_UPPER" => Layers.AutoPad.SameUpper,
                    "SAME_LOWER" => Layers.AutoPad.SameLower,
                    _ => Warn(WarningType.Warning, $"auto_pad `{autoPadString}` is not supported for Conv, using `NOTSET`.", Layers.AutoPad.NotSet)
                };
                var dilations = node.GetOptionalIntArray("dilations", new[] { 1, 1, 1, 1, 1, 1 });
                var group = node.GetOptionalInt("group", 1);
                var pads = node.GetOptionalIntArray("pads", new int[12]);
                var strides = node.GetOptionalIntArray("strides", new[] { 1, 1, 1, 1, 1, 1 });
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);

                SetOutput(gm.Conv(GetInput(0), GetInput(1), GetInput(2), autoPad, dilations, group, pads, strides, kernelShape, Layers.FusableActivation.None));
            }
            else if (opType == "ConvTranspose")
            {
                // ConvTranspose-1, ConvTranspose-11

                node.UnsupportedAttribute("output_shape", "null");

                var outputPadding = node.GetOptionalIntArray("output_padding", new[] { 0, 0, 0, 0, 0, 0 });
                var autoPadString = node.GetOptionalString("auto_pad", "NOTSET");
                var autoPad = autoPadString switch
                {
                    "NOTSET" => Layers.AutoPad.NotSet,
                    "VALID" => Layers.AutoPad.Valid,
                    "SAME_UPPER" => Layers.AutoPad.SameUpper,
                    "SAME_LOWER" => Layers.AutoPad.SameLower,
                    _ => Warn(WarningType.Warning, $"auto_pad `{autoPadString}` is not supported for ConvTranspose, using `NOTSET`.", Layers.AutoPad.NotSet)
                };
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);
                var dilations = node.GetOptionalIntArray("dilations", new[] { 1, 1, 1, 1, 1, 1 });
                var group = node.GetOptionalInt("group", 1);
                var pads = node.GetOptionalIntArray("pads", new int[12]);
                var strides = node.GetOptionalIntArray("strides", new[] { 1, 1, 1, 1, 1, 1 });

                SetOutput(gm.ConvTranspose(GetInput(0), GetInput(1), GetInput(2), autoPad, dilations, group, outputPadding, pads, strides, kernelShape, Layers.FusableActivation.None));
            }
            // Layer.Dimension
            else if (opType == "Shape")
            {
                // Shape-1, Shape-13, Shape-15
                var start = node.GetOptionalInt("start", 0);
                var end = node.GetOptionalInt("end", TensorShape.maxRank);
                SetOutput(gm.Shape(GetInput(0), start, end));
            }
            else if (opType == "Size")
            {
                // Size-1, Size-13
                SetOutput(gm.Size(GetInput(0)));
            }
            // Layer.Generator
            else if (opType == "ConstantOfShape")
            {
                UnityEngine.Debug.Assert(node.InputCount > 0);

                if (!node.HasAttribute("value"))
                {
                    SetOutput(gm.ConstantOfShape(GetInput(0), DataType.Float, 0.0f, 0));
                    return;
                }

                var constantTensor = ONNXConstantsLoader.LoadConstant(node.GetRequiredTensor("value"), m_DirectoryPath);
                if (constantTensor.dataType == DataType.Int)
                {
                    var value = constantTensor.AsSpan<int>()[0];
                    SetOutput(gm.ConstantOfShape(GetInput(0), DataType.Int, 0f, value));
                }
                else if (constantTensor.dataType == DataType.Float)
                {
                    var value = constantTensor.AsSpan<float>()[0];
                    SetOutput(gm.ConstantOfShape(GetInput(0), DataType.Float, value, 0));
                }
            }
            else if (opType == "Range")
            {
                SetOutput(gm.Range(GetInput(0), GetInput(1), GetInput(2)));
            }
            else if (opType == "OneHot")
            {
                // OneHot-9, OneHot-11
                var axis = node.GetOptionalInt("axis", -1);
                var allowNegativeIndexes = true;
                SetOutput(gm.OneHot(GetInput(0), GetInput(1), GetInput(2), axis, allowNegativeIndexes));
            }
            // Layer.Indexing
            else if (opType == "ArgMax")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                SetOutput(gm.ArgMax(GetInput(0), axis, keepdims, selectLastIndex));
            }
            else if (opType == "ArgMin")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                SetOutput(gm.ArgMin(GetInput(0), axis, keepdims, selectLastIndex));
            }
            else if (opType == "Gather")
            {
                var axis = node.GetOptionalInt("axis", 0);
                SetOutput(gm.Gather(GetInput(0), GetInput(1), axis));
            }
            else if (opType == "GatherElements")
            {
                var axis = node.GetOptionalInt("axis", 0);
                SetOutput(gm.GatherElements(GetInput(0), GetInput(1), axis));
            }
            else if (opType == "GatherND")
            {
                var batchDims = node.GetOptionalInt("batch_dims", 0);
                SetOutput(gm.GatherND(GetInput(0), GetInput(1), batchDims));
            }
            else if (opType == "NonZero")
            {
                SetOutput(gm.NonZero(GetInput(0)));
            }
            else if (opType == "Scatter")
            {
                // Scatter-9 maps to ScatterElements
                var axis = node.GetOptionalInt("axis", 0);
                SetOutput(gm.ScatterElements(GetInput(0), GetInput(1), GetInput(2), axis, Layers.ScatterReductionMode.None));
            }
            else if (opType == "ScatterElements")
            {
                var axis = node.GetOptionalInt("axis", 0);
                var reductionString = node.GetOptionalString("reduction", "none");
                var reduction = reductionString switch
                {
                    "none" => Layers.ScatterReductionMode.None,
                    "add" => Layers.ScatterReductionMode.Add,
                    "mul" => Layers.ScatterReductionMode.Mul,
                    "max" => Layers.ScatterReductionMode.Max,
                    "min" => Layers.ScatterReductionMode.Min,
                    _ => Warn(WarningType.Warning, $"reduction `{reductionString}` is not supported for ScatterElements, using `none`.", Layers.ScatterReductionMode.None)
                };

                SetOutput(gm.ScatterElements(GetInput(0), GetInput(1), GetInput(2), axis, reduction));
            }
            else if (opType == "ScatterND")
            {
                var reductionString = node.GetOptionalString("reduction", "none");
                var reduction = reductionString switch
                {
                    "none" => Layers.ScatterReductionMode.None,
                    "add" => Layers.ScatterReductionMode.Add,
                    "mul" => Layers.ScatterReductionMode.Mul,
                    "max" => Layers.ScatterReductionMode.Max,
                    "min" => Layers.ScatterReductionMode.Min,
                    _ => Warn(WarningType.Warning, $"reduction `{reductionString}` is not supported for ScatterND, using `none`.", Layers.ScatterReductionMode.None)
                };

                SetOutput(gm.ScatterND(GetInput(0), GetInput(1), GetInput(2), reduction));
            }
            else if (opType == "TopK")
            {
                var axis = node.GetOptionalInt("axis", -1);
                var largest = node.GetOptionalInt("largest", 1) == 1;
                var sorted = node.GetOptionalInt("sorted", 1) == 1;
                if (defaultOpsetVersion < 10)
                {
                    // TopK-1
                    var kValue = node.GetRequiredInt("k");
                    var k = gm.Constant(new[] { kValue });
                    SetOutputs(gm.TopK(GetInput(0), k, axis, largest, sorted));
                }
                else
                {
                    // TopK-10, TopK-11
                    SetOutputs(gm.TopK(GetInput(0), GetInput(1), axis, largest, sorted));
                }
            }
            // Layer.Logical
            else if (opType == "And")
            {
                SetOutput(gm.And(GetInput(0), GetInput(1)));
            }
            else if (opType == "Compress")
            {
                var hasAxis = node.HasAttribute("axis");
                var axis = node.GetOptionalInt("axis", 0);
                SetOutput(gm.Compress(GetInput(0), GetInput(1), hasAxis, axis));
            }
            else if (opType == "Equal")
            {
                SetOutput(gm.Equal(GetInput(0), GetInput(1)));
            }
            else if (opType == "Greater")
            {
                SetOutput(gm.Greater(GetInput(0), GetInput(1)));
            }
            else if (opType == "GreaterOrEqual")
            {
                SetOutput(gm.GreaterOrEqual(GetInput(0), GetInput(1)));
            }
            else if (opType == "IsInf")
            {
                var detectNegative = node.GetOptionalInt("detect_negative", 1) != 0;
                var detectPositive = node.GetOptionalInt("detect_positive", 1) != 0;
                SetOutput(gm.IsInf(GetInput(0), detectNegative, detectPositive));
            }
            else if (opType == "IsNaN")
            {
                SetOutput(gm.IsNaN(GetInput(0)));
            }
            else if (opType == "Less")
            {
                SetOutput(gm.Less(GetInput(0), GetInput(1)));
            }
            else if (opType == "LessOrEqual")
            {
                SetOutput(gm.LessOrEqual(GetInput(0), GetInput(1)));
            }
            else if (opType == "Not")
            {
                SetOutput(gm.Not(GetInput(0)));
            }
            else if (opType == "Or")
            {
                SetOutput(gm.Or(GetInput(0), GetInput(1)));
            }
            else if (opType == "Xor")
            {
                SetOutput(gm.Xor(GetInput(0), GetInput(1)));
            }
            else if (opType == "Where")
            {
                SetOutput(gm.Where(GetInput(0), GetInput(1), GetInput(2)));
            }
            // Layer.Math
            else if (opType == "Abs")
            {
                SetOutput(gm.Abs(GetInput(0)));
            }
            else if (opType == "Add")
            {
                SetOutput(gm.Add(GetInput(0), GetInput(1)));
            }
            else if (opType == "BitwiseAnd")
            {
                SetOutput(gm.BitwiseAnd(GetInput(0), GetInput(1)));
            }
            else if (opType == "BitwiseNot")
            {
                SetOutput(gm.BitwiseNot(GetInput(0)));
            }
            else if (opType == "BitwiseOr")
            {
                SetOutput(gm.BitwiseOr(GetInput(0), GetInput(1)));
            }
            else if (opType == "BitwiseXor")
            {
                SetOutput(gm.BitwiseXor(GetInput(0), GetInput(1)));
            }
            else if (opType == "Ceil")
            {
                SetOutput(gm.Ceil(GetInput(0)));
            }
            else if (opType == "Clip")
            {
                if (defaultOpsetVersion < 11)
                {
                    // Clip-1, Clip-6
                    var minValue = node.GetOptionalFloat("min", float.MinValue);
                    var min = gm.Constant(minValue);
                    var maxValue = node.GetOptionalFloat("max", float.MaxValue);
                    var max = gm.Constant(maxValue);
                    SetOutput(gm.Clip(GetInput(0), min, max));
                }
                else
                {
                    // Clip-11, Clip-12, Clip-13 or Clip-1, Clip-6 with no min or max
                    SetOutput(gm.Clip(GetInput(0), GetInput(1), GetInput(2)));
                }
            }
            else if (opType == "CumSum")
            {
                var reverse = node.GetOptionalInt("reverse", 0) == 1;
                var exclusive = node.GetOptionalInt("exclusive", 0) == 1;
                SetOutput(gm.CumSum(GetInput(0), GetInput(1), reverse, exclusive));
            }
            else if (opType == "Div")
            {
                SetOutput(gm.Div(GetInput(0), GetInput(1)));
            }
            else if (opType == "Einsum")
            {
                SetOutput(gm.Einsum(GetInputs(), node.GetRequiredString("equation")));
            }
            else if (opType == "Exp")
            {
                SetOutput(gm.Exp(GetInput(0)));
            }
            else if (opType == "Floor")
            {
                SetOutput(gm.Floor(GetInput(0)));
            }
            else if (opType == "Gemm")
            {
                var transposeA = node.GetOptionalInt("transA", 0) == 1;
                var transposeB = node.GetOptionalInt("transB", 0) == 1;

                var alpha = node.GetOptionalFloat("alpha", 1.0f);
                var a = GetInput(0);
                if (alpha != 1f)
                    a = gm.ScalarMad(a, DataType.Float, alpha, 0, 0, 0);

                var res = gm.MatMul2D(a, GetInput(1), transposeA, transposeB);
                var c = GetInput(2);
                if (c is not null)
                {
                    var beta = node.GetOptionalFloat("beta", 1.0f);
                    if (beta != 1f)
                        c = gm.ScalarMad(c, DataType.Float, beta, 0, 0, 0);
                    res = gm.Add(res, c);
                }
                SetOutput(res);
            }
            else if (opType == "Log")
            {
                SetOutput(gm.Log(GetInput(0)));
            }
            else if (opType == "MatMul")
            {
                SetOutput(gm.MatMul(GetInput(0), GetInput(1)));
            }
            else if (opType == "Max")
            {
                var prev = GetInput(0);
                for (var i = 1; i < node.InputCount - 1; i++)
                {
                    var current = GetInput(i);
                    prev = gm.Max(prev, current);
                }
                SetOutput(gm.Max(GetInput(node.InputCount - 1), prev));
            }
            else if (opType == "Mean")
            {
                var prev = GetInput(0);
                for (var i = 1; i < node.InputCount; i++)
                {
                    var current = GetInput(i);
                    prev = gm.Add(prev, current);
                }
                SetOutput(gm.ScalarMad(prev, DataType.Float, 1.0f / node.InputCount, 0, 0, 0));
            }
            else if (opType == "Min")
            {
                var prev = GetInput(0);
                for (var i = 1; i < node.InputCount - 1; i++)
                {
                    var current = GetInput(i);
                    prev = gm.Min(prev, current);
                }
                SetOutput(gm.Min(GetInput(node.InputCount - 1), prev));
            }
            else if (opType == "Mod")
            {
                var fmod = node.GetOptionalInt("fmod", 0) != 0;
                SetOutput(gm.Mod(GetInput(0), GetInput(1), fmod));
            }
            else if (opType == "Mul")
            {
                SetOutput(gm.Mul(GetInput(0), GetInput(1)));
            }
            else if (opType == "Neg")
            {
                SetOutput(gm.Neg(GetInput(0)));
            }
            else if (opType == "Pow")
            {
                // Pow-1, Pow-7, Pow-12, Pow-13
                SetOutput(gm.Pow(GetInput(0), GetInput(1)));
            }
            else if (opType == "Reciprocal")
            {
                SetOutput(gm.Reciprocal(GetInput(0)));
            }
            else if (opType == "Round")
            {
                SetOutput(gm.Round(GetInput(0)));
            }
            else if (opType == "Shrink")
            {
                var bias = node.GetOptionalFloat("bias", 0f);
                var lambd = node.GetOptionalFloat("lambd", 0.5f);
                SetOutput(gm.Shrink(GetInput(0), bias, lambd));
            }
            else if (opType == "Sign")
            {
                SetOutput(gm.Sign(GetInput(0)));
            }
            else if (opType == "Sqrt")
            {
                SetOutput(gm.Sqrt(GetInput(0)));
            }
            else if (opType == "Sub")
            {
                SetOutput(gm.Sub(GetInput(0), GetInput(1)));
            }
            else if (opType == "Sum")
            {
                var prev = GetInput(0);
                for (var i = 1; i < node.InputCount - 1; i++)
                {
                    var current = GetInput(i);
                    prev = gm.Add(prev, current);
                }
                SetOutput(gm.Add(GetInput(node.InputCount - 1), prev));
            }
            // Layer.Normalization
            else if (opType == "BatchNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                SetOutput(gm.BatchNormalization(GetInput(0), GetInput(1), GetInput(2), GetInput(3), GetInput(4), epsilon));
            }
            else if (opType == "InstanceNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                SetOutput(gm.InstanceNormalization(GetInput(0), GetInput(1), GetInput(2), epsilon));
            }
            else if (opType == "LayerNormalization")
            {
                var epsilon = node.GetOptionalFloat("epsilon", 1e-5f);
                node.UnsupportedAttribute("axis", -1);
                SetOutput(gm.LayerNormalization(GetInput(0), GetInput(1), GetInput(2), epsilon));
            }
            else if (opType == "LRN")
            {
                var alpha = node.GetOptionalFloat("alpha", 0.0001f);
                var beta = node.GetOptionalFloat("beta", 0.75f);
                var bias = node.GetOptionalFloat("bias", 1.0f);
                var size = node.GetRequiredInt("size");
                SetOutput(gm.LRN(GetInput(0), alpha, beta, bias, size));
            }
            // Layer.ObjectDetection
            else if (opType == "NonMaxSuppression")
            {
                var centerPointBox = (node.GetOptionalInt("center_point_box", 0) == 0) ? Layers.CenterPointBox.Corners : Layers.CenterPointBox.Center;
                SetOutput(gm.NonMaxSuppression(GetInput(0), GetInput(1), GetInput(2), GetInput(3), GetInput(4), centerPointBox));
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
                    var coordinateTransformModeString = node.GetOptionalString("coordinate_transformation_mode", "half_pixel");
                    coordinateTransformMode = coordinateTransformModeString switch
                    {
                        "output_half_pixel" => Layers.RoiCoordinateTransformationMode.OutputHalfPixel,
                        "half_pixel" => Layers.RoiCoordinateTransformationMode.HalfPixel,
                        _ => Warn(WarningType.Warning, $"coordinate_transformation_mode `{coordinateTransformModeString}` is not supported for RoiAlign, using `half_pixel`.", Layers.RoiCoordinateTransformationMode.HalfPixel)
                    };
                }

                var modeString = node.GetOptionalString("mode", "avg");
                var mode = modeString switch
                {
                    "avg" => Layers.RoiPoolingMode.Avg,
                    "max" => Layers.RoiPoolingMode.Max,
                    _ => Warn(WarningType.Warning, $"mode `{modeString}` is not supported for RoiAlign, using `avg`.", Layers.RoiPoolingMode.Avg)
                };
                var outputHeight = node.GetOptionalInt("output_height", 1);
                var outputWidth = node.GetOptionalInt("output_width", 1);
                var samplingRatio = node.GetOptionalInt("sampling_ratio", 0);
                var spatialScale = node.GetOptionalFloat("spatial_scale", 1.0f);

                SetOutput(gm.RoiAlign(GetInput(0), GetInput(1), GetInput(2), mode, outputHeight, outputWidth, samplingRatio, spatialScale, coordinateTransformMode));
            }
            // Layer.Pooling
            else if (opType == "AveragePool")
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);
                node.UnsupportedAttribute("count_include_pad", 0);

                var autoPadString = node.GetOptionalString("auto_pad", "NOTSET");
                var autoPad = autoPadString switch
                {
                    "NOTSET" => Layers.AutoPad.NotSet,
                    "VALID" => Layers.AutoPad.Valid,
                    "SAME_UPPER" => Layers.AutoPad.SameUpper,
                    "SAME_LOWER" => Layers.AutoPad.SameLower,
                    _ => Warn(WarningType.Warning, $"auto_pad `{autoPadString}` is not supported for AveragePool, using `NOTSET`.", Layers.AutoPad.NotSet)
                };

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", new int[2 * kernelShape.Length]);
                var strides = node.GetOptionalIntArray("strides", null);

                if (strides == null)
                {
                    strides = new int[kernelShape.Length];
                    for (var i = 0; i < strides.Length; i++)
                        strides[i] = 1;
                }

                SetOutput(gm.AveragePool(GetInput(0), kernelShape, strides, pads, autoPad));
            }
            else if (opType == "GlobalAveragePool")
            {
                SetOutput(gm.GlobalAveragePool(GetInput(0)));
            }
            else if (opType == "GlobalMaxPool")
            {
                SetOutput(gm.GlobalMaxPool(GetInput(0)));
            }
            else if (opType == "MaxPool")
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);

                var autoPadString = node.GetOptionalString("auto_pad", "NOTSET");
                var autoPad = autoPadString switch
                {
                    "NOTSET" => Layers.AutoPad.NotSet,
                    "VALID" => Layers.AutoPad.Valid,
                    "SAME_UPPER" => Layers.AutoPad.SameUpper,
                    "SAME_LOWER" => Layers.AutoPad.SameLower,
                    _ => Warn(WarningType.Warning, $"auto_pad `{autoPadString}` is not supported for MaxPool, using `NOTSET`.", Layers.AutoPad.NotSet)
                };

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", new int[2 * kernelShape.Length]);
                var strides = node.GetOptionalIntArray("strides", null);

                if (strides == null)
                {
                    strides = new int[kernelShape.Length];
                    for (var i = 0; i < strides.Length; i++)
                        strides[i] = 1;
                }

                SetOutput(gm.MaxPool(GetInput(0), kernelShape, strides, pads, autoPad));
            }
            // Layer.Random
            else if (opType == "Bernoulli")
            {
                var dataTypeValue = node.GetOptionalInt("dtype", (int)TensorProto.Types.DataType.Float);
                var dataType = ONNXNodeWrapper.DataTypeFromOnnxDataType((TensorProto.Types.DataType)dataTypeValue);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                SetOutput(gm.Bernoulli(GetInput(0), dataType, hasSeed, seed));
            }
            else if (opType == "Multinomial")
            {
                // dtype can only be int32 or int64 which both map to Tensor<int>
                var samples = node.GetOptionalInt("sample_size", 1);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                SetOutput(gm.Multinomial(GetInput(0), samples, hasSeed, seed));
            }
            else if (opType == "RandomNormal")
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                var shape = node.GetRequiredIntArray("shape");
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                SetOutput(gm.RandomNormal(mean, scale, shape, hasSeed, seed));
            }
            else if (opType == "RandomNormalLike")
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                SetOutput(gm.RandomNormalLike(GetInput(0), mean, scale, hasSeed, seed));
            }
            else if (opType == "RandomUniform")
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                var shape = node.GetRequiredIntArray("shape");
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                SetOutput(gm.RandomUniform(low, high, shape, hasSeed, seed));
            }
            else if (opType == "RandomUniformLike")
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                var hasSeed = node.HasAttribute("seed");
                var seed = hasSeed ? math.asint(node.GetRequiredFloat("seed")) : 0;
                SetOutput(gm.RandomUniformLike(GetInput(0), low, high, hasSeed, seed));
            }
            // Layer.Recurrent
            else if (opType == "LSTM")
            {
                var hiddenSize = node.GetRequiredInt("hidden_size");
                var directionString = node.GetOptionalString("direction", "forward");
                var direction = directionString switch
                {
                    "forward" => Layers.RnnDirection.Forward,
                    "reverse" => Layers.RnnDirection.Reverse,
                    "bidirectional" => Layers.RnnDirection.Bidirectional,
                    _ => Warn(WarningType.Warning, $"direction `{directionString}` is not supported for LSTM, using `forward`.", Layers.RnnDirection.Forward)
                };
                var numDirections = direction == Layers.RnnDirection.Bidirectional ? 2 : 1;

                var activationAlphaNode = node.GetOptionalFloatArray("activation_alpha", null);
                var activationBetaNode = node.GetOptionalFloatArray("activation_beta", null);

                var activationAlpha = new float[3 * numDirections];
                var activationBeta = new float[3 * numDirections];

                var activationsStringArray = node.GetOptionalStringArray("activations", null);
                var activations = new Layers.RnnActivation[3 * numDirections];
                for (var i = 0; i < 3 * numDirections; i++)
                {
                    var defaultActivation = i % 3 == 0 ? Layers.RnnActivation.Sigmoid : Layers.RnnActivation.Tanh;
                    if (activationsStringArray == null)
                    {
                        activations[i] = defaultActivation;
                    }
                    else
                    {
                        activations[i] = activationsStringArray[i] switch
                        {
                            "Relu" => Layers.RnnActivation.Relu,
                            "Tanh" => Layers.RnnActivation.Tanh,
                            "Sigmoid" => Layers.RnnActivation.Sigmoid,
                            "Affine" => Layers.RnnActivation.Affine,
                            "LeakyRelu" => Layers.RnnActivation.LeakyRelu,
                            "ThresholdedRelu" => Layers.RnnActivation.ThresholdedRelu,
                            "ScaledTanh" => Layers.RnnActivation.ScaledTanh,
                            "HardSigmoid" => Layers.RnnActivation.HardSigmoid,
                            "Elu" => Layers.RnnActivation.Elu,
                            "Softsign" => Layers.RnnActivation.Softsign,
                            "Softplus" => Layers.RnnActivation.Softplus,
                            _ => Warn(WarningType.Warning, $"activation `{activationsStringArray[i]}` is not supported for LSTM, using `{defaultActivation}`.", defaultActivation)
                        };
                    }

                    if (activationAlphaNode == null || activationAlphaNode.Length <= i)
                    {
                        activationAlpha[i] = activations[i] switch
                        {
                            Layers.RnnActivation.Affine => 1.0f,
                            Layers.RnnActivation.LeakyRelu => 0.01f,
                            Layers.RnnActivation.ThresholdedRelu => 1.0f,
                            Layers.RnnActivation.ScaledTanh => 1.0f,
                            Layers.RnnActivation.HardSigmoid => 0.2f,
                            Layers.RnnActivation.Elu => 1.0f,
                            _ => 0
                        };
                    }
                    else
                    {
                        activationAlpha[i] = activationAlphaNode[i];
                    }

                    if (activationBetaNode == null || activationBetaNode.Length <= i)
                    {
                        activationBeta[i] = activations[i] switch
                        {
                            Layers.RnnActivation.ScaledTanh => 1.0f,
                            Layers.RnnActivation.HardSigmoid => 0.5f,
                            _ => 0
                        };
                    }
                    else
                    {
                        activationBeta[i] = activationBetaNode[i];
                    }
                }

                var clip = node.GetOptionalFloat("clip", float.MaxValue);
                var inputForget = node.GetOptionalInt("input_forget", 0) != 0;
                var layoutInt = node.GetOptionalInt("layout", 0);
                var layout = layoutInt switch
                {
                    0 => Layers.RnnLayout.SequenceFirst,
                    1 => Layers.RnnLayout.BatchFirst,
                    _ => Warn(WarningType.Warning, $"layout `{layoutInt}` is not supported for LSTM, using `0`.", Layers.RnnLayout.SequenceFirst)
                };

                SetOutputs(gm.LSTM(GetInput(0), GetInput(1), GetInput(2), GetInput(3), GetInput(4), GetInput(5), GetInput(6), GetInput(7), hiddenSize, direction, activations, activationAlpha, activationBeta, clip, inputForget, layout));
            }
            // Layer.Reduction
            else if (opType == "ReduceL1")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }

                SetOutput(gm.ReduceL1(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceL2")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }

                SetOutput(gm.ReduceL2(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceLogSum")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }

                SetOutput(gm.ReduceLogSum(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceLogSumExp")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }

                SetOutput(gm.ReduceLogSumExp(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceMax")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }

                SetOutput(gm.ReduceMax(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceMean")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }

                SetOutput(gm.ReduceMean(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceMin")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }

                SetOutput(gm.ReduceMin(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceProd")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axes = GetInput(1);
                }

                SetOutput(gm.ReduceProd(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceSum")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 13)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axes = GetInput(1);
                }

                SetOutput(gm.ReduceSum(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            else if (opType == "ReduceSumSquare")
            {
                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                var axes = GetInput(1);
                if (defaultOpsetVersion < 18)
                {
                    var axesArray = node.GetOptionalIntArray("axes", null);
                    if (axesArray != null)
                    {
                        axes = gm.Constant(axesArray);
                    }
                }
                else if (node.InputCount > 1)
                {
                    axes = GetInput(1);
                }

                SetOutput(gm.ReduceSumSquare(GetInput(0), axes, keepDims, noopWithEmptyAxes));
            }
            // Layer.Spectral
            else if (opType == "BlackmanWindow")
            {
                node.UnsupportedAttribute("output_datatype", 1);
                var periodic = node.GetOptionalInt("periodic", 1) == 1;
                var size = GetInput(0);
                SetOutput(gm.BlackmanWindow(size, periodic));
            }
            else if (opType == "DFT")
            {
                var inverse = node.GetOptionalInt("inverse", 0) == 1;
                var onesided = node.GetOptionalInt("onesided", 0) == 1;
                var input = GetInput(0);
                var dftLength = GetInput(1);
                var axis = GetInput(2);
                SetOutput(gm.DFT(input, dftLength, axis, dftMatrix: null, inverse, onesided));
            }
            else if (opType == "HammingWindow")
            {
                node.UnsupportedAttribute("output_datatype", 1);
                var periodic = node.GetOptionalInt("periodic", 1) == 1;
                var size = GetInput(0);
                SetOutput(gm.HammingWindow(size, periodic));
            }
            else if (opType == "HannWindow")
            {
                node.UnsupportedAttribute("output_datatype", 1);
                var periodic = node.GetOptionalInt("periodic", 1) == 1;
                var size = GetInput(0);
                SetOutput(gm.HannWindow(size, periodic));
            }
            else if (opType == "MelWeightMatrix")
            {
                node.UnsupportedAttribute("output_datatype", 1);
                var numMelBins = GetInput(0);
                var dftLength = GetInput(1);
                var sampleRate = GetInput(2);
                var lowerEdgeHertz = GetInput(3);
                var upperEdgeHertz = GetInput(4);
                SetOutput(gm.MelWeightMatrix(numMelBins, dftLength, sampleRate, lowerEdgeHertz, upperEdgeHertz));
            }
            else if (opType == "STFT")
            {
                var onesided = node.GetOptionalInt("onesided", 1) == 1;
                var signal = GetInput(0);
                var frameStep = GetInput(1);
                var window = GetInput(2);
                var frameLength = GetInput(3);
                SetOutput(gm.STFT(signal, frameStep, window, frameLength, windowedDFTMatrix: null, onesided));
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
                SetOutput(gm.Cast(GetInput(0), toDataType));
            }
            else if (opType == "CastLike")
            {
                SetOutput(gm.CastLike(GetInput(0), GetInput(1)));
            }
            else if (opType == "Concat")
            {
                var axis = node.GetRequiredInt("axis");
                SetOutput(gm.Concat(GetInputs(), axis));
            }
            else if (opType == "DepthToSpace")
            {
                var modeType = node.GetOptionalString("mode", "DCR");
                var mode = modeType == "DCR" ? Layers.DepthToSpaceMode.DepthColumnRow : Layers.DepthToSpaceMode.ColumnRowDepth;
                var blocksize = node.GetRequiredInt("blocksize");
                SetOutput(gm.DepthToSpace(GetInput(0), blocksize, mode));
            }
            else if (opType == "Expand")
            {
                // Expand-8, Expand-13
                SetOutput(gm.Expand(GetInput(0), GetInput(1)));
            }
            else if (opType == "Flatten")
            {
                var axis = node.GetOptionalInt("axis", 1);
                SetOutput(gm.Flatten(GetInput(0), axis));
            }
            else if (opType == "GridSample")
            {
                var modeString = node.GetOptionalString("mode", "linear");
                var mode = modeString switch
                {
                    "bilinear" => Layers.InterpolationMode.Linear, // for opset 16
                    "linear" => Layers.InterpolationMode.Linear,
                    "bicubic" => Layers.InterpolationMode.Cubic, // for opset 16
                    "cubic" => Layers.InterpolationMode.Cubic,
                    "nearest" => Layers.InterpolationMode.Nearest,
                    _ => Warn(WarningType.Warning, $"mode `{modeString}` is not supported for GridSample, using `linear`.", Layers.InterpolationMode.Linear)
                };
                var paddingModeString = node.GetOptionalString("padding_mode", "zeros");
                var paddingMode = paddingModeString switch
                {
                    "zeros" => Layers.PaddingMode.Zeros,
                    "border" => Layers.PaddingMode.Border,
                    "reflection" => Layers.PaddingMode.Reflection,
                    _ => Warn(WarningType.Warning, $"padding_mode `{paddingModeString}` is not supported for GridSample, using `zeros`.", Layers.PaddingMode.Zeros)
                };
                var alignCorners = node.GetOptionalInt("align_corners", 0) == 1;
                SetOutput(gm.GridSample(GetInput(0), GetInput(1), mode, paddingMode, alignCorners));
            }
            else if (opType == "Dropout")
            {
                SetOutput(gm.Identity(GetInput(0)));
            }
            else if (opType == "Identity")
            {
                SetOutput(gm.Identity(GetInput(0)));
            }
            else if (opType == "Pad")
            {
                var modeString = node.GetOptionalString("mode", "constant");
                var mode = modeString switch
                {
                    "constant" => Layers.PadMode.Constant,
                    "reflect" => Layers.PadMode.Reflect,
                    "edge" => Layers.PadMode.Edge,
                    "wrap" => Layers.PadMode.Wrap,
                    _ => Warn(WarningType.Warning, $"mode `{modeString}` is not supported for Pad, using `constant`.", Layers.PadMode.Constant)
                };

                if (defaultOpsetVersion < 11)
                {
                    // Pad-1 or Pad-2
                    var padsArray = node.GetRequiredIntArray(node.HasAttribute("pads") ? "pads" : "paddings");
                    var pads = gm.Constant(padsArray);

                    var valueFloat = node.GetOptionalFloat("value", 0f);
                    var value = gm.Constant(valueFloat);

                    SetOutput(gm.Pad(GetInput(0), pads, value, null, mode));
                }
                else
                {
                    // Pad-11, Pad-13, Pad-18
                    SetOutput(gm.Pad(GetInput(0), GetInput(1), GetInput(2), GetInput(3), mode));
                }
            }
            else if (opType == "Reshape")
            {
                if (defaultOpsetVersion < 5)
                {
                    // Reshape-1
                    var shapeArray = node.GetRequiredIntArray("shape");
                    var shape = gm.Constant(shapeArray);
                    SetOutput(gm.Reshape(GetInput(0), shape, false));
                }
                else
                {
                    // Reshape-5, Reshape-13, Reshape-14
                    var allowZero = node.GetOptionalInt("allowzero", 0) != 0;
                    SetOutput(gm.Reshape(GetInput(0), GetInput(1), allowZero));
                }
            }
            else if (opType == "Resize")
            {
                var modeString = node.GetOptionalString("mode", "nearest");
                var mode = modeString switch
                {
                    "nearest" => Layers.InterpolationMode.Nearest,
                    "linear" => Layers.InterpolationMode.Linear,
                    _ => Warn(WarningType.Warning, $"mode `{modeString}` is not supported for Resize, using `nearest`.", Layers.InterpolationMode.Nearest)
                };

                var axes = node.GetOptionalIntArray("axes", null);
                if (defaultOpsetVersion < 11)
                {
                    // Resize-10
                    SetOutput(gm.Resize(GetInput(0), GetInput(1), Layers.ScaleMode.Scales, Layers.CoordTransformMode.Asymmetric, mode, Layers.NearestMode.Floor, axes));
                }
                else
                {
                    node.UnsupportedAttribute("cubic_coeff_a", -0.75f);
                    node.UnsupportedAttribute("exclude_outside", 0);
                    node.UnsupportedAttribute("extrapolation_value", 0);
                    var coordinateTransformModeString = node.GetOptionalString("coordinate_transformation_mode", "half_pixel");
                    var coordinateTransformMode = coordinateTransformModeString switch
                    {
                        "half_pixel" => Layers.CoordTransformMode.HalfPixel,
                        "pytorch_half_pixel" => Layers.CoordTransformMode.PytorchHalfPixel,
                        "align_corners" => Layers.CoordTransformMode.AlignCorners,
                        "asymmetric" => Layers.CoordTransformMode.Asymmetric,
                        _ => Warn(WarningType.Warning, $"coordinate_transformation_mode `{coordinateTransformModeString}` is not supported for Resize, using `half_pixel`.", Layers.CoordTransformMode.HalfPixel)
                    };

                    var nearestModeString = node.GetOptionalString("nearest_mode", "round_prefer_floor");
                    var nearestMode = nearestModeString switch
                    {
                        "round_prefer_floor" => Layers.NearestMode.RoundPreferFloor,
                        "round_prefer_ceil" => Layers.NearestMode.RoundPreferCeil,
                        "floor" => Layers.NearestMode.Floor,
                        "ceil" => Layers.NearestMode.Ceil,
                        _ => Warn(WarningType.Warning, $"nearest_mode `{nearestModeString}` is not supported for Resize, using `round_prefer_floor`.", Layers.NearestMode.RoundPreferFloor)
                    };

                    if (node.InputCount == 3 || string.IsNullOrEmpty(node.Inputs[3]))
                    {
                        // Resize-11, Resize-13, Resize-18 with scales
                        SetOutput(gm.Resize(GetInput(0), GetInput(2), Layers.ScaleMode.Scales, coordinateTransformMode, mode, nearestMode, axes));
                    }
                    else if (node.InputCount == 4)
                    {
                        // Resize-11, Resize-13, Resize-18 with sizes
                        SetOutput(gm.Resize(GetInput(0), GetInput(3), Layers.ScaleMode.Sizes, coordinateTransformMode, mode, nearestMode, axes));
                    }
                }
            }
            else if (opType == "Slice")
            {
                if (defaultOpsetVersion < 10)
                {
                    // Slice-1
                    var startsArray = node.GetRequiredIntArray("starts");
                    var starts = gm.Constant(startsArray);

                    var endsArray = node.GetRequiredIntArray("ends");
                    var ends = gm.Constant(endsArray);

                    if (node.HasAttribute("axes"))
                    {
                        var axesArray = node.GetRequiredIntArray("axes");
                        var axes = gm.Constant(axesArray);
                        SetOutput(gm.Slice(GetInput(0), starts, ends, axes, null));
                    }
                    else
                    {
                        SetOutput(gm.Slice(GetInput(0), starts, ends, null, null));
                    }
                }
                else
                {
                    // Slice-10, Slice-11, Slice-13
                    SetOutput(gm.Slice(GetInput(0), GetInput(1), GetInput(2), GetInput(3), GetInput(4)));
                }
            }
            else if (opType == "SpaceToDepth")
            {
                var blocksize = node.GetRequiredInt("blocksize");
                SetOutput(gm.SpaceToDepth(GetInput(0), blocksize));
            }
            else if (opType == "Split")
            {
                var axis = node.GetOptionalInt("axis", 0);
                if (node.HasAttribute("split"))
                {
                    // Split-1, Split-2, Split-11 with "split" attribute
                    var splitArray = node.GetRequiredIntArray("split");
                    var split = gm.Constant(splitArray);
                    SetOutputs(gm.Split(GetInput(0), split, axis, node.OutputCount));
                }
                else
                {
                    var split = GetInput(1);

                    if (split is null)
                    {
                        // Split-1, Split-2, Split-11, Split-13, Split-18 with num_outputs
                        var numOutputs = node.GetOptionalInt("num_outputs", node.Outputs.Length);
                        SetOutputs(gm.Split(GetInput(0), null, axis, numOutputs));
                    }
                    else
                    {
                        // Split-1, Split-2, Split-11, Split-13, Split-18 with split tensor
                        SetOutputs(gm.Split(GetInput(0), split, axis, node.OutputCount));
                    }
                }
            }
            else if (opType == "Squeeze")
            {
                if (defaultOpsetVersion < 13 && node.HasAttribute("axes"))
                {
                    // Squeeze-1, Squeeze-11 with given axes
                    var axesArray = node.GetRequiredIntArray("axes");
                    var axes = gm.Constant(axesArray);

                    SetOutput(gm.Squeeze(GetInput(0), axes));
                }
                else
                {
                    // Squeeze-13 or Squeeze-1, Squeeze-11 without given axes
                    SetOutput(gm.Squeeze(GetInput(0), GetInput(1)));
                }
            }
            else if (opType == "Tile")
            {
                SetOutput(gm.Tile(GetInput(0), GetInput(1)));
            }
            else if (opType == "Transpose")
            {
                var permutations = node.GetOptionalIntArray("perm", null);
                SetOutput(gm.Transpose(GetInput(0), permutations));
            }
            else if (opType == "Trilu")
            {
                var upper = node.GetOptionalInt("upper", 1);
                SetOutput(gm.Trilu(GetInput(0), GetInput(1), (Layers.TriluMode)upper));
            }
            else if (opType == "Upsample")
            {
                var coordinateTransformMode = Layers.CoordTransformMode.Asymmetric;
                var modeString = node.GetOptionalString("mode", "nearest");
                var mode = modeString switch
                {
                    "nearest" => Layers.InterpolationMode.Nearest,
                    "linear" => Layers.InterpolationMode.Linear,
                    "bilinear" => Layers.InterpolationMode.Linear, // for opset 1
                    _ => Warn(WarningType.Warning, $"mode `{modeString}` is not supported for Resize, using `nearest`.", Layers.InterpolationMode.Nearest)
                };
                var nearestMode = Layers.NearestMode.Floor;
                if (defaultOpsetVersion < 9)
                {
                    // Upsample-7
                    var scalesArray = node.GetRequiredFloatArray("scales");
                    var scales = gm.Constant(scalesArray);

                    SetOutput(gm.Resize(GetInput(0), scales, Layers.ScaleMode.Scales, coordinateTransformMode, mode, nearestMode, null));
                }
                else
                {
                    // Upsample-9
                    SetOutput(gm.Resize(GetInput(0), GetInput(1), Layers.ScaleMode.Scales, coordinateTransformMode, mode, nearestMode, null));
                }
            }
            else if (opType == "Unsqueeze")
            {
                if (defaultOpsetVersion < 13)
                {
                    // Unsqueeze-1, Unsqueeze-11
                    var axesArray = node.GetRequiredIntArray("axes");
                    var axes = gm.Constant(axesArray);

                    SetOutput(gm.Unsqueeze(GetInput(0), axes));
                }
                else
                {
                    // Unsqueeze-13
                    SetOutput(gm.Unsqueeze(GetInput(0), GetInput(1)));
                }
            }
            // Layer.Trigonometric
            else if (opType == "Acos")
            {
                SetOutput(gm.Acos(GetInput(0)));
            }
            else if (opType == "Acosh")
            {
                SetOutput(gm.Acosh(GetInput(0)));
            }
            else if (opType == "Asin")
            {
                SetOutput(gm.Asin(GetInput(0)));
            }
            else if (opType == "Asinh")
            {
                SetOutput(gm.Asinh(GetInput(0)));
            }
            else if (opType == "Atan")
            {
                SetOutput(gm.Atan(GetInput(0)));
            }
            else if (opType == "Atanh")
            {
                SetOutput(gm.Atanh(GetInput(0)));
            }
            else if (opType == "Cos")
            {
                SetOutput(gm.Cos(GetInput(0)));
            }
            else if (opType == "Cosh")
            {
                SetOutput(gm.Cosh(GetInput(0)));
            }
            else if (opType == "Sin")
            {
                SetOutput(gm.Sin(GetInput(0)));
            }
            else if (opType == "Sinh")
            {
                SetOutput(gm.Sinh(GetInput(0)));
            }
            else if (opType == "Tan")
            {
                SetOutput(gm.Tan(GetInput(0)));
            }
            // Non standard ONNX
            else if (opType == "Swish")
            {
                SetOutput(gm.Swish(GetInput(0)));
            }
            else if (opType == "ImageScaler")
            {
                var attrBias = node.GetRequiredFloatArray("bias");
                var maxElements = attrBias.Length;
                var attrScale = Enumerable.Repeat(node.GetOptionalFloat("scale", 1.0f), maxElements).ToArray();

                var scale = gm.Constant(attrScale);
                var bias = gm.Constant(attrBias);
                SetOutput(gm.ScaleBias(GetInput(0), scale, bias));
            }
            else
            {
                Warn(WarningType.Error, $"Unsupported ONNX Operator: {opType}");
                Debug.LogError(Warnings.Last().Message);
            }
        }

        Model ConvertOnnxModel(ModelProto onnxModel)
        {
            var gm = new GraphModule();
            var tensors = new Dictionary<string, Node>();

            long defaultOpsetVersion = 15;

            // Parse producer meta data
            foreach (var opsetSetIdProto in onnxModel.OpsetImport)
            {
                if (string.IsNullOrEmpty(opsetSetIdProto.Domain))
                    defaultOpsetVersion = opsetSetIdProto.Version;
            }

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

                var inputNode = gm.Input(input.Name, inputDataType, inputShape);
                tensors[input.Name] = inputNode;
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
                    var constantTensor = ONNXConstantsLoader.LoadConstant(initializer, stream);
                    tensors[initializer.Name] = gm.Constant(constantTensor);
                }
                else
                {
                    var constantTensor = ONNXConstantsLoader.LoadConstant(initializer);
                    tensors[initializer.Name] = gm.Constant(constantTensor);
                }
            }
            foreach (var stream in weightsStream.Values)
                stream.Dispose();

            // NOTE: It's questionable whether we should be doing this since the ONNX specification requires the graph to be
            // topologically sorted, but at least one network encountered that was exported from keras2onnx v1.7.0 produced
            // an incorrectly sorted graph. related example: https://github.com/onnx/keras-onnx/issues/184
            var sortedGraph = ONNXModelUtility.StableTopologicalSort(onnxModel.Graph);

            // Convert graph nodes
            foreach (NodeProto onnxNode in sortedGraph)
            {
                var node = new ONNXNodeWrapper(onnxNode);

                try
                {
                    OnNode(gm, tensors, defaultOpsetVersion, node);
                }
                catch (Exception e)
                {
                    Warn(WarningType.Error, e.Message);
                    throw new OnnxImportException(Warnings.Last().Message);
                }
            }

            // delete unused outputs
            var outputs = new List<Node>();
            var outputNames = new List<string>();
            for (var i = 0; i < onnxModel.Graph.Output.Count; i++)
            {
                var outputName = onnxModel.Graph.Output[i].Name;
                if (!tensors.TryGetValue(outputName, out var outputTensor))
                {
                    Warn(WarningType.Warning, $"Output {outputName} is not connected to any tensor in the graph and will be skipped.");
                    continue;
                }
                outputs.Add(outputTensor);
                outputNames.Add(outputName);
            }

            gm.Outputs(outputNames.ToArray(), outputs.ToArray());

            if (!Warnings.Any(w => w.MessageSeverity == WarningType.Error))
            {
                ModelOptimizer.OptimizeGraph(gm);
            }

            var model = GraphConverter.GraphToModel(gm);

            model.ProducerName = onnxModel.ProducerName;
            if (!string.IsNullOrEmpty(onnxModel.ProducerVersion))
                model.ProducerName += $" v{onnxModel.ProducerVersion}";

            // add symbolic names to model
            model.symbolicDimNames = namedDims.ToArray();

            // validate imported model
            if (!Warnings.Any(w => w.MessageSeverity == WarningType.Error))
            {
                model = ModelValidator.ValidateModel(model);
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
    }

    /// <summary>
    /// Represents an exception during the import of an ONNX model.
    /// </summary>
    class OnnxImportException : ImportException
    {
        /// <inheritdoc cref="ImportException"/>
        public OnnxImportException(string message) : base(message) { }
    }

    /// <summary>
    /// Represents an exception during the import of a ONNX layer.
    /// </summary>
    class OnnxLayerImportException : LayerImportException
    {
        /// <inheritdoc cref="LayerImportException"/>
        public OnnxLayerImportException(string message) : base(message) { }
    }
}
