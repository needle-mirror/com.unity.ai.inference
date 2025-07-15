// #define DEBUG_TIMING
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using SentisFlatBuffer;
using Unity.InferenceEngine.Google.FlatBuffers;
using Unity.InferenceEngine.Layers;

[assembly: InternalsVisibleTo("Unity.InferenceEngine.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.InferenceEngine.EditorTests")]

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Provides methods for loading models.
    /// </summary>
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.Sentis")]
    public static class ModelLoader
    {
        /// <summary>
        /// Converts a binary `ModelAsset` representation of a neural network to an object-oriented `Model` representation.
        /// </summary>
        /// <param name="modelAsset">The binary `ModelAsset` model</param>
        /// <returns>The loaded `Model`</returns>
        public static Model Load(ModelAsset modelAsset)
        {
            var modelDescriptionBytes = modelAsset.modelAssetData.value;
            var modelWeightsBytes = new byte[modelAsset.modelWeightsChunks.Length][];
            for (var i = 0; i < modelAsset.modelWeightsChunks.Length; i++)
                modelWeightsBytes[i] = modelAsset.modelWeightsChunks[i].value;
            return LoadModel(modelDescriptionBytes, modelWeightsBytes);
        }

        /// <summary>
        /// Loads a model that has been serialized to disk.
        /// </summary>
        /// <param name="path">The path of the binary serialized model</param>
        /// <returns>The loaded `Model`</returns>
        public static Model Load(string path)
        {
            using var fileStream = File.OpenRead(path);
            return Load(fileStream);
        }

        /// <summary>
        /// Loads a model that has been serialized to a stream.
        /// </summary>
        /// <param name="stream">The stream to load the serialized model from.</param>
        /// <returns>The loaded `Model`.</returns>
        public static Model Load(Stream stream)
        {
            try
            {
                var model = new Model();

                var prefixSizeBytes = new byte[sizeof(int)];
                stream.Read(prefixSizeBytes);
                var modelDescriptionSize = BitConverter.ToInt32(prefixSizeBytes);
                var modelDescriptionBytes = new byte[modelDescriptionSize + sizeof(int)];
                System.Buffer.BlockCopy(prefixSizeBytes, 0, modelDescriptionBytes, 0, sizeof(int));
                stream.Read(modelDescriptionBytes, sizeof(int), modelDescriptionSize);
                var weightBuffersConstantsOffsets = LoadModelDescription(modelDescriptionBytes, ref model);

                for (var i = 0; i < weightBuffersConstantsOffsets.Length; i++)
                {
                    stream.Read(prefixSizeBytes);
                    var modelWeightsChunkSize = BitConverter.ToInt32(prefixSizeBytes);
                    var modelWeightsBufferBytes = new byte[modelWeightsChunkSize + sizeof(int)];
                    System.Buffer.BlockCopy(prefixSizeBytes, 0, modelWeightsBufferBytes, 0, sizeof(int));
                    stream.Read(modelWeightsBufferBytes, sizeof(int), modelWeightsChunkSize);
                    LoadModelWeights(modelWeightsBufferBytes, weightBuffersConstantsOffsets[i], ref model);
                }

                return model;
            }
            catch (Exception e)
            {
                D.LogError($"Failed to load serialized .sentis model, ensure model was exported with Inference Engine or with Sentis 1.4 or newer. ({e.Message})");
                return null;
            }
        }

        internal static int OptionalInput(this Chain chain, int index)
        {
            var input = index < chain.InputsLength ? chain.Inputs(index) : -1;
            return input;
        }

        internal static int RequiredInput(this Chain chain, int index)
        {
            Logger.AssertIsTrue(index < chain.InputsLength, "Input Required");
            var input = chain.Inputs(index);
            Logger.AssertIsTrue(input != -1, "Input Required");
            return input;
        }

        internal static int OptionalOutput(this Chain chain, int index)
        {
            var output = index < chain.OutputsLength ? chain.Outputs(index) : -1;
            return output;
        }

        internal static int RequiredOutput(this Chain chain, int index)
        {
            Logger.AssertIsTrue(index < chain.OutputsLength, "Output Required");
            var output = chain.Outputs(index);
            Logger.AssertIsTrue(output != -1, "Output Required");
            return output;
        }

        internal static Model LoadModelDescription(byte[] modelDescription)
        {
            var model = new Model();
            LoadModelDescription(modelDescription, ref model);
            return model;
        }

        internal static Model LoadModel(byte[] modelDescription, byte[][] modelWeights)
        {
            var model = new Model();
            var weightsConstantIndexesOffsets = LoadModelDescription(modelDescription, ref model);
            for (var i = 0; i < weightsConstantIndexesOffsets.Length; i++)
                LoadModelWeights(modelWeights[i], weightsConstantIndexesOffsets[i], ref model);
            return model;
        }

        static DynamicTensorShape GetDynamicShape(SentisFlatBuffer.Tensor tensorDesc)
        {
            if (tensorDesc.ShapeDynamism == TensorShapeDynamism.STATIC)
                return new DynamicTensorShape(new TensorShape(tensorDesc.GetFixedSizesArray()));
            if (tensorDesc.HasDynamicRank)
                return DynamicTensorShape.DynamicRank;

            var shape = DynamicTensorShape.DynamicOfRank(tensorDesc.DynamicSizesLength);
            for (var i = 0; i < shape.rank; i++)
            {
                var d = tensorDesc.DynamicSizes(i).Value;
                shape[i] = d.ValType switch
                {
                    SymbolicDim.NONE => DynamicTensorDim.Unknown,
                    SymbolicDim.Int => DynamicTensorDim.Int(d.ValAsInt().IntVal),
                    SymbolicDim.Byte => DynamicTensorDim.Param(d.ValAsByte().ByteVal),
                    _ => throw new ArgumentOutOfRangeException()
                };
            }

            return shape;
        }

        static List<(int, int)>[] LoadModelDescription(byte[] modelDescription, ref Model model)
        {
            var bb = new ByteBuffer(modelDescription, sizeof(int));
            var program = Program.GetRootAsProgram(bb);
            var numWeightsBuffers = program.SegmentsLength;
            var weightBuffersConstantsOffsets = new List<(int, int)>[numWeightsBuffers];
            for (var i = 0; i < numWeightsBuffers; i++)
                weightBuffersConstantsOffsets[i] = new List<(int, int)>();

            try
            {
                var originalProgramVersion = program.Version;
                if (originalProgramVersion < ModelWriter.version)
                    program = ModelUpgrader.Upgrade(program);
                if (originalProgramVersion > ModelWriter.version)
                    D.LogWarning("Serialized model was exported in a newer version of Inference Engine than the current installed version and may not work as expected. Update the Inference Engine package to ensure compatibility.");
                var executionPlan = program.ExecutionPlan.Value;

                model.symbolicDimNames = new string[executionPlan.SymbolicDimNamesLength];
                for (var i = 0; i < model.symbolicDimNames.Length; i++)
                    model.symbolicDimNames[i] = executionPlan.SymbolicDimNames(i);

                int inputCount = executionPlan.InputsLength;
                for (int i = 0; i < inputCount; i++)
                {
                    int input = executionPlan.Inputs(i);
                    var tensorDesc = executionPlan.Values(input).Value.ValAsTensor();
                    model.AddInput(executionPlan.InputsName(i), input, (DataType)tensorDesc.ScalarType, GetDynamicShape(tensorDesc));
                }
                model.outputs = new List<Model.Output>();
                for (int i = 0; i < executionPlan.OutputsLength; i++)
                {
                    model.AddOutput(executionPlan.OutputsName(i), executionPlan.Outputs(i));
                }

                // load known data types and shapes
                for (var i = 0; i < executionPlan.ValuesLength; i++)
                {
                    var value = executionPlan.Values(i).Value;
                    if (value.ValType != KernelTypes.Tensor)
                        continue;
                    var tensorDesc = executionPlan.Values(i).Value.ValAsTensor();
                    // defaults for tensor desc mean no data type or shape provided
                    if (tensorDesc.ShapeDynamism == TensorShapeDynamism.STATIC && tensorDesc.GetFixedSizesArray() == null)
                        continue;
                    model.dataTypes[i] = (DataType)tensorDesc.ScalarType;
                    model.shapes[i] = GetDynamicShape(tensorDesc);
                }

                HashSet<int> constants = new HashSet<int>();
                for (int i = 0; i < executionPlan.ChainsLength; i++)
                {
                    var chain = executionPlan.Chains(i).Value;

                    for (int k = 0; k < chain.InputsLength; k++)
                    {
                        var input = chain.Inputs(k);
                        if (input == -1)
                            continue;
                        if (constants.Contains(input))
                            continue;
                        var constantTensor = executionPlan.Values(input).Value.ValAsTensor();
                        if (constantTensor.ConstantBufferIdx == 0)
                            continue;
                        int lengthByte = constantTensor.LengthByte;
                        model.AddConstant(new Constant(input, new TensorShape(constantTensor.GetFixedSizesArray()), (DataType)constantTensor.ScalarType, lengthByte));
                        var idx = (int)(constantTensor.ConstantBufferIdx - 1);
                        var offset = constantTensor.StorageOffset;
                        weightBuffersConstantsOffsets[idx].Add((model.constants.Count - 1, offset));
                        constants.Add(input);
                    }

                    if (chain.Instructions(0).Value.InstrArgsType == InstructionArguments.NONE)
                        continue;

                    var kernel = chain.Instructions(0).Value.InstrArgsAsKernelCall();
                    string kernelName = executionPlan.Operators(kernel.OpIndex).Value.Name;
                    var layer = LayerModelLoader.DeserializeLayer(kernelName, chain, executionPlan);

                    if (layer == null)
                        throw new NotImplementedException(kernelName);

                    model.AddLayer(layer);
                }
            }
            catch (Exception e)
            {
                D.LogError($"Failed to load serialized model description. ({e.Message})");
                throw;
            }

            return weightBuffersConstantsOffsets;
        }

        static void LoadModelWeights(byte[] modelWeightsBufferBytes, List<(int, int)> constantIndexesOffsets, ref Model model)
        {
            try
            {
                var bb = new ByteBuffer(modelWeightsBufferBytes, sizeof(int));
                var weightBuffer = SentisFlatBuffer.Buffer.GetRootAsBuffer(bb);
                var data = weightBuffer.GetStorageArray();
                foreach (var (constantIdx, offset) in constantIndexesOffsets)
                {
                    var constant = model.constants[constantIdx];
                    if (constant.lengthBytes == 0)
                        continue;
                    var elementCount = constant.lengthBytes / NativeTensorArray.k_DataItemSize;
                    constant.m_Weights = new NativeTensorArrayFromManagedArray(data, offset, elementCount);
                }
            }
            catch (InvalidOperationException)
            {
                D.LogError("Failed to load serialized model weights.");
                throw;
            }
        }
    }
}
