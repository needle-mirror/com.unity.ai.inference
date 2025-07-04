using System;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine.Compiler.Analyser;
using Unity.InferenceEngine.Layers;

namespace Unity.InferenceEngine.Compiler.Passes.Optimization
{
    class EinsumToMatMulPass : IModelPass
    {
        static int _isOperandA = 1 << 0;
        static int _isOperandB = 1 << 1;
        static int _isOutput = 1 << 2;

        static void CalculatePermutations(TensorIndex originalDims, TensorIndex[] transformedDims,
            out int[] permutation, out int[] inversePermutation, out bool isPermuted)
        {
            isPermuted = false;
            permutation = new int[originalDims.rank];
            inversePermutation = new int[originalDims.rank];
            var index = 0;
            foreach (var dims in transformedDims)
            {
                for (var i = 0; i < dims.rank; i++)
                {
                    for (var j = 0; j < originalDims.rank; j++)
                    {
                        if (originalDims[j] == dims[i])
                        {
                            permutation[j] = index;
                            inversePermutation[index] = j;
                            if (permutation[j] != j)
                                isPermuted = true;
                            index++;
                        }
                    }
                }
            }
        }

        static bool InsertPermuteReshapeLayers(Model model, Einsum einsumLayer, int inputLayerIndex, TensorIndex originalDims, TensorIndex[] transformedDims, DynamicTensorShape originalShape, ref int uniqueIndex)
        {
            CalculatePermutations(originalDims, transformedDims, out _, out var inversePermutation, out var isPermuted);

            var shape = DynamicTensorShape.Ones(transformedDims.Length);
            var index = 0;
            for (var i = 0; i < transformedDims.Length; i++)
            {
                for (var j = 0; j < transformedDims[i].rank; j++)
                {
                    shape[i] *= originalShape[inversePermutation[index++]];
                }
            }

            var isReshaped = false;
            foreach (var t in transformedDims)
            {
                if (t.rank != 1)
                    isReshaped = true;
            }

            if (isReshaped && !shape.IsStatic())
                return false;

            var inputIndex = einsumLayer.inputs[inputLayerIndex];
            var insertLayerIndex = model.layers.IndexOf(einsumLayer);

            if (isPermuted)
            {
                var inputTransposeIndex = uniqueIndex++;
                var transposeLayer = new Transpose(inputTransposeIndex, inputIndex, inversePermutation);
                inputIndex = inputTransposeIndex;
                model.layers.Insert(insertLayerIndex, transposeLayer);
                insertLayerIndex++;
            }

            if (isReshaped)
            {
                var shapeConstant = new Constant(uniqueIndex++, new TensorShape(shape.rank), shape.ToTensorShape().ToArray());
                model.AddConstant(shapeConstant);
                var inputReshapeIndex = uniqueIndex++;
                var reshapeLayer = new Reshape(inputReshapeIndex, inputIndex, shapeConstant.index);
                inputIndex = inputReshapeIndex;
                model.layers.Insert(insertLayerIndex, reshapeLayer);
            }

            einsumLayer.inputs[inputLayerIndex] = inputIndex;
            return true;
        }

        static bool InsertInversePermuteReshapeLayers(Model model, Einsum einsumLayer, TensorIndex originalDims, TensorIndex[] transformedDims, DynamicTensorShape originalShape, ref int uniqueIndex)
        {
            CalculatePermutations(originalDims, transformedDims, out var permutation, out var inversePermutation, out var isPermuted);

            var shape = DynamicTensorShape.DynamicOfRank(originalDims.rank);
            for (var i = 0; i < originalDims.rank; i++)
            {
                shape[i] = originalShape[permutation[i]];
            }

            var isReshaped = false;
            foreach (var t in transformedDims)
            {
                if (t.rank != 1)
                    isReshaped = true;
            }

            if (isReshaped && !shape.IsStatic())
                return false;

            var insertLayerIndex = model.layers.IndexOf(einsumLayer) + 1;
            var outputIndex = einsumLayer.outputs[0];

            if (isPermuted)
            {
                var outputTransposeIndex = uniqueIndex++;
                var transposeLayer = new Transpose(outputIndex, outputTransposeIndex, inversePermutation);
                outputIndex = outputTransposeIndex;
                model.layers.Insert(insertLayerIndex, transposeLayer);
            }

            if (isReshaped)
            {
                var constantIndex = uniqueIndex++;
                var shapeConstant = new Constant(constantIndex, new TensorShape(shape.rank), shape.ToTensorShape().ToArray());
                model.AddConstant(shapeConstant);
                var outputReshapeIndex = uniqueIndex++;
                var reshapeLayer = new Reshape(outputIndex, outputReshapeIndex, shapeConstant.index);
                outputIndex = outputReshapeIndex;
                model.layers.Insert(insertLayerIndex, reshapeLayer);
            }

            einsumLayer.outputs = new[] { outputIndex };
            return true;
        }

        public void Run(ref Model model)
        {
            var einsumLayers = model.layers.Where(l => l is Einsum).ToList();
            if (einsumLayers.Count == 0)
                return;

            var ctx = PartialInferenceAnalysis.InferModelPartialTensors(model);

            foreach (var layer in einsumLayers)
            {
                var einsumLayer = (Einsum)layer;
                var numOperands = einsumLayer.inputs.Length;
                if (numOperands != 2)
                {
                    // only einsums with two operands can currently be reduced to matmul
                    continue;
                }

                var isInputShapes = true;
                var operandShapes = new DynamicTensorShape[numOperands];
                for (var i = 0; i < numOperands && isInputShapes; i++)
                {
                    var shape = ctx.GetPartialTensor(layer.inputs[i]).shape;
                    if (shape.hasRank)
                        operandShapes[i] = shape;
                    else
                        isInputShapes = false;
                }

                if (!isInputShapes)
                {
                    // input shapes not found so can't parse equation
                    continue;
                }

                var equation = einsumLayer.equation;

                var operandIndices = new TensorIndex[numOperands];
                EinsumHelper.ParseEquationStringShape(equation, operandShapes, ref operandIndices, out var outputIndices, out var numIndices);

                var isUnsupportedConversion = false;

                var operandPositions = new List<int>[numOperands];

                for (var i = 0; i < numOperands; i++)
                {
                    operandPositions[i] = new List<int>();
                    for (var j = 0; j < numIndices; j++)
                    {
                        operandPositions[i].Add(-1);
                    }
                    for (var j = 0; j < operandIndices[i].rank; j++)
                    {
                        if (operandPositions[i][operandIndices[i][j]] >= 0)
                        {
                            // label repeats in operand, no optimization supported
                            isUnsupportedConversion = true;
                            break;
                        }

                        operandPositions[i][operandIndices[i][j]] = j;
                    }
                }

                var outputPositions = new List<int>();
                for (var j = 0; j < numIndices; j++)
                {
                    outputPositions.Add(-1);
                }
                for (var j = 0; j < outputIndices.rank; j++)
                {
                    outputPositions[outputIndices[j]] = j;
                }

                if (isUnsupportedConversion)
                    continue;

                var dimClassification = new int[numIndices];

                // classify einsum dimensions depending on type using flags
                for (var i = 0; i < numIndices; i++)
                {
                    dimClassification[i] += operandPositions[0][i] >= 0 ? _isOperandA : 0;
                    dimClassification[i] += operandPositions[1][i] >= 0 ? _isOperandB : 0;
                    dimClassification[i] += outputPositions[i] >= 0 ? _isOutput : 0;
                    if (dimClassification[i] == _isOperandA || dimClassification[i] == _isOperandB)
                    {
                        // label only appears in one operand, not a matmul
                        isUnsupportedConversion = true;
                        break;
                    }
                }

                if (isUnsupportedConversion)
                    continue;

                // categorize dims into sum, broadcast and operandOut
                var broadcastDims = Enumerable.Range(0, numIndices).Where(i => dimClassification[i] == _isOperandA + _isOperandB + _isOutput).ToList();
                var sumDims = Enumerable.Range(0, numIndices).Where(i => dimClassification[i] == _isOperandA + _isOperandB).ToList();
                var outputOperandDimsA = Enumerable.Range(0, numIndices).Where(i => dimClassification[i] == _isOperandA + _isOutput).ToList();
                var outputOperandDimsB = Enumerable.Range(0, numIndices).Where(i => dimClassification[i] == _isOperandB + _isOutput).ToList();

                // reorder dimensions within classifications to match given
                // inputs and outputs as closely as possible
                broadcastDims.Sort((a, b) => outputPositions[a].CompareTo(outputPositions[b]));
                sumDims.Sort((a, b) => operandPositions[0][a].CompareTo(operandPositions[0][b]));
                outputOperandDimsA.Sort((a, b) => outputPositions[a].CompareTo(outputPositions[b]));
                outputOperandDimsB.Sort((a, b) => outputPositions[a].CompareTo(outputPositions[b]));

                var broadcastDimsIndex = new TensorIndex(broadcastDims.ToArray());
                var sumDimsIndex = new TensorIndex(sumDims.ToArray());
                var outputOperandDimsAIndex = new TensorIndex(outputOperandDimsA.ToArray());
                var outputOperandDimsBIndex = new TensorIndex(outputOperandDimsB.ToArray());

                var transformedDims0 = broadcastDimsIndex.rank > 0 ? new[] { broadcastDimsIndex, outputOperandDimsAIndex, sumDimsIndex } : new[] { outputOperandDimsAIndex, sumDimsIndex };
                var transformedDims1 = broadcastDimsIndex.rank > 0 ? new[] { broadcastDimsIndex, sumDimsIndex, outputOperandDimsBIndex } : new[] { sumDimsIndex, outputOperandDimsBIndex };
                var transformedDimsOut = broadcastDimsIndex.rank > 0 ? new[] { broadcastDimsIndex, outputOperandDimsAIndex, outputOperandDimsBIndex } : new[] { outputOperandDimsAIndex, outputOperandDimsBIndex };

                var uniqueIndex = model.GetUniqueIndex();

                // insert permute and reshape layers to take inputs to desired matmul inputs
                if (!InsertPermuteReshapeLayers(model, einsumLayer, 0, operandIndices[0], transformedDims0, ctx.GetPartialTensor(einsumLayer.inputs[0]).shape, ref uniqueIndex))
                    break;
                if (!InsertPermuteReshapeLayers(model, einsumLayer, 1, operandIndices[1], transformedDims1, ctx.GetPartialTensor(einsumLayer.inputs[1]).shape, ref uniqueIndex))
                    break;

                // insert reshape and permute layers to get desired output from matmul output
                if (!InsertInversePermuteReshapeLayers(model, einsumLayer, outputIndices, transformedDimsOut, ctx.GetPartialTensor(einsumLayer.outputs[0]).shape, ref uniqueIndex))
                    break;

                model.layers[model.layers.IndexOf(einsumLayer)] = new MatMul(einsumLayer.outputs[0], einsumLayer.inputs[0], einsumLayer.inputs[1]);
            }
        }
    }
}
