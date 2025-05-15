using System;
using System.Linq;
using Unity.InferenceEngine.Compiler.Analyser;
using Unity.InferenceEngine.Layers;
using UnityEngine;

namespace Unity.InferenceEngine.Compiler.Passes.Optimization
{
    class SimplifyReshapeInputPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var reshapeLayers = model.layers.Where(l => l is Reshape).ToList();
            if (reshapeLayers.Count == 0)
                return;

            var ctx = PartialInferenceAnalysis.InferModelPartialTensors(model);

            foreach (var layer in reshapeLayers)
            {
                var reshapeLayer = (Reshape)layer;
                var shapePartialTensor = ctx.GetPartialTensor(reshapeLayer.inputs[1]) as PartialTensor<int>;
                if (!shapePartialTensor.isPartiallyKnown)
                    continue;
                var newShape = new PartialTensor<int>(shapePartialTensor.shape);
                for (var i = 0; i < shapePartialTensor.length; i++)
                    newShape[i] = shapePartialTensor[i];

                var input = ctx.GetPartialTensor(reshapeLayer.inputs[0]);
                var output = ctx.GetPartialTensor(reshapeLayer.outputs[0]);

                // try and replace params and unknowns with values
                for (var i = 0; i < output.shape.rank; i++)
                {
                    if (!output.shape[i].isValue)
                        continue;
                    newShape[i] = (PartialTensorElement<int>)output.shape[i];
                }

                // try and replace params with 0
                if (input.shape.hasRank && !reshapeLayer.allowZero)
                {
                    for (var i = 0; i < Mathf.Min(input.shape.rank, shapePartialTensor.length); i++)
                    {
                        if (input.shape[i].EqualsParam(output.shape[i]))
                            newShape[i] = PartialTensorElement<int>.Zero;
                    }
                }

                // try and replace single param or unknown with -1
                var numZero = 0;
                var numMinusOne = 0;
                var numUnknown = 0;
                var unknownIndex = 0;
                for (var i = 0; i < newShape.length; i++)
                {
                    if (!newShape[i].isValue)
                    {
                        numUnknown++;
                        unknownIndex = i;
                        continue;
                    }

                    if (newShape[i].value == 0)
                        numZero++;
                    else if (newShape[i].value == -1)
                        numMinusOne++;
                }

                if (numMinusOne == 0 && numUnknown == 1 && (!reshapeLayer.allowZero || numZero == 0))
                    newShape[unknownIndex] = PartialTensorElement<int>.Value(-1);

                if (!newShape.IsStatic())
                    continue;

                var shapeIndex = model.GetUniqueIndex();
                var shapeConstant = new Constant(model.GetUniqueIndex(), newShape.shape.ToTensorShape(), newShape.ToArray());
                reshapeLayer.inputs[1] = shapeIndex;
                model.AddConstant(shapeConstant);
            }
        }
    }
}
