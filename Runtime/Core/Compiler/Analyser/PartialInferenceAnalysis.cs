using System;
using UnityEngine;

namespace Unity.InferenceEngine.Compiler.Analyser
{
    static class PartialInferenceAnalysis
    {
        public static PartialInferenceContext InferModelPartialTensors(Model model)
        {
            ProfilerMarkers.InferModelPartialTensors.Begin();

            var ctx = new PartialInferenceContext();

            foreach (var constant in model.constants)
            {
                ctx.AddPartialTensor(constant.index, PartialTensor.FromTensor(constant.WeightsToTensorWithSharedTensorData()));
            }

            // model inputs
            foreach (var input in model.inputs)
            {
                ctx.AddPartialTensor(input.index, PartialTensor.Create(input.dataType, input.shape));
            }

            // Partial tensor inference
            foreach (var layer in model.layers)
                layer.InferPartial(i => ctx.GetPartialTensor(layer.inputs[i]), (i, partialTensor) => ctx.AddPartialTensor(layer.outputs[i], partialTensor));

            ProfilerMarkers.InferModelPartialTensors.End();

            return ctx;
        }
    }
}

