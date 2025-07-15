using System;
using System.Collections.Generic;

namespace Unity.InferenceEngine
{
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.Sentis")]
    public static partial class Functional
    {
        /// <summary>
        /// Creates and returns an array of `FunctionalTensor` as the output of the forward pass of an existing model.
        ///
        /// Inference Engine will make destructive edits of the source model.
        /// </summary>
        /// <param name="model">The model to use as the source.</param>
        /// <param name="inputs">The functional tensors to use as the inputs to the model.</param>
        /// <returns>The functional tensor array.</returns>
        public static FunctionalTensor[] Forward(Model model, params FunctionalTensor[] inputs)
        {
            Logger.AssertIsTrue(inputs.Length == model.inputs.Count, "ModelOutputs.ValueError: inputs length does not equal model input count {0}, {1}", inputs.Length, model.inputs.Count);
            var expressions = new Dictionary<int, FunctionalTensor>();

            for (var i = 0; i < inputs.Length; i++)
                expressions[model.inputs[i].index] = inputs[i];

            foreach (var constant in model.constants)
                expressions[constant.index] = FunctionalTensor.FromConstant(constant);

            foreach (var layer in model.layers)
            {
                layer.inputs = (int[])layer.inputs.Clone();
                layer.outputs = (int[])layer.outputs.Clone();
                var layerInputs = new FunctionalTensor[layer.inputs.Length];
                for (var i = 0; i < layerInputs.Length; i++)
                {
                    if (layer.inputs[i] == -1)
                        continue;
                    layerInputs[i] = expressions[layer.inputs[i]];
                }

                var node = new LayerNode(layerInputs, layer);
                var layerOutputs = node.CreateOutputs();

                for (var i = 0; i < layer.outputs.Length; i++)
                {
                    if (layer.outputs[i] == -1)
                        continue;
                    expressions[layer.outputs[i]] = layerOutputs[i];
                }
            }

            var outputs = new FunctionalTensor[model.outputs.Count];
            for (var i = 0; i < model.outputs.Count; i++)
            {
                outputs[i] = expressions[model.outputs[i].index];
                outputs[i].SetName(model.outputs[i].name);
            }
            return outputs;
        }

        /// <summary>
        /// Creates and returns an array of `FunctionalTensor` as the output of the forward pass of an existing model.
        ///
        /// Inference Engine will copy the source model and not make edits to it.
        /// </summary>
        /// <param name="model">The model to use as the source.</param>
        /// <param name="inputs">The functional tensors to use as the inputs to the model.</param>
        /// <returns>The functional tensor array.</returns>
        public static FunctionalTensor[] ForwardWithCopy(Model model, params FunctionalTensor[] inputs)
        {
            model = model.DeepCopy();
            return Forward(model, inputs);
        }
    }
}
