using System;
using System.Collections.Generic;
using Unity.InferenceEngine.Graph;

namespace Unity.InferenceEngine
{
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.Sentis")]
    public static partial class Functional
    {
        /// <summary>
        /// Creates and returns an array of `FunctionalTensor` as the output of the forward pass of an existing model.
        ///
        /// Sentis will make destructive edits of the source model.
        /// </summary>
        /// <param name="model">The model to use as the source.</param>
        /// <param name="inputs">The functional tensors to use as the inputs to the model.</param>
        /// <returns>The functional tensor array.</returns>
        public static FunctionalTensor[] Forward(Model model, params FunctionalTensor[] inputs)
        {
            return Forward(model, inputs, false);
        }

        /// <summary>
        /// Creates and returns an array of `FunctionalTensor` as the output of the forward pass of an existing model.
        ///
        /// Sentis will copy the source model and not make edits to it.
        /// </summary>
        /// <param name="model">The model to use as the source.</param>
        /// <param name="inputs">The functional tensors to use as the inputs to the model.</param>
        /// <returns>The functional tensor array.</returns>
        public static FunctionalTensor[] ForwardWithCopy(Model model, params FunctionalTensor[] inputs)
        {
            return Forward(model, inputs, true);
        }

        internal static FunctionalTensor[] Forward(Model model, FunctionalTensor[] inputs, bool withCopy)
        {
            Logger.AssertIsTrue(inputs.Length == model.inputs.Count, "ModelOutputs.ValueError: inputs length does not equal model input count {0}, {1}", inputs.Length, model.inputs.Count);
            var expressions = new Dictionary<int, FunctionalTensor>();

            for (var i = 0; i < inputs.Length; i++)
                expressions[model.inputs[i].index] = inputs[i];

            foreach (var constant in model.constants)
            {
                var weights = constant.array;
                if (withCopy)
                    weights = weights.ToArray();
                var constantTensor = new ConstantTensor(constant.shape, constant.dataType, weights);
                var constantNode = new ConstantNode(constantTensor);
                expressions[constant.index] = new FunctionalTensor(constantTensor.GetPartialTensor(), constantNode);
            }

            foreach (var layer in model.layers)
            {
                var layerInputs = new Node[layer.inputs.Length];
                for (var i = 0; i < layerInputs.Length; i++)
                    layerInputs[i] = layer.inputs[i] == -1 ? null : new FakeNode(expressions[layer.inputs[i]]);

                var args = GraphConverter.LayerToArgs(layer, layerInputs);
                var layerOutputs = FromLayer(layer.opName, args);

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
    }
}
