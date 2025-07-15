using System;
using System.Collections.Generic;
using Unity.InferenceEngine.Layers;

namespace Unity.InferenceEngine.Compiler.Passes.Optimization
{
    class RemoveDuplicateLayersPass : IModelPass
    {
        static int GetHashCode(Layer layer, Dictionary<int, int> duplicateConstants)
        {
            var hashCode = layer.GetHashCode();
            foreach (var input in layer.inputs)
            {
                var remappedInput = duplicateConstants.GetValueOrDefault(input, input);
                hashCode = HashCode.Combine(hashCode, remappedInput);
            }

            return hashCode;
        }

        public void Run(ref Model model)
        {
            var duplicateConstants = DuplicateConstantCalculator.CalculateDuplicateConstants(model);

            // Algorithm: remove same layers
            // a layer is the same if it has the same types and all fields and inputs are the same
            // foreach layer:
            //  compute soft hash on layer inputs + type
            //  foreach collision:
            //    remove layer if equal (full param check) to collision
            var remapRemovedIndexes = new Dictionary<int, int>();
            var layersToRemove = new HashSet<int>();
            var layerByInput = new Dictionary<long, List<Layer>>();
            var comparableFieldsByLayer = new Dictionary<int, List<object>>();
            foreach (var layer in model.layers)
            {
                // in place input rename, to propagate removal stat mid traversal
                for (int i = 0; i < layer.inputs.Length; i++)
                {
                    var input = layer.inputs[i];
                    if (remapRemovedIndexes.ContainsKey(input))
                        layer.inputs[i] = remapRemovedIndexes[input];
                }

                long hash = GetHashCode(layer, duplicateConstants);
                if (!layerByInput.TryGetValue(hash, out var collisionLayers))
                {
                    layerByInput.Add(hash, new List<Layer>() { layer });
                    continue;
                }

                bool removed = false;
                foreach (var similarLayer in collisionLayers)
                {
                    if (!layer.IsEquivalent(similarLayer))
                        continue;

                    if (layer is RandomLayer { hasSeed: false })
                        continue;

                    var inputsAllEqual = true;
                    for (int i = 0; i < layer.inputs.Length && inputsAllEqual; i++)
                        inputsAllEqual &= duplicateConstants.GetValueOrDefault(layer.inputs[i], layer.inputs[i]) == duplicateConstants.GetValueOrDefault(similarLayer.inputs[i], similarLayer.inputs[i]);
                    if (!inputsAllEqual)
                        continue;

                    remapRemovedIndexes.Add(layer.outputs[0], similarLayer.outputs[0]);

                    layersToRemove.Add(layer.outputs[0]);
                    removed = true;

                    if (layer.outputs.Length != similarLayer.outputs.Length)
                        break;

                    for (int i = 0; i < layer.outputs.Length; i++)
                    {
                        if (!remapRemovedIndexes.ContainsKey(layer.outputs[i]))
                            remapRemovedIndexes.Add(layer.outputs[i], similarLayer.outputs[i]);
                    }

                    break;
                }

                if (!removed)
                    collisionLayers.Add(layer);
            }

            model.layers.RemoveAll(l => layersToRemove.Contains(l.outputs[0]));

            // all inputs have been remapped in place, no need to update layers

            for (var i = 0; i < model.outputs.Count; i++)
            {
                if (remapRemovedIndexes.TryGetValue(model.outputs[i].index, out var remappedIndex))
                {
                    model.outputs[i] = new Model.Output{
                        name = model.outputs[i].name,
                        index = remappedIndex
                    };
                }
            }
        }
    }

    static class DuplicateConstantCalculator
    {
        static long GetHashCode(Constant constant)
        {
            var hashCode = constant.shape.GetHashCode();

            if (constant.shape.HasZeroDims())
                return hashCode;

            for (var i = 0; i < constant.shape.length; i++)
                hashCode = HashCode.Combine(hashCode, constant.weights.Get<int>(i));

            return hashCode;
        }

        static bool AreEqual(Constant c0, Constant c1)
        {
            if (c0.shape != c1.shape)
                return false;

            if (c0.shape.HasZeroDims() && c1.shape.HasZeroDims())
                return true;

            for (int i = 0; i < c0.shape.length; i++)
            {
                int v0 = c0.weights.Get<int>(i);
                int v1 = c1.weights.Get<int>(i);
                if (v0 != v1)
                    return false;
            }

            return true;
        }

        public static Dictionary<int, int> CalculateDuplicateConstants(Model model)
        {
            // Algorithm: remove same constant
            // a constant is the same if it's length/shape/weights are all identical
            // foreach constant:
            //  compute first soft hash on constant length
            //  if equal compute hash on constant weights
            //     check secondary hashmap on weight.hash
            //     if collision, hard comparison
            // N.B: no handling of potential wrong collision on weight.hash
            var constantsToRemove = new Dictionary<int, int>();
            var shapeHashTupleToConstants = new Dictionary<Tuple<TensorShape, long>, List<Constant>>();
            foreach (var constant in model.constants)
            {
                if (constant.dataType != DataType.Int)
                    continue;

                var key = new Tuple<TensorShape, long>(constant.shape, GetHashCode(constant));
                if (!shapeHashTupleToConstants.TryGetValue(key, out var potentialSimilarConstants))
                {
                    shapeHashTupleToConstants.Add(key, new List<Constant> { constant });
                    continue;
                }

                bool removed = false;
                foreach (var similarConstant in potentialSimilarConstants)
                {
                    // collision, double check values
                    if (!AreEqual(constant, similarConstant))
                        continue;

                    removed = true;
                    constantsToRemove.Add(constant.index, similarConstant.index);
                    break;
                }

                if (!removed)
                    potentialSimilarConstants.Add(constant);
            }

            return constantsToRemove;
        }
    }

    class RemoveDuplicatesPass : IModelPass
    {
        public void Run(ref Model model)
        {
            var removeLayers = new RemoveDuplicateLayersPass();
            removeLayers.Run(ref model);
        }
    }
}
