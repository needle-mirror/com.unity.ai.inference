using System;
using System.Collections.Generic;

namespace Unity.InferenceEngine
{
    static class CPUFallbackCalculator
    {
        public static HashSet<int> Calculate(Model model, BackendType backendType)
        {
            // Algorithm:
            // start to gather all CPU seeds:
            //  - all layers that needs a given input to be on the CPU (ie read-back)
            //  - they set their respective inputs to need to run on the CPU
            // foreach layers (starting from the bottom's-up)
            //  if a layer is flagged to need to run on the CPU, all inputs also should run on CPU
            //  exception is holes nodes that operate regardless of their input's data
            // Ex:
            //               c = add   d = concat
            //       ...         \    /
            //        |   s = div(c, d)
            //         \  |
            // t = tile(a, s)
            //      \
            //       mul ...
            // * s is set to need to run on cpu = cpu seed
            // * bottoms up:
            //      - mul -> no cpu skip
            //      - tile -> no cpu skip
            //      - a -> no cpu skip
            //      - s -> is cpu, all inputs (a, d) needs to run on cpu
            //   + continue propagating up to start of graph
            var layerCPUFallback = new HashSet<int>();
            if (backendType == BackendType.CPU)
                return layerCPUFallback;

            for (var i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];

                for (var j = 0; j < layer.inputs.Length; j++)
                {
                    var input = layer.inputs[j];
                    if (input == -1)
                        continue;

                    if (layer.IsInputCPURead(j))
                        layerCPUFallback.Add(input);
                }
            }

            for (var i = model.layers.Count - 1; i >= 0; i--)
            {
                var layer = model.layers[i];

                var isLayerCPU = false;

                foreach (var output in layer.outputs)
                {
                    isLayerCPU |= layerCPUFallback.Contains(output);
                }

                if (!isLayerCPU)
                    continue;

                for (var j = 0; j < layer.inputs.Length; j++)
                {
                    var input = layer.inputs[j];
                    if (input == -1)
                        continue;

                    if (!layer.IsInputNoDataDependency(j))
                        layerCPUFallback.Add(input);
                }
            }

            return layerCPUFallback;
        }
    }
}
