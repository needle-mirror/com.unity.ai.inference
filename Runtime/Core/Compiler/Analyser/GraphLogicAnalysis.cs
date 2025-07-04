using System;
using System.Linq;
using UnityEngine;

namespace Unity.InferenceEngine.Compiler.Analyser
{
    static class GraphLogicAnalysis
    {
        public static int GetDownStreamLayersCount(Model model, int index)
        {
            return model.layers.Count(x => x.inputs.Contains(index));
        }
    }
}

