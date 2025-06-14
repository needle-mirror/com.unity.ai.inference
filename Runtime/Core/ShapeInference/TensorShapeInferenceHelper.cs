using System;
using UnityEngine;

namespace Unity.InferenceEngine
{
    static class TensorShapeHelper
    {
        public static TensorShape BroadcastShape(Tensor a, Tensor b)
        {
            return a.shape.Broadcast(b.shape);
        }
    }
}

