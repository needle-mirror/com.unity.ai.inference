using UnityEngine;

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents data storage for the constant weights of a model.
    /// </summary>
    [PreferBinarySerialization]
    class ModelAssetWeightsData : ScriptableObject
    {
        /// <summary>
        /// The serialized byte array of the data.
        /// </summary>
        [HideInInspector]
        public byte[] value;
    }
}
