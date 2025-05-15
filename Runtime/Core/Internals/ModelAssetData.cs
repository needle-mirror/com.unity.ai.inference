using UnityEngine;

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents data storage for an Inference Engine model asset.
    /// </summary>
    [PreferBinarySerialization]
    class ModelAssetData : ScriptableObject
    {
        /// <summary>
        /// The serialized byte array of the data.
        /// </summary>
        [HideInInspector]
        public byte[] value;
    }
}
