using UnityEngine;
using UnityEditor.AssetImporters;

namespace Unity.InferenceEngine.Editor
{
    /// <summary>
    /// Base class for model importer
    /// </summary>
    abstract class ModelImporterBase : ScriptedImporter
    {
        /// <summary>
        /// Callback that Sentis calls when the model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx) { }
    }
}
