using UnityEditor;
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
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var model = LoadModel(ctx);
            if (model == null)
            {
                UnityEngine.Debug.LogError("Failed to load model.");
                return;
            }

            var asset = ScriptableObject.CreateInstance<ModelAsset>();
            ModelWriter.SaveModel(model, out var modelDescriptionBytes, out var modelWeightsBytes);

            var modelAssetData = ScriptableObject.CreateInstance<ModelAssetData>();
            modelAssetData.value = modelDescriptionBytes;
            modelAssetData.name = "Data";
            modelAssetData.hideFlags = HideFlags.HideInHierarchy;
            asset.modelAssetData = modelAssetData;

            asset.modelWeightsChunks = new ModelAssetWeightsData[modelWeightsBytes.Length];
            for (var i = 0; i < modelWeightsBytes.Length; i++)
            {
                asset.modelWeightsChunks[i] = ScriptableObject.CreateInstance<ModelAssetWeightsData>();
                asset.modelWeightsChunks[i].value = modelWeightsBytes[i];
                asset.modelWeightsChunks[i].name = "Data";
                asset.modelWeightsChunks[i].hideFlags = HideFlags.HideInHierarchy;

                ctx.AddObjectToAsset($"model data weights {i}", asset.modelWeightsChunks[i]);
            }

            ctx.AddObjectToAsset("main obj", asset);
            ctx.AddObjectToAsset("model data", modelAssetData);

            ctx.SetMainObject(asset);

            EditorUtility.SetDirty(this);
        }

        protected abstract Model LoadModel(AssetImportContext ctx);
    }
}
