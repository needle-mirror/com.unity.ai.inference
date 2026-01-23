using UnityEngine;
using UnityEditor.AssetImporters;

namespace Unity.InferenceEngine.Editor.Torch
{
    /// <summary>
    /// Represents an importer for serialized Inference Engine model files.
    /// </summary>
    [ScriptedImporter(1, new[] { "pt2" })]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.ai.inference@latest/index.html")]
    class TorchModelImporter : ScriptedImporter
    {
        /// <summary>
        /// Callback that Inference Engine calls when the model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var converter = new TorchModelConverter(ctx.assetPath);
            var model = converter.Convert();

            ModelAsset asset = ScriptableObject.CreateInstance<ModelAsset>();
            ModelWriter.SaveModel(model, out var modelDescriptionBytes, out var modelWeightsBytes);

            ModelAssetData modelAssetData = ScriptableObject.CreateInstance<ModelAssetData>();
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
        }
    }
}
