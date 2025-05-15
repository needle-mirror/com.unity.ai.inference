using UnityEngine;
using UnityEditor.AssetImporters;
using System.IO;
using System.Runtime.CompilerServices;
using System.Collections.Generic;

[assembly: InternalsVisibleTo("Unity.InferenceEngine.EditorTests")]
[assembly: InternalsVisibleTo("Unity.InferenceEngine.RuntimeTests")]

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents an importer for serialized Inference Engine model files.
    /// </summary>
    [ScriptedImporter(2, new[] { "sentis" })]
    class InferenceEngineModelImporter : ScriptedImporter
    {
        /// <summary>
        /// Callback that Inference Engine calls when the ONNX model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var model = ModelLoader.Load(ctx.assetPath);
            if (model == null)
                return;

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
            model.DisposeWeights();
        }
    }
}
