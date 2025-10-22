using System;
using UnityEngine;
using UnityEditor.AssetImporters;
using System.Runtime.CompilerServices;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]

namespace Unity.InferenceEngine.Editor.LiteRT
{
    /// <summary>
    /// Represents an importer for TensorFlow Lite (LiteRT) files.
    /// </summary>
    [ScriptedImporter(2, new[] { "tflite" })]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.ai.inference@latest/index.html")]
    class LiteRTModelImporter : ModelImporterBase
    {
        [SerializeField]
        internal string[] signatureKeys;

        [SerializeField]
        internal string signatureKey;

        /// <summary>
        /// Callback that Sentis calls when the LiteRT model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var converter = new LiteRTModelConverter(ctx.assetPath, signatureKey);
            var model = converter.Convert();
            foreach (var warning in converter.Warnings)
            {
                switch (warning.MessageSeverity)
                {
                    case ModelConverterBase.WarningType.Warning:
                        ctx.LogImportWarning(warning.Message);
                        break;
                    case ModelConverterBase.WarningType.Error:
                        ctx.LogImportError(warning.Message);
                        break;
                    default:
                    case ModelConverterBase.WarningType.None:
                    case ModelConverterBase.WarningType.Info:
                        break;
                }
            }

            signatureKeys = converter.signatureKeys;
            signatureKey = converter.signatureKey;

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
        }
    }
}
