using System;
using Unity.InferenceEngine.Editor.DynamicDims;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Unity.InferenceEngine.Editor.Sentis
{
    /// <summary>
    /// Represents an importer for serialized Sentis model files.
    /// </summary>
    [ScriptedImporter(4, new[] { "sentis" })]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.ai.inference@latest/index.html")]
    class SentisModelImporter : ModelImporterBase, IDynamicDimImporter
    {
        [SerializeField]
        internal DynamicDimConfig[] dynamicDimConfigs = Array.Empty<DynamicDimConfig>();

        DynamicDimConfig[] IDynamicDimImporter.dynamicDimConfigs
        {
            get => dynamicDimConfigs;
            set => dynamicDimConfigs = value;
        }

        protected override Model LoadModel(AssetImportContext ctx)
        {
            var model = ModelLoader.Load(ctx.assetPath);
            if (model == null)
                return null;

            this.InitializeDynamicDimsConfig(model);
            this.ApplyDynamicDimConfigs(model);
            this.CleanModelDynamicDims(model);
            return model;

        }
    }
}
