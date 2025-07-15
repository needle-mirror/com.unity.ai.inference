using System;
using UnityEngine;
using UnityEditor.AssetImporters;
using System.Runtime.CompilerServices;
using System.Collections.Generic;
using Unity.InferenceEngine.ONNX.Editor;
using UnityEditor;

[assembly: InternalsVisibleTo("Unity.InferenceEngine.EditorTests")]

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents an importer for Open Neural Network Exchange (ONNX) files.
    /// </summary>
    [ScriptedImporter(69, new[] { "onnx" })]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.ai.inference@latest/index.html")]
    class ONNXModelImporter : ScriptedImporter
    {
        [Serializable]
        internal struct DynamicDimConfig
        {
            public string name;
            public int size;
        }

        [SerializeField]
        internal DynamicDimConfig[] dynamicDimConfigs = Array.Empty<DynamicDimConfig>();

        static readonly List<IONNXMetadataImportCallbackReceiver> k_MetadataImportCallbackReceivers;

        static ONNXModelImporter()
        {
            k_MetadataImportCallbackReceivers = new List<IONNXMetadataImportCallbackReceiver>();

            foreach (var type in TypeCache.GetTypesDerivedFrom<IONNXMetadataImportCallbackReceiver>())
            {
                if (type.IsInterface || type.IsAbstract)
                    continue;

                if (Attribute.IsDefined(type, typeof(DisableAutoRegisterAttribute)))
                    continue;

                var receiver = (IONNXMetadataImportCallbackReceiver)Activator.CreateInstance(type);
                RegisterMetadataReceiver(receiver);
            }
        }

        internal static void RegisterMetadataReceiver(IONNXMetadataImportCallbackReceiver receiver)
        {
            k_MetadataImportCallbackReceivers.Add(receiver);
        }

        internal static void UnregisterMetadataReceiver(IONNXMetadataImportCallbackReceiver receiver)
        {
            k_MetadataImportCallbackReceivers.Remove(receiver);
        }

        /// <summary>
        /// Callback that Inference Engine calls when the ONNX model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var converter = new ONNXModelConverter(ctx.assetPath);
            foreach (var dynamicDimConfig in dynamicDimConfigs)
            {
                if(dynamicDimConfig.size == -1)
                    continue;

                if (!converter.DynamicDimConfigs.TryAdd(dynamicDimConfig.name, dynamicDimConfig.size))
                    Debug.LogWarning($"Static size provided multiple times for dynamic dimension {dynamicDimConfig.name}.");
            }

            converter.MetadataLoaded += metadata => InvokeMetadataHandlers(ctx, metadata);
            var model = converter.Convert();

            ModelAsset asset = ScriptableObject.CreateInstance<ModelAsset>();
            ModelWriter.SaveModel(model, out var modelDescriptionBytes, out var modelWeightsBytes);

            ModelAssetData modelAssetData = ScriptableObject.CreateInstance<ModelAssetData>();
            modelAssetData.value = modelDescriptionBytes;
            modelAssetData.name = "Data";
            modelAssetData.hideFlags = HideFlags.HideInHierarchy;
            asset.modelAssetData = modelAssetData;

            asset.modelWeightsChunks = new ModelAssetWeightsData[modelWeightsBytes.Length];
            for (int i = 0; i < modelWeightsBytes.Length; i++)
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

            if (dynamicDimConfigs.Length != model.symbolicDimNames.Length)
            {
                dynamicDimConfigs = new DynamicDimConfig[model.symbolicDimNames.Length];

                for (var i = 0; i < model.symbolicDimNames.Length; i++)
                {
                    var dim = model.symbolicDimNames[i];
                    dynamicDimConfigs[i] = new DynamicDimConfig { name = dim, size =  -1 };
                }
            }

            EditorUtility.SetDirty(this);
        }

        static void InvokeMetadataHandlers(AssetImportContext ctx, ONNXModelMetadata onnxModelMetadata)
        {
            if (k_MetadataImportCallbackReceivers == null)
                return;

            foreach (var receiver in k_MetadataImportCallbackReceivers)
            {
                receiver.OnMetadataImported(ctx, onnxModelMetadata);
            }
        }

        /// <summary>
        /// Attribute to disable automatic registration of <see cref="IONNXMetadataImportCallbackReceiver"/>
        /// implementations. Recommended for testing purposes.
        /// </summary>
        internal class DisableAutoRegisterAttribute : Attribute
        {
        }
    }
}
