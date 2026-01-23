using System;
using UnityEngine;
using UnityEditor.AssetImporters;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.InferenceEngine.Editor.DynamicDims;
using UnityEditor;

[assembly: InternalsVisibleTo("Unity.InferenceEngine.EditorTests")]

namespace Unity.InferenceEngine.Editor.Onnx
{
    /// <summary>
    /// Represents an importer for Open Neural Network Exchange (ONNX) files.
    /// </summary>
    [ScriptedImporter(72, new[] { "onnx" })]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.ai.inference@latest/index.html")]
    class ONNXModelImporter : ModelImporterBase, IDynamicDimImporter
    {
        [SerializeField]
        internal DynamicDimConfig[] dynamicDimConfigs = Array.Empty<DynamicDimConfig>();

        DynamicDimConfig[] IDynamicDimImporter.dynamicDimConfigs
        {
            get => dynamicDimConfigs;
            set => dynamicDimConfigs = value;
        }

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

        protected override Model LoadModel(AssetImportContext ctx)
        {
            var converter = new ONNXModelConverter(ctx.assetPath);
            converter.MetadataLoaded += metadata => InvokeMetadataHandlers(ctx, metadata);

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

            this.InitializeDynamicDimsConfig(model);
            this.ApplyDynamicDimConfigs(model);
            this.CleanModelDynamicDims(model);

            return model;
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
        internal class DisableAutoRegisterAttribute : Attribute { }

    }
}
