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

        protected override InferenceEngine.Model LoadModel(AssetImportContext ctx)
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

            return model;
        }
    }
}
