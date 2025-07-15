using System;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEngine;

namespace Unity.InferenceEngine.Editor.Visualizer.Editor
{
    static class ModelAssetOpenHandler
    {
        [OnOpenAsset(9999)]
        public static bool OnOpenAssetCallback(int instanceID, int line)
        {
            var obj = EditorUtility.InstanceIDToObject(instanceID);

            if (obj is ModelAsset modelAsset)
            {
                ModelVisualizerWindow.VisualizeModel(modelAsset);
                return true;
            }

            return false;
        }
    }
}
