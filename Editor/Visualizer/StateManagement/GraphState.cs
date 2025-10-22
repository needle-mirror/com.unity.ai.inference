using System;
using System.Collections.Generic;

namespace Unity.InferenceEngine.Editor.Visualizer.StateManagement
{
    record GraphState: IDisposable
    {
        [NonSerialized]
        public Model Model;
        public ModelAsset ModelAsset;
        [NonSerialized]
        public PartialInferenceContext PartialInferenceContext;
        public Graph Graph;
        public object FocusedObject = null;
        public List<object> SelectionHistory = new();
        public int CurrentSelectionIndex = -1;
        public List<object> HoveredObjects = new();

        public object SelectedObject
        {
            get
            {
                try
                {
                    return SelectionHistory[CurrentSelectionIndex];
                }
                catch (ArgumentOutOfRangeException)
                {
                    return null;
                }
            }
        }

        public void Dispose()
        {
            Graph?.Dispose();
            PartialInferenceContext = null;
            Model = null;
            ModelAsset = null;
        }
    }
}
