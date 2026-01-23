using System;
using System.Collections.Generic;
using Unity.InferenceEngine.Editor.Visualizer.GraphData;
using Unity.InferenceEngine.Editor.Visualizer.Views;

namespace Unity.InferenceEngine.Editor.Visualizer.StateManagement
{
    record GraphState: IDisposable
    {

        // Model State
        [NonSerialized]
        public Model Model;
        public ModelAsset ModelAsset;

        // Computation State
        [NonSerialized]
        public PartialInferenceContext PartialInferenceContext;
        public GraphView GraphView;
        public LoadingState LoadingStatus;
        public string ErrorMessage;
        public List<NodeData> Nodes = new();
        public List<EdgeData> Edges = new();

        // UI State
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
            PartialInferenceContext = null;
            Model = null;
            ModelAsset = null;
        }

        public enum LoadingState
        {
            Idle,
            LoadingModel,
            LayoutComputation,
            Done,
            Error
        }
    }
}
