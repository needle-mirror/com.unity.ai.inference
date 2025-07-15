using System;
using Unity.AppUI.Redux;

namespace Unity.InferenceEngine.Editor.Visualizer.StateManagement
{
    sealed class GraphStoreManager : IDisposable
    {
        public IStore<PartitionedState> Store { get; }
        public ActionCreator<object> SetFocusedObject = new(GraphSlice.SetFocusedObject);
        public ActionCreator<object> SetSelectedObject = new(GraphSlice.SetSelectedObject);
        public ActionCreator MoveStackIndexUp = new(GraphSlice.MoveStackIndexUp);
        public ActionCreator MoveStackIndexDown = new(GraphSlice.MoveStackIndexDown);
        public ActionCreator<object> AddHoveredObject = new(GraphSlice.AddHoveredObject);
        public ActionCreator<object> RemoveHoveredObject = new(GraphSlice.RemoveHoveredObject);
        Model m_Model;
        Graph m_Graph;

        public GraphStoreManager(ModelAsset modelAsset)
        {
            m_Model = ModelLoader.Load(modelAsset);

            m_Graph = new Graph(m_Model);
            m_Graph.InitializeNodes();

            var state = new GraphState { ModelAsset = modelAsset, Model = m_Model, Graph = m_Graph };

            var slice = StoreFactory.CreateSlice(
                GraphSlice.Name,
                state,
                builder =>
                {
                    builder.AddCase(SetFocusedObject, GraphReducers.SetFocusedNode);
                    builder.AddCase(SetSelectedObject, GraphReducers.SetSelectedObject);
                    builder.AddCase(MoveStackIndexUp, GraphReducers.MoveStackIndexUp);
                    builder.AddCase(MoveStackIndexDown, GraphReducers.MoveStackIndexDown);
                    builder.AddCase(AddHoveredObject, GraphReducers.AddHoveredObject);
                    builder.AddCase(RemoveHoveredObject, GraphReducers.RemoveHoveredObject);
                });
            Store = StoreFactory.CreateStore(new[] { slice });
        }

        public void Dispose()
        {
            Store?.Dispose();
            m_Graph?.Dispose();
            m_Graph = null;
            m_Model?.DisposeWeights();
            m_Model = null;
        }
    }
}
