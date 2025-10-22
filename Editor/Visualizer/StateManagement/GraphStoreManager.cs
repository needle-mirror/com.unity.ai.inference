using System;
using Unity.AppUI.Redux;
using Unity.InferenceEngine.Compiler.Analyser;

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

        public GraphStoreManager(ModelAsset modelAsset)
        {
            var model = ModelLoader.Load(modelAsset);

            var graph = new Graph(model);
            graph.InitializeNodes();

            var partialInferenceContext = PartialInferenceAnalysis.InferModelPartialTensors(model);

            var state = new GraphState { ModelAsset = modelAsset, Model = model, PartialInferenceContext = partialInferenceContext, Graph = graph};

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
            var state = Store.GetState<GraphState>(GraphSlice.Name);
            state.Dispose();
            Store?.Dispose();
        }
    }
}
