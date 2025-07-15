using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.AppUI.Redux;
using Unity.AppUI.UI;
using Unity.InferenceEngine.Editor.Visualizer.GraphData;
using Unity.InferenceEngine.Editor.Visualizer.StateManagement;
using Unity.InferenceEngine.Editor.Visualizer.Views.Edges;
using Unity.InferenceEngine.Editor.Visualizer.Views.Graph;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;
using Canvas = Unity.AppUI.UI.Canvas;

namespace Unity.InferenceEngine.Editor.Visualizer.Views
{
    class GraphView : Canvas
    {
        const string k_StylePath = "Packages/com.unity.ai.inference/Editor/Visualizer/Styles/GraphView.uss";

        GraphStoreManager m_StoreManager;
        GraphInspector m_GraphInspector;
        GraphFinder m_GraphFinder;
        EdgeEventsManipulator m_EdgeEventsManipulator;
        VisualsStateHandler m_VisualsStateHandler;
        ScrollOffsetManipulator m_ScrollOffsetManipulator;

        readonly Dictionary<NodeData, NodeView> m_NodeViews = new();
        readonly Dictionary<EdgeData, EdgeView> m_EdgeViews = new();

        NodeData m_HighestNodeData;
        EditorApplication.CallbackFunction m_OnAnimationUpdate;

        public Dictionary<NodeData, NodeView> NodeViews => m_NodeViews;
        public Dictionary<EdgeData, EdgeView> EdgeViews => m_EdgeViews;

        public GraphView()
        {
            var styleSheet = AssetDatabase.LoadAssetAtPath<StyleSheet>(k_StylePath);
            styleSheets.Add(styleSheet);
        }

        public void Initialize(GraphStoreManager storeManager)
        {
            m_StoreManager = storeManager;
            CreateGraph();
            this.AddManipulator(new FramingManipulator(m_StoreManager));
            maxZoom = 2.5f;
        }

        void CreateGraph()
        {
            Clear();

            m_ScrollOffsetManipulator = new ScrollOffsetManipulator();
            this.AddManipulator(m_ScrollOffsetManipulator);

            m_EdgeEventsManipulator = new EdgeEventsManipulator(m_StoreManager);
            this.AddManipulator(m_EdgeEventsManipulator);

            m_GraphInspector = new GraphInspector(m_StoreManager, this);

            // Setup finder and add it to the panel
            m_GraphFinder = new GraphFinder(this, m_StoreManager);
            var appUiPanel = GetFirstAncestorOfType<Panel>();
            appUiPanel.Add(m_GraphFinder);

            RegisterCallback<PointerDownEvent>(OnPointerDown);
            InitializeVisuals();
            MarkDirtyRepaint();

            RegisterCallback<GeometryChangedEvent>(OnGeometryChanged);

            // We use trickle down to intercept the event before base class can consume it
            RegisterCallback<KeyDownEvent>(OnKeyDown, TrickleDown.TrickleDown);
        }

        void InitializeVisuals()
        {
            Clear();
            var state = m_StoreManager.Store.GetState<GraphState>(GraphSlice.Name);
            foreach (var node in state.Graph.Nodes)
            {
                var nodeView = new NodeView(m_StoreManager, node);
                m_NodeViews.Add(node, nodeView);
                Add(nodeView);
            }
        }

        void OnKeyDown(KeyDownEvent evt)
        {
            if (evt.keyCode == KeyCode.F)
            {
                if (evt.commandKey || evt.ctrlKey) //Handle Ctrl+F/Cmd+F to focus the search bar
                {
                    m_GraphFinder.searchBar.Focus();
                }
                else //Handle view reset
                {
                    _ = FrameHighestNodeNextFrames(0);
                }

                evt.StopPropagation();
                evt.StopImmediatePropagation();
            }
        }

        void OnGeometryChanged(GeometryChangedEvent evt)
        {
            UnregisterCallback<GeometryChangedEvent>(OnGeometryChanged);
            ComputeLayout();

            m_VisualsStateHandler = new VisualsStateHandler(m_StoreManager, m_NodeViews, m_EdgeViews);
            this.AddManipulator(m_VisualsStateHandler);
        }

        void OnPointerDown(PointerDownEvent evt)
        {
            if (evt.button != 0)
                return;

            m_StoreManager.Store.Dispatch(m_StoreManager.SetSelectedObject?.Invoke(null));
        }

        void ComputeLayout()
        {
            var state = m_StoreManager.Store.GetState<GraphState>(GraphSlice.Name);
            UpdateNodesCanvasSize();
            state.Graph.ComputeLayout();
            CreateEdges();
            UpdateNodesCanvasPosition();
            _ = FrameHighestNodeNextFrames(3);
        }

        void UpdateNodesCanvasSize()
        {
            foreach (var nodeKvp in m_NodeViews)
            {
                var node = nodeKvp.Key;
                var nodeView = nodeKvp.Value;
                node.CanvasSize = new Vector2(nodeView.worldBound.width, nodeView.worldBound.height);
            }
        }

        void UpdateNodesCanvasPosition()
        {
            foreach (var nodeKvp in m_NodeViews)
            {
                var node = nodeKvp.Value;
                node.UpdateCanvasPosition();

                // Track the topmost node for initial framing
                if (m_HighestNodeData == null || nodeKvp.Key.CanvasPosition.y < m_HighestNodeData.CanvasPosition.y)
                {
                    m_HighestNodeData = nodeKvp.Key;
                }
            }
        }

        void CreateEdges()
        {
            foreach (var edge in m_EdgeViews.Values)
            {
                Remove(edge);
            }

            m_EdgeViews.Clear();

            var state = m_StoreManager.Store.GetState<GraphState>(GraphSlice.Name);
            foreach (var edge in state.Graph.Edges)
            {
                var edgeView = new EdgeView(m_StoreManager, edge);
                m_EdgeViews.Add(edge, edgeView);
                Insert(0, edgeView); // Insert at index 0 to render edges behind nodes
            }
        }

        public async Task FramePosition(Vector2 position, Vector2 rectSize, bool skipAnimation)
        {
            this.RemoveManipulator(m_ScrollOffsetManipulator);
            const float animationSpeed = 0.1f;

            var currentZoom = zoom;
            var currentTranslation = scrollOffset;

            // Enforce minimum size constraints for better framing
            rectSize.x = Mathf.Max(rectSize.x, ZoomLevel.MinZoom);
            rectSize.y = Mathf.Max(rectSize.y, ZoomLevel.MinZoom);

            // Adjust position to center the rect
            position -= rectSize / 2f;

            // Calculate target zoom and position
            FrameArea(new Rect(position, rectSize));

            if (skipAnimation)
            {
                this.AddManipulator(m_ScrollOffsetManipulator);
                return;
            }

            var targetZoom = zoom;
            var trayElement = m_GraphInspector.Tray.view.Q("appui-tray__tray");
            var targetTranslation = scrollOffset + new Vector2(trayElement.resolvedStyle.width / 2f, 0);

            // Reset to starting values before animation
            zoom = currentZoom;
            scrollOffset = currentTranslation;

            var elapsed = 0.0;
            var lastTime = EditorApplication.timeSinceStartup;

            // Setup animation callback to run each editor update
            m_OnAnimationUpdate = () =>
            {
                var now = EditorApplication.timeSinceStartup;
                var deltaTime = now - lastTime;
                lastTime = now;
                elapsed += deltaTime;
                var progress = Mathf.Clamp01((float)(elapsed / animationSpeed));

                // Animate zoom and position
                zoom = Mathf.Lerp(currentZoom, targetZoom, progress);
                scrollOffset = Vector2.Lerp(currentTranslation, targetTranslation, progress);
                MarkDirtyRepaint();

                if (elapsed >= animationSpeed)
                {
                    // Cleanup when animation completes
                    zoom = targetZoom;
                    scrollOffset = targetTranslation;
                    EditorApplication.update -= m_OnAnimationUpdate;
                    m_OnAnimationUpdate = null;
                    this.AddManipulator(m_ScrollOffsetManipulator);
                }
            };

            EditorApplication.update += m_OnAnimationUpdate;

            // Wait for animation to complete
            while (elapsed < animationSpeed)
            {
                await Task.Yield();
            }
        }

        async Task FrameHighestNodeNextFrames(int frameCount = 1)
        {
            // Hide content until framing is complete
            contentContainer.style.display = DisplayStyle.None;

            // Wait for layout to stabilize
            for (var i = 0; i < frameCount; i++)
            {
                await Task.Yield();
            }

            // Center highest node vertically in view
            var startPosition = m_HighestNodeData.CanvasPosition;
            startPosition.y += (resolvedStyle.height - m_HighestNodeData.CanvasSize.y) / 2f;

            await FramePosition(startPosition, new Vector2(ZoomLevel.MaxZoom, ZoomLevel.MaxZoom), true);

            // Show content after framing
            contentContainer.style.display = DisplayStyle.Flex;
        }

        public static class ZoomLevel
        {
            public const float MinZoom = 300f;
            public const float MaxZoom = 800f;
        }
    }
}
