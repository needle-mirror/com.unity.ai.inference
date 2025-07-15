using System;
using System.Linq;
using Microsoft.Msagl.Core.Geometry;
using Unity.AppUI.Redux;
using Unity.InferenceEngine.Editor.Visualizer.GraphData;
using Unity.InferenceEngine.Editor.Visualizer.StateManagement;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Editor.Visualizer.Views.Graph
{
    class FramingManipulator : Manipulator
    {
        GraphView m_Target;
        readonly GraphStoreManager m_GraphStoreManager;
        IDisposableSubscription m_StoreUnsub;

        public FramingManipulator(GraphStoreManager storeManager)
        {
            m_GraphStoreManager = storeManager;
        }

        protected override void RegisterCallbacksOnTarget()
        {
            m_Target = target as GraphView;

            m_StoreUnsub = m_GraphStoreManager.Store.Subscribe(GraphSlice.Name, (GraphState state) =>
            {
                if (state.FocusedObject != null)
                {
                    StartFramingObject(state.FocusedObject);
                    m_GraphStoreManager.Store.Dispatch(m_GraphStoreManager.SetFocusedObject?.Invoke(null));
                }
            });
        }

        protected override void UnregisterCallbacksFromTarget()
        {
            m_Target = null;

            m_StoreUnsub?.Dispose();
        }

        void StartFramingObject(object targetObj)
        {
            switch (targetObj)
            {
                case int index:
                    FrameEdges(index);
                    break;

                case NodeData node:
                    FrameNode(node);
                    break;

                default:
                    throw new NotImplementedException();
            }
        }

        void FrameEdges(int sentisIndex)
        {
            Rect? ogRect = null;
            var relatedEdges = m_Target.EdgeViews.Values.Where(x => x.edgeData.TensorIndex == sentisIndex);
            foreach (var edge in relatedEdges)
            {
                var edgeRect = ToRect(edge.edgeData.Edge.BoundingBox);
                if (ogRect == null)
                {
                    ogRect = edgeRect;
                    continue;
                }

                ogRect = MergeRects(ogRect.Value, edgeRect);
            }

            if (ogRect == null)
                return;

            _ = m_Target.FramePosition(ogRect.Value.center, ogRect.Value.size, false);
        }

        static Rect ToRect(Rectangle rectangle)
        {
            var size = new Vector2((float)rectangle.Width, (float)rectangle.Height);
            var position = new Vector2((float)rectangle.Center.X - size.x / 2f, -(float)rectangle.Center.Y - size.y / 2f);
            return new Rect(position, size);
        }

        static Rect MergeRects(Rect a, Rect b)
        {
            // Find the minimum x and y coordinates
            var xMin = Mathf.Min(a.xMin, b.xMin);
            var yMin = Mathf.Min(a.yMin, b.yMin);

            // Find the maximum x and y coordinates
            var xMax = Mathf.Max(a.xMax, b.xMax);
            var yMax = Mathf.Max(a.yMax, b.yMax);

            // Create a new rect with the calculated dimensions
            return new Rect(xMin, yMin, xMax - xMin, yMax - yMin);
        }

        void FrameNode(NodeData nodeData)
        {
            _ = m_Target.FramePosition(nodeData.CanvasPosition, new Vector2(GraphView.ZoomLevel.MinZoom, GraphView.ZoomLevel.MinZoom), false);
        }
    }
}
