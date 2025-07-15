using System;
using Unity.InferenceEngine.Editor.Visualizer.Extensions;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Editor.Visualizer.Views.Graph
{
    class ScrollOffsetManipulator : Manipulator
    {
        GraphView m_GraphView;
        Rect Bound => m_GraphView.GetViewportContent().GetWorldBoundingBox();

        protected override void RegisterCallbacksOnTarget()
        {
            m_GraphView = target as GraphView;
            if (m_GraphView is null)
            {
                throw new InvalidOperationException("ScrollOffsetManipulator can only be used on GraphView.");
            }

            m_GraphView.scrollOffsetChanged += OnScrollOffsetChanged;
        }

        protected override void UnregisterCallbacksFromTarget()
        {
            if (m_GraphView is not null)
            {
                m_GraphView.scrollOffsetChanged -= OnScrollOffsetChanged;
            }
        }

        void OnScrollOffsetChanged()
        {
            var bound = Bound;
            var topLeft = new Rect(bound.xMin, bound.yMin, bound.width, bound.height - bound.yMin);
            var bottomRight = new Rect(bound.xMax, bound.yMax, 0, 0);

            var minOffset = ScrollOffsetForWorldRect(topLeft);
            var maxOffset = ScrollOffsetForWorldRect(bottomRight);

            m_GraphView.scrollOffsetChanged -= OnScrollOffsetChanged;

            var resolvedSize = new Vector2(m_GraphView.resolvedStyle.width, m_GraphView.resolvedStyle.height);
            m_GraphView.scrollOffset = new Vector2(
                Mathf.Clamp(m_GraphView.scrollOffset.x, minOffset.x - bound.width / 2f, maxOffset.x),
                Mathf.Clamp(m_GraphView.scrollOffset.y, minOffset.y - bound.height / 2f + resolvedSize.y / 4f, maxOffset.y - resolvedSize.y / 2f)
            );

            m_GraphView.scrollOffsetChanged += OnScrollOffsetChanged;
        }

        Vector2 ScrollOffsetForWorldRect(Rect worldRect)
        {
            var container = m_GraphView.frameContainer.IsSet ? m_GraphView.frameContainer.Value : m_GraphView.contentRect;
            var containerCenter = new Vector2(container.width * 0.5f, container.height * 0.5f) + container.position;

            var localRect = m_GraphView.WorldToLocal(worldRect);
            var centerDelta = localRect.center - containerCenter;
            var offset = m_GraphView.scrollOffset + centerDelta;

            return offset;
        }
    }
}
