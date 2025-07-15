using System;
using Unity.AppUI.Redux;
using Unity.InferenceEngine.Editor.Visualizer.GraphData;
using Unity.InferenceEngine.Editor.Visualizer.StateManagement;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace Unity.InferenceEngine.Editor.Visualizer.Views
{
    class NodeView : Label
    {
        readonly GraphStoreManager m_StoreManager;
        public NodeData nodeData { get; }
        bool Selected { get; set; }

        const string k_HoveredClass = "hovered";

        public NodeView(GraphStoreManager storeManager, NodeData nodeData)
            : base(nodeData.Name)
        {
            m_StoreManager = storeManager;
            this.nodeData = nodeData;

            usageHints = UsageHints.DynamicTransform;
            style.position = Position.Absolute;
            switch (nodeData)
            {
                case InputNodeData _:
                    AddToClassList("input-node");
                    break;
                case OutputNodeData _:
                    AddToClassList("output-node");
                    break;
                case LayerNodeData layerNode:
                    AddToClassList($"layer-node-{layerNode.Category}");
                    break;
            }

            var sheet = AssetDatabase.LoadAssetAtPath<StyleSheet>("Packages/com.unity.ai.inference/Editor/Visualizer/Styles/Node.uss");
            styleSheets.Add(sheet);

            RegisterCallback<PointerDownEvent>(OnPointerDown);
            RegisterCallback<PointerMoveEvent>(OnPointerMove);
            RegisterCallback<PointerLeaveEvent>(OnPointerLeave);
        }

        public void UpdateState(GraphState state)
        {
            SetSelected(state.SelectedObject == nodeData);
            UpdateHoveredState(state);
        }

        void UpdateHoveredState(GraphState state)
        {
            if (state.HoveredObjects.Contains(nodeData) && state.SelectedObject != nodeData)
            {
                AddToClassList(k_HoveredClass);
                MarkDirtyRepaint();
            }
            else if (ClassListContains(k_HoveredClass))
            {
                RemoveFromClassList(k_HoveredClass);
                MarkDirtyRepaint();
            }
        }

        void OnPointerLeave(PointerLeaveEvent evt)
        {
            if (!m_StoreManager.Store.GetState<GraphState>(GraphSlice.Name).HoveredObjects.Contains(nodeData))
                return;

            evt.StopImmediatePropagation();
            evt.StopPropagation();

            m_StoreManager.Store.Dispatch(m_StoreManager.RemoveHoveredObject.Invoke(nodeData));
        }

        void OnPointerMove(PointerMoveEvent evt)
        {
            if (m_StoreManager.Store.GetState<GraphState>(GraphSlice.Name).HoveredObjects.Contains(nodeData))
                return;

            evt.StopImmediatePropagation();
            evt.StopPropagation();

            m_StoreManager.Store.Dispatch(m_StoreManager.AddHoveredObject.Invoke(nodeData));
        }

        void OnPointerDown(PointerDownEvent evt)
        {
            evt.StopImmediatePropagation();
            evt.StopPropagation();

            if (evt.button != 0 || Selected)
                return;

            m_StoreManager.Store.Dispatch(m_StoreManager.SetSelectedObject.Invoke(nodeData));
        }

        void SetSelected(bool selected)
        {
            if (Selected == selected)
                return;

            Selected = selected;
            if (Selected)
                AddToClassList("selected");
            else
                RemoveFromClassList("selected");

            MarkDirtyRepaint();
        }

        public void UpdateCanvasPosition()
        {
            style.translate = new Translate(nodeData.CanvasPosition.x, nodeData.CanvasPosition.y);
        }
    }
}
