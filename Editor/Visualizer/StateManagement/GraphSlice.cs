using System;

namespace Unity.InferenceEngine.Editor.Visualizer.StateManagement
{
    static class GraphSlice
    {
        public const string Name = "GraphSlice";
        public const string SetFocusedObject = Name + "/SetFocusedNode";
        public const string SetSelectedObject = Name + "/SetSelectedObject";
        public const string MoveStackIndexUp = Name + "/MoveStackIndexUp";
        public const string MoveStackIndexDown = Name + "/MoveStackIndexDown";
        public const string AddHoveredObject = Name + "/AddHoveredObject";
        public const string RemoveHoveredObject = Name + "/RemoveHoveredObject";
    }
}
