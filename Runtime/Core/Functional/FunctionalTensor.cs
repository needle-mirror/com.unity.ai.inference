using System;
using UnityEngine;

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents a tensor that is a result of tensor operations.
    /// </summary>
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.Sentis")]
    public partial class FunctionalTensor
    {
        PartialTensor m_PartialTensor;
        FunctionalNode m_Source;
        string m_Name;

        internal PartialTensor partialTensor => m_PartialTensor;
        internal DataType dataType => m_PartialTensor.dataType;
        internal DynamicTensorShape shape => m_PartialTensor.shape;
        internal FunctionalNode source => m_Source;
        internal string name => m_Name;

        internal FunctionalTensor(PartialTensor partialTensor, FunctionalNode source, string name = null)
        {
            m_PartialTensor = partialTensor;
            m_Source = source;
            m_Name = name;
        }

        internal void SetName(string name)
        {
            m_Name = name;
        }

        internal FunctionalTensor Copy()
        {
            return new FunctionalTensor(m_PartialTensor.Copy(), source, name);
        }

        internal static FunctionalTensor FromTensor(Tensor tensor)
        {
            var constantTensor = new ConstantTensor(tensor);
            var constantNode = new ConstantNode(constantTensor);
            return new FunctionalTensor(PartialTensor.FromTensor(tensor), constantNode);
        }

        /// <summary>
        /// Returns a string that represents the `FunctionalTensor` with the data type and shape if known.
        /// </summary>
        /// <returns>The string representation of the `FunctionalTensor`.</returns>
        public override string ToString()
        {
            if (shape.IsStatic())
                return $"{dataType}{shape.ToTensorShape()}";
            else
                return $"{dataType}(?)";
        }
    }
}
