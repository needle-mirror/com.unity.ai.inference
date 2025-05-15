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

        Node m_Source;
        int m_OutputIndex;
        string m_Name;

        internal PartialTensor partialTensor => m_PartialTensor;
        internal DataType dataType => m_PartialTensor.dataType;
        internal DynamicTensorShape shape => m_PartialTensor.shape;
        internal Node source => m_Source;
        internal int outputIndex => m_OutputIndex;
        internal int index => m_Source.OutputIndices[m_OutputIndex];
        internal string name => m_Name;

        internal FunctionalTensor(PartialTensor partialTensor, Node source, int outputIndex)
        {
            m_PartialTensor = partialTensor;
            m_Source = source;
            m_OutputIndex = outputIndex;
        }

        internal void SetName(string name)
        {
            m_Name = name;
        }

        internal FunctionalTensor Copy()
        {
            return new FunctionalTensor(m_PartialTensor.Copy(), source, m_OutputIndex);
        }

        internal static FunctionalTensor FromTensor(Tensor tensor)
        {
            Constant constant;
            switch (tensor.dataType)
            {
                case DataType.Float:
                {
                    constant = new Constant(-1, tensor.shape, (tensor as Tensor<float>).DownloadToNativeArray().ToArray());
                    break;
                }
                case DataType.Int:
                {
                    constant = new Constant(-1, tensor.shape, (tensor as Tensor<int>).DownloadToNativeArray().ToArray());
                    break;
                }
                default:
                    throw new NotImplementedException();
            }

            var constantNode = new ConstantNode(constant);
            return constantNode.CreateOutputs()[0];
        }

        internal static FunctionalTensor FromConstant(Constant constant)
        {
            var constantNode = new ConstantNode(constant);
            return constantNode.CreateOutputs()[0];
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
