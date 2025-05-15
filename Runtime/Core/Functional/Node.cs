using System;

namespace Unity.InferenceEngine
{
    abstract class Node
    {
        protected FunctionalTensor[] m_Inputs;
        protected int[] m_OutputIndices;

        public FunctionalTensor[] Inputs => m_Inputs;
        public int[] OutputIndices => m_OutputIndices;

        protected Node(FunctionalTensor[] inputs, int numOutputs)
        {
            m_Inputs = new FunctionalTensor[inputs.Length];
            for (var i = 0; i < m_Inputs.Length; i++)
                m_Inputs[i] = inputs[i] == null ? null : inputs[i].Copy();
            m_OutputIndices = new int[numOutputs];
        }

        public abstract void AddToModel(Model model, ref int index);

        public abstract FunctionalTensor[] CreateOutputs();
    }

    class InputNode : Node
    {
        int m_Index;
        string m_Name;
        DataType m_DataType;
        DynamicTensorShape m_Shape;

        public InputNode(int index, DataType dataType, DynamicTensorShape shape, string name)
            : base(Array.Empty<FunctionalTensor>(), 1)
        {
            m_Index = index;
            m_Name = name ?? "input_" + index;
            m_DataType = dataType;
            m_Shape = shape;
        }

        public override void AddToModel(Model model, ref int index)
        {
            m_OutputIndices[0] = index;
            while (model.inputs.Count <= m_Index)
                model.inputs.Add(default);
            model.inputs[m_Index] = new Model.Input { name = m_Name, index = index, dataType = m_DataType, shape = m_Shape };
            index++;
        }

        public override FunctionalTensor[] CreateOutputs()
        {
            return new[] { new FunctionalTensor(PartialTensor.Create(m_DataType, m_Shape), this, 0) };
        }
    }

    class OutputNode : Node
    {
        int m_Index;
        string m_Name;

        public OutputNode(int index, FunctionalTensor input, string name)
            : base(new[] { input }, 0)
        {
            m_Index = index;
            m_Name = name ?? (input.name ?? "output_" + index);
        }

        public override void AddToModel(Model model, ref int index)
        {
            while (model.outputs.Count <= m_Index)
                model.outputs.Add(default);
            model.outputs[m_Index] = new Model.Output { name = m_Name, index = m_Inputs[0].index };
        }

        public override FunctionalTensor[] CreateOutputs()
        {
            return Array.Empty<FunctionalTensor>();
        }
    }

    class ConstantNode : Node
    {
        Constant m_Constant;

        public ConstantNode(Constant constant)
            : base(Array.Empty<FunctionalTensor>(), 1)
        {
            m_Constant = constant;
        }

        public override void AddToModel(Model model, ref int index)
        {
            m_Constant.index = index;
            m_OutputIndices[0] = index;
            model.constants.Add(m_Constant);
            index++;
        }

        public override FunctionalTensor[] CreateOutputs()
        {
            return new[] { new FunctionalTensor(PartialTensor.FromTensor(m_Constant.WeightsToTensorWithSharedTensorData()), this, 0) };
        }
    }

    class LayerNode : Node
    {
        Layer m_Layer;

        public LayerNode(FunctionalTensor[] inputs, Layer layer)
            : base(inputs, layer.outputs.Length)
        {
            m_Layer = layer;
        }

        public override void AddToModel(Model model, ref int index)
        {
            for (var i = 0; i < m_Inputs.Length; i++)
                m_Layer.inputs[i] = m_Inputs[i] is null ? -1 : m_Inputs[i].index;

            for (var i = 0; i < m_OutputIndices.Length; i++)
            {
                m_OutputIndices[i] = index;
                m_Layer.outputs[i] = index;
                index++;
            }

            model.layers.Add(m_Layer);
        }

        public override FunctionalTensor[] CreateOutputs()
        {
            var outputs = new FunctionalTensor[m_Layer.outputs.Length];
            m_Layer.InferPartial(i => m_Inputs[i] != null ? m_Inputs[i].partialTensor : null, (i, partialTensor) =>
            {
                outputs[i] = new FunctionalTensor(partialTensor, this, i);
            });
            return outputs;
        }
    }
}
