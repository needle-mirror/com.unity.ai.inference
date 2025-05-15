using System.Collections.Generic;
using UnityEngine.Assertions;

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents a model graph using the functional API.
    ///
    /// Input functional tensors can be added to the graph, then manipulated using the functional API methods.
    ///
    /// The functional graph can be compiled to return an optimized Inference Engine runtime model.
    /// </summary>
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.Sentis")]
    public class FunctionalGraph
    {
        List<InputNode> m_Inputs = new();
        List<OutputNode> m_Outputs = new();

        /// <summary>
        /// Append an input to the graph with an input def.
        /// </summary>
        /// <param name="dataType">The data type of the input.</param>
        /// <param name="shape">The shape of the input.</param>
        /// <param name="name">The name of the input.</param>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput(DataType dataType, DynamicTensorShape shape, string name = null)
        {
            var index = m_Inputs.Count;
            var input = new InputNode(index, dataType, shape, name);
            m_Inputs.Add(input);
            return input.CreateOutputs()[0];
        }

        /// <summary>
        /// Append an input to the graph with an input def.
        /// </summary>
        /// <param name="dataType">The data type of the input.</param>
        /// <param name="shape">The shape of the input.</param>
        /// <param name="name">The name of the input.</param>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput(DataType dataType, TensorShape shape, string name = null)
        {
            return AddInput(dataType, new DynamicTensorShape(shape), name);
        }

        /// <summary>
        /// Append an input to the graph with a type T and dynamic tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <param name="name">The name of the input.</param>
        /// <typeparam name="T">The data type of the input.</typeparam>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput<T>(DynamicTensorShape shape, string name = null) where T : unmanaged
        {
            return AddInput(AllocatorUtils.ToDataType<T>(), shape, name);
        }

        /// <summary>
        /// Append an input to the graph with a type T and static tensor shape.
        /// </summary>
        /// <param name="shape">The shape of the input.</param>
        /// <param name="name">The name of the input.</param>
        /// <typeparam name="T">The data type of the input.</typeparam>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput<T>(TensorShape shape, string name = null) where T : unmanaged
        {
            return AddInput(AllocatorUtils.ToDataType<T>(), shape, name);
        }

        /// <summary>
        /// Append an input to the graph matching a model input.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="index">The input index of the input in the provided model.</param>
        /// <param name="name">Name to use for this input.
        ///
        /// If name is null Inference Engine uses the name from the model.</param>
        /// <returns>The functional tensor input.</returns>
        public FunctionalTensor AddInput(Model model, int index, string name = null)
        {
            var modelInput = model.inputs[index];
            return AddInput(modelInput.dataType, modelInput.shape, name ?? modelInput.name);
        }

        /// <summary>
        /// Append inputs to the graph matching all of a model's input.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <returns>The functional tensor input array.</returns>
        public FunctionalTensor[] AddInputs(Model model)
        {
            var inputTensors = new FunctionalTensor[model.inputs.Count];
            for (var i = 0; i < inputTensors.Length; i++)
                inputTensors[i] = AddInput(model, i);
            return inputTensors;
        }

        /// <summary>
        /// Append an output to the graph from a functional tensor.
        /// </summary>
        /// <param name="output">The output functional tensor.</param>
        /// <param name="name">The name for the output.
        ///
        /// If null the name will be "output_{idx}" or inferred from the original model in case this functional tensor is the output of a forward pass of an existing model.</param>
        public void AddOutput(FunctionalTensor output, string name = null)
        {
            if (output.source is not LayerNode)
                output = output.Clone();

            m_Outputs.Add(new OutputNode(m_Outputs.Count, output, name));
        }

        /// <summary>
        /// Append outputs to the graph from multiple functional tensors.
        /// </summary>
        /// <param name="outputs">The output functional tensors.</param>
        public void AddOutputs(params FunctionalTensor[] outputs)
        {
            foreach (var output in outputs)
                AddOutput(output);
        }

        /// <summary>
        /// Compile and return an optimized runtime model.
        /// </summary>
        /// <returns>The compiled runtime model.</returns>
        public Model Compile()
        {
            var model = Build();
            ModelOptimizer.OptimizeModel(ref model);
            return model;
        }

        /// <summary>
        /// Compile and return an optimized runtime model with given outputs.
        /// </summary>
        /// <param name="outputs">The outputs.</param>
        /// <returns>The compiled runtime model.</returns>
        public Model Compile(params FunctionalTensor[] outputs)
        {
            Logger.AssertIsTrue(m_Outputs.Count == 0, "Graph outputs have already been added using FunctionalGraph.AddOutput. Call FunctionalGraph.Compile() with no arguments to compile the graph.");
            AddOutputs(outputs);
            return Compile();
        }

        enum NodeProgress
        {
            NotVisited,
            InProgress,
            Done
        }

        internal Model Build()
        {
            // create empty model
            var model = new Model();

            // create for post order traversal algorithm
            var nodeStack = new Stack<Node>(); // stack of nodes to inspect and then process
            var nodeProgress = new Dictionary<Node, NodeProgress>(); // nodes which have been processed and added to the model

            var tensorIndex = 0;

            // iterate inputs to ensure they are in the right order on the model
            foreach (var input in m_Inputs)
            {
                input.AddToModel(model, ref tensorIndex);
                nodeProgress[input] = NodeProgress.Done;
            }

            // queue nodes for the output expressions in reverse order
            for (var i = m_Outputs.Count - 1; i >= 0; i--)
                nodeStack.Push(m_Outputs[i]);

            // push dependency nodes ahead of current node in stack
            // only process node once dependencies have been processed
            while (nodeStack.TryPeek(out var n))
            {
                var nProgress = nodeProgress.GetValueOrDefault(n, NodeProgress.NotVisited);
                if (nProgress == NodeProgress.InProgress)
                {
                    // add node to model
                    Logger.AssertIsTrue(n is not InputNode, "Input expression from incorrect source.");
                    n.AddToModel(model, ref tensorIndex);
                    nodeProgress[n] = NodeProgress.Done;
                    nodeStack.Pop();
                    continue;
                }

                if (nProgress == NodeProgress.Done)
                {
                    // node already added to model
                    nodeStack.Pop();
                    continue;
                }

                // node is not visited, iterate descendants
                nodeProgress[n] = NodeProgress.InProgress;

                for (var i = n.Inputs.Length - 1; i >= 0; i--)
                {
                    if (n.Inputs[i] is null)
                        continue;
                    var m = n.Inputs[i].source;
                    var mProgress = nodeProgress.GetValueOrDefault(m, NodeProgress.NotVisited);
                    if (mProgress == NodeProgress.NotVisited)
                        nodeStack.Push(m);
                    else
                        Assert.IsTrue(mProgress != NodeProgress.InProgress, "Model graph has cycle");
                }
            }

            return model;
        }
    }
}
