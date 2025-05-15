using System.Runtime.CompilerServices;
using Unity.InferenceEngine.Compiler.Passes;
using Unity.InferenceEngine.Compiler.Passes.Cleanup;
using Unity.InferenceEngine.Compiler.Passes.Optimization; // ToArray(), ToDictionary()

[assembly: InternalsVisibleTo("Unity.InferenceEngine.ONNX.Editor")]
[assembly: InternalsVisibleTo("Unity.InferenceEngine.RuntimeTests")]
[assembly: InternalsVisibleTo("Unity.InferenceEngine.EditorTests")]

namespace Unity.InferenceEngine
{
    static class ModelOptimizer
    {
        static void RunPasses(ref Model model, IModelPass[] passes)
        {
            foreach (var pass in passes)
            {
                pass.Run(ref model);
            }
        }

        internal static void OptimizeModel(ref Model model)
        {
            var optimizationPasses = new IModelPass[]
            {
                new EinsumToMatMulPass(),
                new FuseConstantsPass(),
                new RemoveNoOpsPass(),
                new RemoveUnusedPass(),
                new ConcatenateTransposesPass(),
                new ContractToSimplerLayerPass(),
                new RemoveNoOpsPass(),
                new SimplifyReshapeInputPass(),
                new ContractSubExpressionPass(),
                new FuseDensePass(),
                new FuseLinearLayersPass(),
                new FuseActivationPass(),
                new RemoveDuplicatesPass(),
                new RemoveNoOpsPass(),
                // Good to do those passes at the very end
                new RemoveUnusedPass(),
                new RoundDenormalWeightsPass(),
            };

            RunPasses(ref model, optimizationPasses);
        }
    }
}
