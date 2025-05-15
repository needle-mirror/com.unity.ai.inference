using Unity.Profiling;

namespace Unity.InferenceEngine
{
    static class ProfilerMarkers
    {
        public static readonly ProfilerMarker Schedule = new ProfilerMarker("InferenceEngine.Schedule");
        public static readonly ProfilerMarker LoadComputeShader = new ProfilerMarker("InferenceEngine.ComputeShader.Load");
        public static readonly ProfilerMarker LoadPixelShader = new ProfilerMarker("InferenceEngine.PixelShader.Load");
        public static readonly ProfilerMarker ComputeTensorDataDownload = new ProfilerMarker("InferenceEngine.ComputeTensorData.DownloadDataFromGPU");
        public static readonly ProfilerMarker TextureTensorDataDownload = new ProfilerMarker("InferenceEngine.TextureTensorData.DownloadDataFromGPU");
        public static readonly ProfilerMarker TensorDataPoolAdopt = new ProfilerMarker("InferenceEngine.TensorDataPool.Adopt");
        public static readonly ProfilerMarker TensorDataPoolRelease = new ProfilerMarker("InferenceEngine.TensorDataPool.Release");
        public static readonly ProfilerMarker InferModelPartialTensors = new ProfilerMarker("InferenceEngine.Compiler.Analyser.ShapeInferenceAnalysis.InferModelPartialTensors");
        public static readonly ProfilerMarker LoadModelDesc = new ProfilerMarker("InferenceEngine.ModelLoader.LoadModelDesc");
        public static readonly ProfilerMarker LoadModelWeights = new ProfilerMarker("InferenceEngine.ModelLoader.LoadModelWeights");
        public static readonly ProfilerMarker SaveModelDesc = new ProfilerMarker("InferenceEngine.ModelLoader.SaveModelDesc");
        public static readonly ProfilerMarker SaveModelWeights = new ProfilerMarker("InferenceEngine.ModelLoader.SaveModelWeights");
        public static readonly ProfilerMarker ComputeTensorDataNewEmpty = new ProfilerMarker("InferenceEngine.ComputeTensorData.NewEmpty");
        public static readonly ProfilerMarker ComputeTensorDataNewArray = new ProfilerMarker("InferenceEngine.ComputeTensorData.NewArray");
    }
}
