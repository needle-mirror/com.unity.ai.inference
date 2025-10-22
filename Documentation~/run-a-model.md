# Run a model

After you [create a worker](create-an-engine.md), call [`Schedule`](xref:Unity.InferenceEngine.Worker.Schedule*) to run the model.

```
worker.Schedule(inputTensor);
```

The first scheduling of a model within the Unity Editor might be slow as Sentis needs to compile code and shaders, including allocating internal memory. Subsequent runs will be faster due to caching.

It’s a good idea to include a test run when you start the application to help improve the initial load time.

For an example, refer to the `Run a model` sample in the [sample scripts](package-samples.md).

## Additional resources

- [Split inference over multiple frames](split-inference-over-multiple-frames.md)
- [Sentis models](models-concept.md)
- [Create an engine to run a model](create-an-engine.md)
- [Profile a model](profile-a-model.md)
