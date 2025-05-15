# Upgrade to Inference Engine 2.2

To upgrade from Sentis 2.1 to Inference Engine 2.2, follow these steps:

1. Open the **Package Manager** and remove the **Sentis** package from your project.
2. In the **Package Manager**, select **+** > **Add package by name**. 
3. Enter `com.unity.ai.inference` and select **Add**.
4. When prompted, use Unity’s automatic API updater or manually replace all instances of `Unity.Sentis` with `Unity.InferenceEngine`.

## Additional resources

* [Get started](get-started.md)
* [Create a model](create-a-model.md)
* [Run an imported model](run-an-imported-model.md)
* [Use Tensors](use-tensors.md)