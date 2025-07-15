# Upgrade to Inference Engine 2.3

You do not need to take any actions to upgrade your project when upgrading from Inference Engine 2.2. If you are upgrading from Sentis 2.1 please follow the instructions below.

Inference Engine 2.3 makes extensive use of [source generators](https://learn.microsoft.com/en-us/shows/on-dotnet/c-source-generators). If you're using an older IDE such as Visual Studio 2019, you may need to upgrade to a newer version to continue debugging your project successfully.

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
