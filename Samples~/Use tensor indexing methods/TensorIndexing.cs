using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.Assertions;

public class TensorIndexing : MonoBehaviour
{
    [SerializeField]
    Texture2D textureInput;

    void Start()
    {
        // Texture to Tensor and vice versa.
        using var textureAsATensor = new Tensor<float>(new TensorShape(1, 3, textureInput.height, textureInput.width));
        TextureConverter.ToTensor(textureInput, textureAsATensor);

        var renderTexture = RenderTexture.GetTemporary(textureAsATensor.shape[3], textureAsATensor.shape[2], 0);
        TextureConverter.RenderToTexture(textureAsATensor, renderTexture);

        // Clean up the render texture after use.
        RenderTexture.ReleaseTemporary(renderTexture);

        // Declares a tensor of rank 3 of shape (1,2,3).
        using var tensorA = new Tensor<int>(shape: new TensorShape(1, 2, 3), srcData: new int[] { 1, 2, 3, 4, 5, 6 });

        // You can access the tensor shape with the .shape accessor.
        Assert.AreEqual(3, tensorA.shape.rank);
        Assert.AreEqual(1 * 2 * 3, tensorA.shape.length);

        // Shapes can be manipulated like a int[], it supports negative indexing.
        Assert.AreEqual(1, tensorA.shape[0]);
        Assert.AreEqual(2, tensorA.shape[1]);
        Assert.AreEqual(3, tensorA.shape[2]);
        Assert.AreEqual(3, tensorA.shape[-1]);
        Assert.AreEqual(2, tensorA.shape[-2]);
        Assert.AreEqual(1, tensorA.shape[-3]);

        // Shapes can be manipulated in different ways.
        TensorShape shapeB = TensorShape.Ones(rank: 4); // (1,1,1,1)
        shapeB[1] = 2;
        shapeB[2] = 3;
        shapeB[3] = 4;
        Assert.AreEqual(1 * 2 * 3 * 4, shapeB.length);

        // Tensor zero-filled of shape (1,2,3,4).
        using var tensorB = new Tensor<float>(shape: shapeB);

        // You can access tensors via their accessors.
        // If your tensor data is on the GPU you need to call ReadbackAndClone() before accessing with indexes.
        Assert.AreEqual(1, tensorA[0, 0, 0]);
        Assert.AreEqual(2, tensorA[0, 0, 1]);
        Assert.AreEqual(3, tensorA[0, 0, 2]);
        Assert.AreEqual(4, tensorA[0, 1, 0]);
        Assert.AreEqual(5, tensorA[0, 1, 1]);
        Assert.AreEqual(6, tensorA[0, 1, 2]);

        // Each accessors internally flattens the index and uses that to access a flattened representation of the array.
        Assert.AreEqual(1, tensorA[0]); // [0,0,0] = 0*2*3+0*3+0 = 0
        Assert.AreEqual(2, tensorA[1]); // [0,0,1] = 0*2*3+0*3+1 = 1
        Assert.AreEqual(3, tensorA[2]); // [0,0,2] = 0*2*3+0*3+2 = 2
        Assert.AreEqual(4, tensorA[3]); // [0,1,1] = 0*2*3+1*3+0 = 3
        Assert.AreEqual(5, tensorA[4]); // [0,1,2] = 0*2*3+1*3+1 = 4
        Assert.AreEqual(6, tensorA[5]); // [0,1,3] = 0*2*3+1*3+2 = 5

        // Accessors can be used to set values in the tensor.
        // If your tensor data is on the GPU you need to call ReadbackAndClone() before accessing with indexes.
        tensorB[0, 0, 0, 0] = 2.0f;
        tensorB[0, 1, 1, 1] = 3.0f;
        Assert.AreEqual(2.0f, tensorB[0, 0, 0, 0]);
        Assert.AreEqual(3.0f, tensorB[0, 1, 1, 1]);

        // To get the tensor as a flattened array, call ToReadOnlyArray.
        // Tensors can also be created from a slice of a bigger array
        // If your tensor data is on the GPU you need to call ReadbackAndClone() before calling ToReadOnlyArray.
        var arrayA = tensorA.DownloadToArray();
        using var tensorC = new Tensor<int>(shape: new TensorShape(2), srcData: arrayA, dataStartIndex: 4);
        Assert.AreEqual(5, tensorC[0]);
        Assert.AreEqual(6, tensorC[1]);

        // Tensors can also have 0-dim shape, in this case they are empty.
        using var tensorE = new Tensor<float>(shape: new TensorShape(0, 4, 5));
        Assert.AreEqual(3, tensorE.shape.rank);
        Assert.AreEqual(0, tensorE.shape.length);
        Assert.IsTrue(tensorE.shape.HasZeroDims());

        // Array.Empty<float>()
        var arrayE = tensorE.DownloadToArray();
        Assert.AreEqual(0, arrayE.Length);
    }
}
