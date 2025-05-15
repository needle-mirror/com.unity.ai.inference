using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using static Unity.InferenceEngine.ShaderPropertyID;

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Provides methods for converting between textures and tensors.
    /// </summary>
    [UnityEngine.Scripting.APIUpdating.MovedFrom("Unity.Sentis")]
    public static class TextureConverter
    {
        /// <summary>
        /// Converts a texture to a `Tensor&lt;float&gt;`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="tensor">The output tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void ToTensor(Texture texture, Tensor<float> tensor, TextureTransform transform = default)
        {
            if (SystemInfo.supportsComputeShaders && ComputeInfo.supportsCompute)
            {
                var cb = new CommandBuffer();
                cb.ToTensor(texture, tensor, transform);
                Graphics.ExecuteCommandBuffer(cb);
                return;
            }

            Logger.AssertIsTrue(tensor.shape.rank == 4, "ToTensor.ValueError: tensor must have rank 4 but has rank {0}", tensor.shape.rank);
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            var width = tensor.shape[transform.tensorLayoutAxisW];
            var height = tensor.shape[transform.tensorLayoutAxisH];
            var channels = tensor.shape[transform.tensorLayoutAxisC];
            Logger.AssertIsTrue(textureChannels >= channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, channels);

            var isExact = texture.height == height && texture.width == width;
            transform.InferChannelSettings(textureChannels);

            var pinO = TextureTensorData.Pin(tensor, transform.tensorLayoutAxisC, false);

            var func = new PixelFunc("Hidden/InferenceEngine/TextureConversion/TextureToTensor");
            func.EnableKeyword(isExact ? "EXACT" : "LINEAR");

            var enableSrgbConversion = GraphicsFormatUtility.IsSRGBFormat(texture.graphicsFormat);
            func.SetKeyword("LINEAR_TO_SRGB", enableSrgbConversion);

            func.SetTexture(k_ID_Xptr, texture);
            func.SetInt(k_ID_StrideWidthO, pinO.blockedShape.Strides(transform.tensorLayoutAxisW));
            func.SetInt(k_ID_StrideWidthO, pinO.blockedShape.Strides(transform.tensorLayoutAxisW));
            func.SetInt(k_ID_StrideHeightO, pinO.blockedShape.Strides(transform.tensorLayoutAxisH));
            func.SetInt(k_ID_WidthO, width);
            func.SetInt(k_ID_HeightO, height);
            func.SetInt(k_ID_Channels, channels);
            func.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
            func.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
            func.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
            func.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
            func.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);

            func.Dispatch(pinO);
        }

        /// <summary>
        /// Appends the conversion of a texture to a `Tensor&lt;float&gt;`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="tensor">The output tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void ToTensor(this CommandBuffer cb, Texture texture, Tensor<float> tensor, TextureTransform transform = default)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "ToTensor.ValueError: tensor must have rank 4 but has rank {0}", tensor.shape.rank);
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            var width = tensor.shape[transform.tensorLayoutAxisW];
            var height = tensor.shape[transform.tensorLayoutAxisH];
            var channels = tensor.shape[transform.tensorLayoutAxisC];
            Logger.AssertIsTrue(textureChannels >= channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, channels);

            var isExact = texture.height == height && texture.width == width;
            transform.InferChannelSettings(textureChannels);

            var fn = isExact ? ComputeFunctions.k_TextureToTensorExact : ComputeFunctions.k_TextureToTensorLinear;
            // NOTE: It's possible that the Graphics team changes the functionality for sRGB in compute shaders
            //       and we might not need this workaround at that point.
            var enableSrgbConversion = GraphicsFormatUtility.IsSRGBFormat(texture.graphicsFormat);
            cb.SetKeyword(fn.shader, new LocalKeyword(fn.shader, "LINEAR_TO_SRGB"), enableSrgbConversion);
            cb.SetComputeTextureParam(fn.shader, fn.kernelIndex, k_ID_X_tex2D, texture);
            cb.SetTensorAsBuffer(fn, k_ID_Optr, ComputeTensorData.Pin(tensor, false));

            cb.SetComputeIntParam(fn.shader, k_ID_O_width, width);
            cb.SetComputeIntParam(fn.shader, k_ID_O_height, height);
            cb.SetComputeIntParam(fn.shader, k_ID_O_channels, channels);
            cb.SetComputeIntParam(fn.shader, k_ID_O_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
            cb.SetComputeIntParam(fn.shader, k_ID_O_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
            cb.SetComputeIntParam(fn.shader, k_ID_O_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
            cb.SetComputeIntParam(fn.shader, k_ID_CoordOrigin, (int)transform.coordOrigin);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleR, transform.channelSwizzleR);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleG, transform.channelSwizzleG);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleB, transform.channelSwizzleB);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleA, transform.channelSwizzleA);

            cb.Dispatch(fn, height, width, 1);
        }

        /// <summary>
        /// Appends the conversion of a RenderTargetIdentifier to a `Tensor&lt;float&gt;`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="rte">The RenderTargetIdentifier to convert.</param>
        /// <param name="tensor">The output tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void ToTensor(this CommandBuffer cb, RenderTargetIdentifier rte, Tensor<float> tensor, TextureTransform transform = default)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "ToTensor.ValueError: tensor must have rank 4 but has rank {0}", tensor.shape.rank);
            var textureChannels = 4;
            var width = tensor.shape[transform.tensorLayoutAxisW];
            var height = tensor.shape[transform.tensorLayoutAxisH];
            var channels = tensor.shape[transform.tensorLayoutAxisC];
            Logger.AssertIsTrue(textureChannels >= channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, channels);

            var isExact = Screen.height == height && Screen.width == width;
            transform.InferChannelSettings(textureChannels);

            var fn = isExact ? ComputeFunctions.k_TextureToTensorExact : ComputeFunctions.k_TextureToTensorLinear;

            cb.SetComputeTextureParam(fn.shader, fn.kernelIndex, k_ID_X_tex2D, rte);
            cb.SetTensorAsBuffer(fn, k_ID_Optr, ComputeTensorData.Pin(tensor));

            cb.SetComputeIntParam(fn.shader, k_ID_O_width, width);
            cb.SetComputeIntParam(fn.shader, k_ID_O_height, height);
            cb.SetComputeIntParam(fn.shader, k_ID_O_channels, channels);
            cb.SetComputeIntParam(fn.shader, k_ID_O_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
            cb.SetComputeIntParam(fn.shader, k_ID_O_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
            cb.SetComputeIntParam(fn.shader, k_ID_O_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
            cb.SetComputeIntParam(fn.shader, k_ID_CoordOrigin, (int)transform.coordOrigin);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleR, transform.channelSwizzleR);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleG, transform.channelSwizzleG);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleB, transform.channelSwizzleB);
            cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleA, transform.channelSwizzleA);

            cb.Dispatch(fn, height, width, 1);
        }

        /// <summary>
        /// Returns the default `RenderTextureFormat` with the provided number of `channels`.
        /// </summary>
        /// <param name="channels">The number of channels.</param>
        /// <returns>The render texture format.</returns>
        static RenderTextureFormat GetDefaultRenderTextureFormatFromComponentCount(int channels)
        {
            Logger.AssertIsTrue(channels <= 4 && channels > 0, "GetDefaultRenderTextureFormatFromComponentCount.ValueError: cannot convert channel count {0} to texture format", channels);
            return channels switch
            {
                1 => RenderTextureFormat.RFloat,
                2 => RenderTextureFormat.RGFloat,
                3 => RenderTextureFormat.RGB111110Float,
                _ => RenderTextureFormat.ARGBFloat
            };
        }

        /// <summary>
        /// Write the float data in a tensor to a render texture. Inference Engine only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the tensor don't match the width and height of the render texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="renderTexture">The render texture to write to. If the value is `null`, Inference Engine blits the tensor data to the screen.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToTexture(Tensor<float> tensor, RenderTexture renderTexture, TextureTransform transform = default)
        {
            if (SystemInfo.supportsComputeShaders && ComputeInfo.supportsCompute)
            {
                var cb = new CommandBuffer();
                cb.RenderToTexture(tensor, renderTexture, transform);
                Graphics.ExecuteCommandBuffer(cb);
                return;
            }

            Logger.AssertIsTrue(tensor.shape.rank == 4, "BlitTensorToTexture.RankError: tensor rank should be equal to 4, got {0}.", tensor.shape.rank);
            if (renderTexture != null)
                Logger.AssertIsTrue(renderTexture.dimension == TextureDimension.Tex2D, "BlitTensorToTexture.ValueError: renderTexture must have dimension Tex2D, got {0}.", renderTexture.dimension);

            var renderWidth = renderTexture != null ? renderTexture.width : Screen.width;
            var renderHeight = renderTexture != null ? renderTexture.height : Screen.height;

            var tensorWidth = tensor.shape[transform.tensorLayoutAxisW];
            var tensorHeight = tensor.shape[transform.tensorLayoutAxisH];
            var tensorChannels = tensor.shape[transform.tensorLayoutAxisC];

            var isExact = renderWidth == tensorWidth && renderHeight == tensorHeight;
            transform.InferChannelSettings(tensorChannels);

            var func = new PixelFunc("Hidden/InferenceEngine/TextureConversion/TensorToTexture");
            func.EnableKeyword(isExact ? "EXACT" : "LINEAR");

            var enableSrgbConversion = renderTexture != null ? renderTexture.sRGB : Display.main.requiresSrgbBlitToBackbuffer;
            func.SetKeyword("SRGB_TO_LINEAR", enableSrgbConversion);

            var pinX = TextureTensorData.Pin(tensor, transform.tensorLayoutAxisC);
            func.SetTensor(k_TensorPropertiesX, pinX);

            switch (transform)
            {
                case { channelSwizzleR: 0, channelSwizzleG: 1, channelSwizzleB: 2, channelSwizzleA: 3 }:
                    func.EnableKeyword("RGBA");
                    break;
                case { channelSwizzleB: 0, channelSwizzleG: 1, channelSwizzleR: 2, channelSwizzleA: 3 }:
                    func.EnableKeyword("BGRA");
                    break;
                default:
                    func.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                    func.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                    func.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                    func.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                    break;
            }

            func.SetInt(k_ID_Stride0X, pinX.blockedShape.Strides(transform.tensorLayoutAxisW));
            func.SetInt(k_ID_Stride1X, pinX.blockedShape.Strides(transform.tensorLayoutAxisH));

            func.SetInt(k_ID_Channels, tensorChannels);
            func.SetInt(k_ID_WidthX, tensorWidth);
            func.SetInt(k_ID_HeightX, tensorHeight);
            func.SetInt(k_ID_WidthO, renderWidth);
            func.SetInt(k_ID_HeightO, renderHeight);
            func.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
            func.SetVector(k_ID_ChannelScale, transform.channelScale);
            func.SetVector(k_ID_ChannelBias, transform.channelBias);

            func.Dispatch(renderTexture);
        }

        /// <summary>
        /// Appends the write the float data in a tensor to a render texture in a CommandBuffer. Inference Engine only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the tensor don't match the width and height of the render texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="renderTexture">The render texture to write to. If the value is `null`, Inference Engine blits the tensor data to the screen.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToTexture(this CommandBuffer cb, Tensor<float> tensor, RenderTexture renderTexture, TextureTransform transform = default)
        {
            Logger.AssertIsTrue(tensor.shape.rank == 4, "BlitTensorToTexture.RankError: tensor rank should be equal to 4, got {0}.", tensor.shape.rank);
            if (renderTexture != null)
                Logger.AssertIsTrue(renderTexture.dimension == TextureDimension.Tex2D, "BlitTensorToTexture.ValueError: renderTexture must have dimension Tex2D, got {0}.", renderTexture.dimension);

            var renderWidth = renderTexture != null ? renderTexture.width : Screen.width;
            var renderHeight = renderTexture != null ? renderTexture.height : Screen.height;

            var tensorWidth = tensor.shape[transform.tensorLayoutAxisW];
            var tensorHeight = tensor.shape[transform.tensorLayoutAxisH];
            var tensorChannels = tensor.shape[transform.tensorLayoutAxisC];

            var isExact = renderWidth == tensorWidth && renderHeight == tensorHeight;
            transform.InferChannelSettings(tensorChannels);

            if (renderTexture != null)
            {
                if (!renderTexture.enableRandomWrite || !renderTexture.IsCreated())
                {
                    renderTexture.Release();
                    renderTexture.enableRandomWrite = true;
                    renderTexture.Create();
                }

                var fn = isExact ? ComputeFunctions.k_TensorToTextureExact : ComputeFunctions.k_TensorToTextureLinear;


                // NOTE: It's possible that the Graphics team changes the functionality for sRGB in compute shaders
                //       and we might not need this workaround at that point.
                var enableSrgbConversion = renderTexture.sRGB && ComputeInfo.IsAppleMGPU();
                cb.SetKeyword(fn.shader, new LocalKeyword(fn.shader, "SRGB_TO_LINEAR"), enableSrgbConversion);

                cb.SetTensorAsBuffer(fn, k_ID_Xptr, ComputeTensorData.Pin(tensor));
                cb.SetComputeTextureParam(fn.shader, fn.kernelIndex, k_ID_O_tex2D, (Texture)renderTexture);

                cb.SetComputeIntParam(fn.shader, k_ID_X_channels, tensorChannels);
                cb.SetComputeIntParam(fn.shader, k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
                cb.SetComputeIntParam(fn.shader, k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
                cb.SetComputeIntParam(fn.shader, k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
                cb.SetComputeIntParam(fn.shader, k_ID_O_width, renderWidth);
                cb.SetComputeIntParam(fn.shader, k_ID_O_height, renderHeight);
                cb.SetComputeIntParam(fn.shader, k_ID_CoordOrigin, (int)transform.coordOrigin);
                cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                cb.SetComputeIntParam(fn.shader, k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                cb.SetComputeVectorParam(fn.shader, k_ID_ChannelScale, transform.channelScale);
                cb.SetComputeVectorParam(fn.shader, k_ID_ChannelBias, transform.channelBias);

                if (!isExact)
                {
                    cb.SetComputeIntParam(fn.shader, k_ID_X_width, tensorWidth);
                    cb.SetComputeIntParam(fn.shader, k_ID_X_height, tensorHeight);
                }

                cb.Dispatch(fn, renderHeight, renderWidth, 1);
            }
            else
            {
                var material = PixelShaderSingleton.Instance.FindMaterial("Hidden/InferenceEngine/TextureConversion/ComputeBufferToTexture");
                material.EnableKeyword(isExact ? "EXACT" : "LINEAR");

                material.SetBuffer(k_ID_Xptr, ComputeTensorData.Pin(tensor).buffer);

                material.SetInt(k_ID_X_channels, tensorChannels);
                material.SetInt(k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
                material.SetInt(k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
                material.SetInt(k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
                material.SetInt(k_ID_O_width, renderWidth);
                material.SetInt(k_ID_O_height, renderHeight);
                material.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
                material.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
                material.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
                material.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
                material.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
                material.SetVector(k_ID_ChannelScale, transform.channelScale);
                material.SetVector(k_ID_ChannelBias, transform.channelBias);

                if (!isExact)
                {
                    material.SetInt(k_ID_X_width, tensorWidth);
                    material.SetInt(k_ID_X_height, tensorHeight);
                }

                cb.Blit(null, renderTexture, material);
            }
        }

        /// <summary>
        /// Write the float data in a tensor to the frame buffer. Inference Engine only writes batch == 0 to the frame buffer.
        ///
        /// If the width and height of the tensor don't match the width and height of the frame buffer, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToScreen(Tensor<float> tensor, TextureTransform transform = default)
        {
            RenderToTexture(tensor, null, transform);
        }

        /// <summary>
        ///  Appends the write the float data in a tensor to the frame buffer texture in a CommandBuffer. Inference Engine only writes batch == 0 to the frame buffer.
        ///
        /// If the width and height of the tensor don't match the width and height of the frame buffer, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToScreen(this CommandBuffer cb, Tensor<float> tensor, TextureTransform transform = default)
        {
            cb.RenderToTexture(tensor, null, transform);
        }

        /// <summary>
        ///  Appends the write the float data in a tensor to the render target identifier in a CommandBuffer. Inference Engine only writes batch == 0 to the frame buffer.
        ///
        /// If the width and height of the tensor don't match the width and height of the frame buffer, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="rte">The render target identifier to write to.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        public static void RenderToScreen(this CommandBuffer cb, Tensor<float> tensor, RenderTargetIdentifier rte, TextureTransform transform = default)
        {
            var renderWidth = Screen.width;
            var renderHeight = Screen.height;

            var tensorWidth = tensor.shape[transform.tensorLayoutAxisW];
            var tensorHeight = tensor.shape[transform.tensorLayoutAxisH];
            var tensorChannels = tensor.shape[transform.tensorLayoutAxisC];

            var isExact = renderWidth == tensorWidth && renderHeight == tensorHeight;
            transform.InferChannelSettings(tensorChannels);

            var material =
                PixelShaderSingleton.Instance.FindMaterial("Hidden/InferenceEngine/TextureConversion/ComputeBufferToTexture");
            material.EnableKeyword(isExact ? "EXACT" : "LINEAR");

            material.SetBuffer(k_ID_Xptr, ComputeTensorData.Pin(tensor).buffer);

            material.SetInt(k_ID_X_channels, tensorChannels);
            material.SetInt(k_ID_X_strideW, tensor.shape.Strides(transform.tensorLayoutAxisW));
            material.SetInt(k_ID_X_strideH, tensor.shape.Strides(transform.tensorLayoutAxisH));
            material.SetInt(k_ID_X_strideC, tensor.shape.Strides(transform.tensorLayoutAxisC));
            material.SetInt(k_ID_O_width, renderWidth);
            material.SetInt(k_ID_O_height, renderHeight);
            material.SetInt(k_ID_CoordOrigin, (int)transform.coordOrigin);
            material.SetInt(k_ID_ChannelSwizzleR, transform.channelSwizzleR);
            material.SetInt(k_ID_ChannelSwizzleG, transform.channelSwizzleG);
            material.SetInt(k_ID_ChannelSwizzleB, transform.channelSwizzleB);
            material.SetInt(k_ID_ChannelSwizzleA, transform.channelSwizzleA);
            material.SetVector(k_ID_ChannelScale, transform.channelScale);
            material.SetVector(k_ID_ChannelBias, transform.channelBias);

            if (!isExact)
            {
                material.SetInt(k_ID_X_width, tensorWidth);
                material.SetInt(k_ID_X_height, tensorHeight);
            }

            cb.Blit(null, rte, material);
        }

        // Obsolete methods

        /// <summary>
        /// Converts a texture to a `Tensor&lt;float&gt;`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="width">The width of the output tensor. If the value is -1, Inference Engine uses the texture to infer the width.</param>
        /// <param name="height">The height of the output tensor. If the value is -1, Inference Engine uses the texture to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output tensor. If the value is -1, Inference Engine uses the texture to infer the number of channels.</param>
        /// <returns>The converted tensor.</returns>
        [Obsolete("`Tensor<float> ToTensor(Texture texture, int width = -1, int height = -1, int channels = -1)` is deprecated, please use `void ToTensor(Texture texture, Tensor<float> tensor, TextureTransform transform)` instead.")]
        public static Tensor<float> ToTensor(Texture texture, int width = -1, int height = -1, int channels = -1)
        {
            return ToTensor(texture, new TextureTransform().SetDimensions(width, height, channels));
        }

        /// <summary>
        /// Converts a texture to a `Tensor&lt;float&gt;`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The converted tensor.</returns>
        [Obsolete("`Tensor<float> ToTensor(Texture texture, TextureTransform transform)` is deprecated, please use `void ToTensor(Texture texture, Tensor<float> tensor, TextureTransform transform)` instead.")]
        public static Tensor<float> ToTensor(Texture texture, TextureTransform transform)
        {
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            transform.InferDimensions(texture.width, texture.height, textureChannels);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var shape = transform.GetTensorShape();

            var O = new Tensor<float>(shape, data: null);
            ToTensor(texture, O, transform);
            return O;
        }

        /// <summary>
        /// Appends the conversion of a texture to a `Tensor&lt;float&gt;`to a CommandBuffer`. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="width">The width of the output tensor. If the value is -1, Inference Engine uses the texture to infer the width.</param>
        /// <param name="height">The height of the output tensor. If the value is -1, Inference Engine uses the texture to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output tensor. If the value is -1, Inference Engine uses the texture to infer the number of channels.</param>
        /// <returns>The converted tensor.</returns>
        [Obsolete("`Tensor<float> ToTensor(this CommandBuffer cb, Texture texture, int width = -1, int height = -1, int channels = -1)` is deprecated, please use `void ToTensor(this CommandBuffer cb, Texture texture, Tensor<float> tensor, TextureTransform transform)` instead.")]
        public static Tensor<float> ToTensor(this CommandBuffer cb, Texture texture, int width = -1, int height = -1, int channels = -1)
        {
            return cb.ToTensor(texture, new TextureTransform().SetDimensions(width, height, channels));
        }

        /// <summary>
        /// Appends the conversion of a texture to a `Tensor&lt;float&gt;`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="texture">The texture to convert. Must be a Texture2D.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The converted tensor.</returns>
        [Obsolete("`Tensor<float> ToTensor(this CommandBuffer cb, Texture texture, TextureTransform transform)` is deprecated, please use `void ToTensor(this CommandBuffer cb, Texture texture, Tensor<float> tensor, TextureTransform transform)` instead.")]
        public static Tensor<float> ToTensor(this CommandBuffer cb, Texture texture, TextureTransform transform)
        {
            var textureChannels = (int)GraphicsFormatUtility.GetComponentCount(texture.graphicsFormat);
            transform.InferDimensions(texture.width, texture.height, textureChannels);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var shape = transform.GetTensorShape();

            var O = new Tensor<float>(shape, data: null);
            cb.ToTensor(texture, O, transform);
            return O;
        }

        /// <summary>
        /// Appends the conversion of a RenderTargetIdentifier to a `Tensor&lt;float&gt;`to a CommandBuffer. The number of channels of the output tensor can be at most the number of channels in the input texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="rte">The RenderTargetIdentifier to convert.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The converted tensor.</returns>
        [Obsolete("`Tensor<float> ToTensor(this CommandBuffer cb, RenderTargetIdentifier rte, TextureTransform transform = default)` is deprecated, please use `void ToTensor(this CommandBuffer cb, RenderTargetIdentifier rte, Tensor<float> tensor, TextureTransform transform = default)` instead.")]
        public static Tensor<float> ToTensor(this CommandBuffer cb, RenderTargetIdentifier rte, TextureTransform transform = default)
        {
            var textureChannels = 4;
            transform.InferDimensions(Screen.width, Screen.height, textureChannels);
            Logger.AssertIsTrue(textureChannels >= transform.channels, "TextureAsTensorInputData.ValueError: texture has fewer channels {0}, than tensor shape {1}", textureChannels, transform.channels);

            var shape = transform.GetTensorShape();

            var O = new Tensor<float>(shape, data: null);
            cb.ToTensor(rte, O, transform);
            return O;
        }

        /// <summary>
        /// Converts the data in a tensor to a render texture. Inference Engine only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the render texture don't match the width and height of the tensor, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="width">The width of the output render texture. If the value is -1, Inference Engine uses the tensor to infer the width.</param>
        /// <param name="height">The height of the output render texture. If the value is -1, Inference Engine uses the tensor to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output render texture. If the value is -1, Inference Engine uses the tensor to infer the number of channels.</param>
        /// <param name="broadcastChannels">When the value is `true`, Inference Engine broadcasts the tensor values to additional channels in the render texture. For example, a tensor with a single channel R maps to (R, R, R, R) if `channels` is 4. When the value is `false`, Inference Engine applies a (0, 0, 0, 1) color mask to additional channels in the render texture. For example, a tensor with a single channel R becomes (R, 0, 0, 1) if `channels` is 4.</param>
        /// <returns>The created render texture.</returns>
        [Obsolete("`RenderTexture ToTexture(Tensor<float> tensor, int width = -1, int height = -1, int channels = -1, bool broadcastChannels = false)` is deprecated, please use `void RenderToTexture(Tensor<float> tensor, RenderTexture renderTexture, TextureTransform transform = default)` instead.")]
        public static RenderTexture ToTexture(Tensor<float> tensor, int width = -1, int height = -1, int channels = -1, bool broadcastChannels = false)
        {
            return ToTexture(tensor, new TextureTransform().SetDimensions(width, height, channels).SetBroadcastChannels(broadcastChannels));
        }

        /// <summary>
        /// Appends the conversion of the data in a tensor to a render texture to a CommandBuffer. Inference Engine only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the render texture don't match the width and height of the tensor, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="width">The width of the output render texture. If the value is -1, Inference Engine uses the tensor to infer the width.</param>
        /// <param name="height">The height of the output render texture. If the value is -1, Inference Engine uses the tensor to infer the height.</param>
        /// <param name="channels">The numbers of channels of the output render texture. If the value is -1, Inference Engine uses the tensor to infer the number of channels.</param>
        /// <param name="broadcastChannels">When the value is `true`, Inference Engine broadcasts the tensor values to additional channels in the render texture. For example, a tensor with a single channel R maps to (R, R, R, R) if `channels` is 4. When the value is `false`, Inference Engine applies a (0, 0, 0, 1) color mask to additional channels in the render texture. For example, a tensor with a single channel R becomes (R, 0, 0, 1) if `channels` is 4.</param>
        /// <returns>The created render texture.</returns>
        [Obsolete("`RenderTexture ToTexture(this CommandBuffer cb, Tensor<float> tensor, int width = -1, int height = -1, int channels = -1, bool broadcastChannels = false)` is deprecated, please use `RenderToTexture(this CommandBuffer cb, Tensor<float> tensor, RenderTexture renderTexture, TextureTransform transform = default)` instead.")]
        public static RenderTexture ToTexture(this CommandBuffer cb, Tensor<float> tensor, int width = -1, int height = -1, int channels = -1, bool broadcastChannels = false)
        {
            return cb.ToTexture(tensor, new TextureTransform().SetDimensions(width, height, channels).SetBroadcastChannels(broadcastChannels));
        }

        /// <summary>
        /// Converts the data in a tensor to a render texture. Inference Engine only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The created render texture.</returns>
        [Obsolete("`RenderTexture ToTexture(Tensor<float> tensor, TextureTransform transform)` is deprecated, please use `RenderToTexture(Tensor<float> tensor, RenderTexture renderTexture, TextureTransform transform = default)` instead.")]
        public static RenderTexture ToTexture(Tensor<float> tensor, TextureTransform transform)
        {
            var width = transform.width == -1 ? tensor.shape[transform.tensorLayoutAxisW] : transform.width;
            var height = transform.height == -1 ? tensor.shape[transform.tensorLayoutAxisH] : transform.height;
            Logger.AssertIsTrue(width <= SystemInfo.maxTextureSize, "TensorToTexture:InputError width can be at most SystemInfo.maxTextureSize, got {0}", width);
            Logger.AssertIsTrue(height <= SystemInfo.maxTextureSize, "TensorToTexture:InputError height can be at most SystemInfo.maxTextureSize, got {0}", height);
            var channels = transform.channels == -1 ? tensor.shape[transform.tensorLayoutAxisC] : transform.channels;

            var renderTexture = new RenderTexture(width, height, 0, GetDefaultRenderTextureFormatFromComponentCount(channels));
            RenderToTexture(tensor, renderTexture, transform);

            return renderTexture;
        }

        /// <summary>
        /// Appends the conversion of the data in a tensor to a render texture in a CommandBuffer. Inference Engine only writes batch == 0 to the render texture.
        ///
        /// If the width and height of the output tensor don't match the width and height of the texture, Inference Engine applies linear resampling.
        /// </summary>
        /// <param name="cb">The CommandBuffer buffer to append graphics command to.</param>
        /// <param name="tensor">The input tensor.</param>
        /// <param name="transform">The optional settings for the conversion. Refer to <see cref="TextureTransform"/> for more information.</param>
        /// <returns>The created render texture.</returns>
        [Obsolete("`RenderTexture ToTexture(this CommandBuffer cb, Tensor<float> tensor, TextureTransform transform)` is deprecated, please use `void RenderToTexture(this CommandBuffer cb, Tensor<float> tensor, RenderTexture renderTexture, TextureTransform transform = default)` instead.")]
        public static RenderTexture ToTexture(this CommandBuffer cb, Tensor<float> tensor, TextureTransform transform)
        {
            var width = transform.width == -1 ? tensor.shape[transform.tensorLayoutAxisW] : transform.width;
            var height = transform.height == -1 ? tensor.shape[transform.tensorLayoutAxisH] : transform.height;
            Logger.AssertIsTrue(width <= SystemInfo.maxTextureSize, "TensorToTexture:InputError width can be at most SystemInfo.maxTextureSize, got {0}", width);
            Logger.AssertIsTrue(height <= SystemInfo.maxTextureSize, "TensorToTexture:InputError height can be at most SystemInfo.maxTextureSize, got {0}", height);
            var channels = transform.channels == -1 ? tensor.shape[transform.tensorLayoutAxisC] : transform.channels;

            var renderTexture = new RenderTexture(width, height, 0, GetDefaultRenderTextureFormatFromComponentCount(channels));
            cb.RenderToTexture(tensor, renderTexture, transform);

            return renderTexture;
        }
    }
}
