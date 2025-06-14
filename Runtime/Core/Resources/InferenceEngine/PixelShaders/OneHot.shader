Shader "Hidden/InferenceEngine/OneHot"
{
    Properties
    {
    }
    SubShader
    {
        // No culling or depth
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma multi_compile_local _ OneHotInt
            #pragma vertex vert
            #pragma fragment frag

            #include "CommonVertexShader.cginc"
            #include "CommonPixelShader.cginc"

            DECLARE_TENSOR(X, int);
            #ifdef OneHotInt
            int onValueInt, offValueInt;
            #define DTYPE4 int4
            #else
            float onValue, offValue;
            #define DTYPE4 float4
            #endif

            uint StrideAxis, DimAxisO;
            int negativeIndexOffset;

            DTYPE4 frag(v2f i, UNITY_VPOS_TYPE screenPos : VPOS) : SV_Target
            {
                uint blockIndexO = GetBlockIndexO(screenPos);
                uint3 lowerAxisUpper = Unravel(uint2(StrideAxis, DimAxisO), blockIndexO);
                int4 indices = SampleBlockX(Ravel(uint1(StrideAxis), lowerAxisUpper.xz));
                bool4 mask4 = false;
                mask4.x = (indices.x == (int4)lowerAxisUpper.y) || ((indices.x + negativeIndexOffset) == (int4)lowerAxisUpper.y);
                mask4.y = (indices.y == (int4)lowerAxisUpper.y) || ((indices.y + negativeIndexOffset) == (int4)lowerAxisUpper.y);
                mask4.z = (indices.z == (int4)lowerAxisUpper.y) || ((indices.z + negativeIndexOffset) == (int4)lowerAxisUpper.y);
                mask4.w = (indices.w == (int4)lowerAxisUpper.y) || ((indices.w + negativeIndexOffset) == (int4)lowerAxisUpper.y);
                #ifdef OneHotInt
                return mask4 ? onValueInt : offValueInt;
                #else
                return mask4 ? onValue : offValue;
                #endif
            }
            ENDCG
        }
    }
}
