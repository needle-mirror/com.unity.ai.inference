// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>
#define ENABLE_SPAN_T
#define UNSAFE_BYTEBUFFER
#define BYTEBUFFER_NO_BOUNDS_CHECK

namespace SentisFlatBuffer
{

using global::System;
using global::System.Collections.Generic;
using global::Unity.InferenceEngine.Google.FlatBuffers;

struct FloatList : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_23_5_26(); }
  public static FloatList GetRootAsFloatList(ByteBuffer _bb) { return GetRootAsFloatList(_bb, new FloatList()); }
  public static FloatList GetRootAsFloatList(ByteBuffer _bb, FloatList obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public FloatList __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public float Items(int j) { int o = __p.__offset(4); return o != 0 ? __p.bb.GetFloat(__p.__vector(o) + j * 4) : (float)0; }
  public int ItemsLength { get { int o = __p.__offset(4); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<float> GetItemsBytes() { return __p.__vector_as_span<float>(4, 4); }
#else
  public ArraySegment<byte>? GetItemsBytes() { return __p.__vector_as_arraysegment(4); }
#endif
  public float[] GetItemsArray() { return __p.__vector_as_array<float>(4); }

  public static Offset<SentisFlatBuffer.FloatList> CreateFloatList(FlatBufferBuilder builder,
      VectorOffset itemsOffset = default(VectorOffset)) {
    builder.StartTable(1);
    FloatList.AddItems(builder, itemsOffset);
    return FloatList.EndFloatList(builder);
  }

  public static void StartFloatList(FlatBufferBuilder builder) { builder.StartTable(1); }
  public static void AddItems(FlatBufferBuilder builder, VectorOffset itemsOffset) { builder.AddOffset(0, itemsOffset.Value, 0); }
  public static VectorOffset CreateItemsVector(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); for (int i = data.Length - 1; i >= 0; i--) builder.AddFloat(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateItemsVectorBlock(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); builder.Add(data); return builder.EndVector(); }
  public static VectorOffset CreateItemsVectorBlock(FlatBufferBuilder builder, ArraySegment<float> data) { builder.StartVector(4, data.Count, 4); builder.Add(data); return builder.EndVector(); }
  public static VectorOffset CreateItemsVectorBlock(FlatBufferBuilder builder, IntPtr dataPtr, int sizeInBytes) { builder.StartVector(1, sizeInBytes, 1); builder.Add<float>(dataPtr, sizeInBytes); return builder.EndVector(); }
  public static void StartItemsVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(4, numElems, 4); }
  public static Offset<SentisFlatBuffer.FloatList> EndFloatList(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<SentisFlatBuffer.FloatList>(o);
  }
}


static class FloatListVerify
{
  static public bool Verify(Unity.InferenceEngine.Google.FlatBuffers.Verifier verifier, uint tablePos)
  {
    return verifier.VerifyTableStart(tablePos)
      && verifier.VerifyVectorOfData(tablePos, 4 /*Items*/, 4 /*float*/, false)
      && verifier.VerifyTableEnd(tablePos);
  }
}

}
