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

struct DataSegment : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_23_5_26(); }
  public static DataSegment GetRootAsDataSegment(ByteBuffer _bb) { return GetRootAsDataSegment(_bb, new DataSegment()); }
  public static DataSegment GetRootAsDataSegment(ByteBuffer _bb, DataSegment obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public DataSegment __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public ulong Offset { get { int o = __p.__offset(4); return o != 0 ? __p.bb.GetUlong(o + __p.bb_pos) : (ulong)0; } }
  public ulong Size { get { int o = __p.__offset(6); return o != 0 ? __p.bb.GetUlong(o + __p.bb_pos) : (ulong)0; } }

  public static Offset<SentisFlatBuffer.DataSegment> CreateDataSegment(FlatBufferBuilder builder,
      ulong offset = 0,
      ulong size = 0) {
    builder.StartTable(2);
    DataSegment.AddSize(builder, size);
    DataSegment.AddOffset(builder, offset);
    return DataSegment.EndDataSegment(builder);
  }

  public static void StartDataSegment(FlatBufferBuilder builder) { builder.StartTable(2); }
  public static void AddOffset(FlatBufferBuilder builder, ulong offset) { builder.AddUlong(0, offset, 0); }
  public static void AddSize(FlatBufferBuilder builder, ulong size) { builder.AddUlong(1, size, 0); }
  public static Offset<SentisFlatBuffer.DataSegment> EndDataSegment(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<SentisFlatBuffer.DataSegment>(o);
  }
}


static class DataSegmentVerify
{
  static public bool Verify(Unity.InferenceEngine.Google.FlatBuffers.Verifier verifier, uint tablePos)
  {
    return verifier.VerifyTableStart(tablePos)
      && verifier.VerifyField(tablePos, 4 /*Offset*/, 8 /*ulong*/, 8, false)
      && verifier.VerifyField(tablePos, 6 /*Size*/, 8 /*ulong*/, 8, false)
      && verifier.VerifyTableEnd(tablePos);
  }
}

}
