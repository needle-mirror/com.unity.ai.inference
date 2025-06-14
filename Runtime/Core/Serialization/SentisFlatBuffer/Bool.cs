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

struct Bool : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_23_5_26(); }
  public static Bool GetRootAsBool(ByteBuffer _bb) { return GetRootAsBool(_bb, new Bool()); }
  public static Bool GetRootAsBool(ByteBuffer _bb, Bool obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public Bool __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public bool BoolVal { get { int o = __p.__offset(4); return o != 0 ? 0!=__p.bb.Get(o + __p.bb_pos) : (bool)false; } }

  public static Offset<SentisFlatBuffer.Bool> CreateBool(FlatBufferBuilder builder,
      bool bool_val = false) {
    builder.StartTable(1);
    Bool.AddBoolVal(builder, bool_val);
    return Bool.EndBool(builder);
  }

  public static void StartBool(FlatBufferBuilder builder) { builder.StartTable(1); }
  public static void AddBoolVal(FlatBufferBuilder builder, bool boolVal) { builder.AddBool(0, boolVal, false); }
  public static Offset<SentisFlatBuffer.Bool> EndBool(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<SentisFlatBuffer.Bool>(o);
  }
}


static class BoolVerify
{
  static public bool Verify(Unity.InferenceEngine.Google.FlatBuffers.Verifier verifier, uint tablePos)
  {
    return verifier.VerifyTableStart(tablePos)
      && verifier.VerifyField(tablePos, 4 /*BoolVal*/, 1 /*bool*/, 1, false)
      && verifier.VerifyTableEnd(tablePos);
  }
}

}
