using System;
using System.Text;

namespace Unity.InferenceEngine
{
    /// <summary>
    /// Represents a tensor during PartialInference, when input data and shapes are not fully defined.
    /// </summary>
    abstract class PartialTensor
    {
        protected DataType m_DataType;
        protected DynamicTensorShape m_Shape;

        /// <summary>
        /// The data type of the partial tensor as a DataType.
        /// </summary>
        public DataType dataType => m_DataType;

        /// <summary>
        /// The shape of the partial tensor as a DynamicTensorShape.
        /// </summary>
        public DynamicTensorShape shape => m_Shape;

        /// <summary>
        /// Whether partial elements are stored.
        /// </summary>
        public abstract bool isPartiallyKnown { get; }

        /// <summary>
        /// The number of partial elements stored, is equal to the length of the tensor if partial elements are stored.
        /// </summary>
        public abstract int length { get; }

        public PartialTensorElement<T> Get<T>(int index = 0) where T : unmanaged
        {
            return (this as PartialTensor<T>)[index];
        }

        public void Set<T>(int index, PartialTensorElement<T> element) where T : unmanaged
        {
            (this as PartialTensor<T>)[index] = element;
        }

        public static PartialTensor Create(DataType dataType) => Create(dataType, DynamicTensorShape.DynamicRank);

        public static PartialTensor Create(DataType dataType, DynamicTensorShape shape)
        {
            return dataType switch
            {
                DataType.Float => new PartialTensor<float>(shape),
                DataType.Int => new PartialTensor<int>(shape),
                DataType.Byte => new PartialTensor<byte>(shape),
                DataType.Short => new PartialTensor<short>(shape),
                _ => throw new ArgumentOutOfRangeException(nameof(dataType), dataType, null)
            };
        }

        public static PartialTensor FromTensor(Tensor t)
        {
            return t.dataType switch
            {
                DataType.Float => PartialTensor<float>.FromTensor(t as Tensor<float>),
                DataType.Int => PartialTensor<int>.FromTensor(t as Tensor<int>),
                DataType.Byte => PartialTensor<byte>.FromTensor(t as Tensor<byte>),
                DataType.Short => PartialTensor<short>.FromTensor(t as Tensor<short>),
                _ => throw new ArgumentOutOfRangeException(nameof(t.dataType), t.dataType, null)
            };
        }

        /// <summary>
        /// Creates and returns an integer partial tensor with a given 'shape' and elements all ones if the length is small enough.
        /// </summary>
        public static PartialTensor<int> Ones(DynamicTensorShape shape)
        {
            var partialTensor = new PartialTensor<int>(shape);
            if (!partialTensor.isPartiallyKnown)
                return partialTensor;
            for (var i = 0; i < partialTensor.length; i++)
            {
                partialTensor[i] = PartialTensorElement<int>.Value(1);
            }

            return partialTensor;
        }

        /// <summary>
        /// Creates and returns an integer partial tensor with a given 'shape' and elements from an range if the length is small enough.
        /// </summary>
        public static PartialTensor<int> Range(int start, int end)
        {
            var partialTensor = new PartialTensor<int>(new DynamicTensorShape(DynamicTensorDim.Int(end - start)));
            if (!partialTensor.isPartiallyKnown)
                return partialTensor;
            for (var i = 0; i < partialTensor.length; i++)
            {
                partialTensor[i] = PartialTensorElement<int>.Value(start + i);
            }

            return partialTensor;
        }

        /// <summary>
        /// Whether two partial tensors are identical.
        /// note this can return false even when the partial tensors are compatible and represent the same tensor.
        /// </summary>
        public static bool IsEquivalent(PartialTensor a, PartialTensor b)
        {
            if (a is null)
                return b is null;
            return a.IsEquivalent(b);
        }

        protected abstract bool IsEquivalent(PartialTensor other);

        /// <summary>
        /// Whether the partial tensor is fully known, i.e. the shape is known and all the partial elements are known.
        /// If so this partial tensor can be converted to a tensor.
        /// </summary>
        public abstract bool IsStatic();

        /// <summary>
        /// Returns a tensor represented by this partial tensor.
        /// If this partial tensor is not fully known returns 'null'.
        /// </summary>
        public abstract Tensor ToTensor();

        /// <summary>
        /// Returns a copy of a partial tensor.
        /// </summary>
        public abstract PartialTensor Copy();

        /// <summary>
        /// Returns a new partial tensor resulting from reshaping this partial tensor with a given dynamic shape.
        /// A single unknown output dimension can be inferred when the input shape is fully known.
        /// If 'allowZeroLength' is false then this method assumes no dimensions are 0, which allows for more general
        /// inference.
        /// </summary>
        public abstract PartialTensor Reshape(DynamicTensorShape newShape, bool allowZeroLength = true);

        /// <summary>
        /// Returns a partial tensor that is the most defined of two partial tensors known to represent equal tensors.
        /// e.g. if one of the input partial tensors has a fully defined dim or element then this will be used in the
        /// return partial tensor of this method.
        /// </summary>
        public static PartialTensor MaxDefinedPartialTensor(PartialTensor a, PartialTensor b)
        {
            return a is null ? b : a.MaxDefinedPartialTensor(b);
        }

        protected abstract PartialTensor MaxDefinedPartialTensor(PartialTensor other);

        public abstract void CopyElement(int dstIndex, PartialTensor src, int srcIndex);

        /// <summary>
        /// Returns a string that represents the `PartialTensor`.
        /// </summary>
        /// <returns>The string representation of the `PartialTensor`.</returns>
        public override string ToString()
        {
            return ToString(p => "d" + p);
        }

        internal abstract string ToString(Func<byte, string> GetParamName);
    }

    /// <summary>
    /// Represents a PartialTensor with a given type T
    /// </summary>
    class PartialTensor<T> : PartialTensor where T : unmanaged
    {
        const int k_MaxLength = TensorShape.maxRank * 2;
        PartialTensorElement<T>[] m_Elements;

        public override int length => m_Elements.Length;

        public override bool isPartiallyKnown => m_Elements != null;

        /// <summary>
        /// Initializes and returns a partial tensor with the specified `dataType` and 'shape'.
        /// If the shape is small enough unknown partial tensor elements are tracked and 'isPartiallyKnown' returns 'true'.
        /// </summary>
        public PartialTensor(DynamicTensorShape shape)
        {
            m_DataType = AllocatorUtils.ToDataType<T>();
            m_Shape = shape;
            if (shape.IsStatic() && shape.Length() <= k_MaxLength)
                m_Elements = new PartialTensorElement<T>[shape.Length().value];
        }

        /// <summary>
        /// Initializes and returns a partial tensor with the specified `dataType` and unknown shape.
        /// </summary>
        public PartialTensor()
            : this(DynamicTensorShape.DynamicRank) { }

        /// <summary>
        /// Initializes and returns a partial tensor from a given 'tensor'. The data type shape and potential partial
        /// tensor elements are inferred from the given 'tensor'.
        /// </summary>
        public static PartialTensor<T> FromTensor(Tensor<T> tensor)
        {
            var partialTensor = new PartialTensor<T>(new DynamicTensorShape(tensor.shape));
            if (!partialTensor.isPartiallyKnown)
                return partialTensor;

            var valueArray = tensor.DownloadToArray();
            for (var i = 0; i < partialTensor.length; i++)
                partialTensor[i] = PartialTensorElement<T>.Value(valueArray[i]);

            return partialTensor;
        }

        /// <inheritdoc/>
        public override PartialTensor Copy()
        {
            var ret = new PartialTensor<T>(m_Shape);
            if (m_Elements is null)
                return ret;
            for (var i = 0; i < m_Elements.Length; i++)
                ret.m_Elements[i] = m_Elements[i];
            return ret;
        }

        /// <inheritdoc/>
        protected override PartialTensor MaxDefinedPartialTensor(PartialTensor other)
        {
            if (other == null)
                return this;
            if (!isPartiallyKnown && !other.isPartiallyKnown)
                return new PartialTensor<T>(DynamicTensorShape.MaxDefinedShape(shape, other.shape));
            if (!isPartiallyKnown)
                return other;
            if (!other.isPartiallyKnown)
                return this;
            Logger.AssertIsTrue(length == other.length, "InputError: incompatible tensors");
            var maxDefinedPartialTensor = new PartialTensor<T>(shape);
            for (var i = 0; i < maxDefinedPartialTensor.length; i++)
            {
                maxDefinedPartialTensor[i] = PartialTensorElement<T>.MaxDefinedElement(this[i], (other as PartialTensor<T>)[i]);
            }

            return maxDefinedPartialTensor;
        }

        /// <inheritdoc/>
        public override PartialTensor Reshape(DynamicTensorShape newShape, bool allowZeroLength = true)
        {
            if (!newShape.hasRank)
                return new PartialTensor<T>(newShape);

            if (!newShape.IsStatic())
            {
                DynamicTensorShape.ReduceCommonFactors(shape, newShape, out var reducedPrev, out var reducedNew, !allowZeroLength);
                var reducedPrevLength = reducedPrev.Length();

                if (!reducedPrevLength.isUnknown)
                {
                    var reducedPrevLengthNonZero = DynamicTensorDim.One;
                    for (var i = 0; i < reducedPrev.rank; i++)
                    {
                        if (!(reducedPrev[i] == DynamicTensorDim.Zero))
                            reducedPrevLengthNonZero *= reducedPrev[i];
                    }

                    var isZero = false;
                    var cumNonZeroValue = 1;
                    var numUnknowns = 0;
                    var unknownIndex = 0;
                    for (var i = 0; i < reducedNew.rank; i++)
                    {
                        if (reducedNew[i].isValue)
                        {
                            if (reducedNew[i].value == 0)
                                isZero = true;
                            else
                                cumNonZeroValue *= reducedNew[i].value;
                            continue;
                        }

                        numUnknowns++;
                        unknownIndex = i;
                    }

                    if (cumNonZeroValue == 1 && numUnknowns == 1)
                    {
                        if (reducedPrevLength == DynamicTensorDim.Zero && !isZero)
                            newShape[unknownIndex] = DynamicTensorDim.Zero;
                        else
                            newShape[unknownIndex] = reducedPrevLengthNonZero;
                    }
                }
            }

            var reshapedPartialTensor = new PartialTensor<T>(newShape);
            if (!reshapedPartialTensor.isPartiallyKnown || !isPartiallyKnown)
                return reshapedPartialTensor;

            for (var i = 0; i < reshapedPartialTensor.length; i++)
                reshapedPartialTensor.m_Elements[i] = m_Elements[i];

            return reshapedPartialTensor;
        }

        /// <inheritdoc/>
        public override bool IsStatic()
        {
            if (!isPartiallyKnown)
                return false;

            for (var i = 0; i < m_Elements.Length; i++)
            {
                if (!m_Elements[i].isValue)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets or sets the element of a given index from the flattened array of partial tensor elements.
        /// </summary>
        public PartialTensorElement<T> this[int d0]
        {
            get
            {
                if (m_Elements == null)
                    return PartialTensorElement<T>.Unknown;
                Logger.AssertIsTrue(d0 < m_Elements.Length, "InputError: index out of bounds");
                return m_Elements[d0];
            }
            set
            {
                if (m_Elements != null)
                    m_Elements[d0] = value;
            }
        }

        public PartialTensorElement<T> this[PartialTensorElement<int> d0] => !d0.isValue ? PartialTensorElement<T>.Unknown : this[d0.value];

        /// <inheritdoc/>
        public override Tensor ToTensor()
        {
            return new Tensor<T>(shape.ToTensorShape(), ToArray());
        }

        /// <summary>
        /// Returns a array represented by this partial tensor.
        /// If this partial tensor is not fully known returns 'null'.
        /// </summary>
        public T[] ToArray()
        {
            if (!IsStatic())
                return null;

            var values = new T[length];
            for (var i = 0; i < length; i++)
                values[i] = (T)Convert.ChangeType(m_Elements[i].value, typeof(T));
            return values;
        }

        /// <inheritdoc/>
        public override void CopyElement(int dstIndex, PartialTensor src, int srcIndex)
        {
            Set(dstIndex, src.Get<T>(srcIndex));
        }

        /// <inheritdoc/>
        protected override bool IsEquivalent(PartialTensor other)
        {
            if (other == null)
                return false;
            if (dataType != other.dataType)
                return false;
            if (!shape.hasRank && !other.shape.hasRank)
                return true;
            if (shape != other.shape)
                return false;
            if (!isPartiallyKnown && !other.isPartiallyKnown)
                return true;
            for (var i = 0; i < length; i++)
            {
                if (this[i] != (other as PartialTensor<T>)[i])
                    return false;
            }

            return true;
        }

        internal override string ToString(Func<byte, string> GetParamName)
        {
            var sb = new StringBuilder();
            if (isPartiallyKnown)
            {
                if (m_Shape.rank > 0)
                    sb.Append("[");
                for (var i = 0; i < m_Elements.Length; i++)
                {
                    if (i != 0)
                        sb.Append(", ");
                    var element = this[i];
                    sb.Append(element.ToString(GetParamName));
                }
                if (m_Shape.rank > 0)
                    sb.Append("]");
            }
            else
            {
                sb.Append(dataType.ToString().ToLower());
                sb.Append(m_Shape.ToString(GetParamName));
            }
            return sb.ToString();
        }
    }
}
