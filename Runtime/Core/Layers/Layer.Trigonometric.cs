using System;

namespace Unity.InferenceEngine.Layers
{
    /// <summary>
    /// Represents an element-wise `Acos` trigonometric layer: f(x) = acos(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Acos : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Acos(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Acosh` trigonometric layer: f(x) = acosh(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Acosh : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Acosh(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Asin` trigonometric layer: f(x) = asin(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Asin : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Asin(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Asinh` trigonometric layer: f(x) = asinh(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Asinh : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Asinh(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Atan` trigonometric layer: f(x) = atan(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Atan : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Atan(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Atanh` trigonometric layer: f(x) = atanh(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Atanh : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Atanh(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Cos` trigonometric layer: f(x) = cos(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Cos : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Cos(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Cosh` trigonometric layer: f(x) = cosh(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Cosh : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Cosh(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Sin` trigonometric layer: f(x) = sin(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Sin : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sin(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Sinh` trigonometric layer: f(x) = sinh(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Sinh : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Sinh(X, O);
        }
    }

    /// <summary>
    /// Represents an element-wise `Tan` trigonometric layer: f(x) = tan(x).
    /// </summary>
    [Operator(category = "Trigonometric")]
    partial class Tan : Activation
    {
        internal override void Execute(ExecutionContext ctx)
        {
            var X = ctx.storage.GetTensor(inputs[0]) as Tensor<float>;
            var O = ctx.storage.AllocateTensorAndStore(outputs[0], X.shape, DataType.Float, ctx.backend.backendType) as Tensor<float>;
            if (O.shape.HasZeroDims())
                return;
            ctx.backend.Tan(X, O);
        }
    }
}
