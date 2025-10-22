using System;
using System.Collections.Generic;
using System.IO;

namespace Unity.InferenceEngine.Editor
{
    /// <summary>
    /// Base class for a converter from an ONNX or LiteRT model to Sentis format.
    /// </summary>
    abstract class ModelConverterBase
    {
        // Configuration
        protected string m_DirectoryPath;
        protected string m_FilePath;

        /// <summary>
        /// Initializes and returns a `ModelConverter` for this file.
        /// </summary>
        /// <param name="filePath">The path of the asset to convert.</param>
        protected ModelConverterBase(string filePath)
        {
            m_FilePath = filePath;
            m_DirectoryPath = Path.GetDirectoryName(m_FilePath);
        }

        /// <summary>
        /// Converts a model to a Sentis `Model` object.
        /// </summary>
        /// <returns>The converted Sentis model.</returns>
        public abstract Model Convert();

        // Logging helpers
        protected void Warn(WarningType severity, string message)
        {
            Warnings.Add(new ImporterWarning(message, severity));
        }

        protected T Warn<T>(WarningType severity, string message, T defaultValue)
        {
            Warnings.Add(new ImporterWarning(message, severity));
            return defaultValue;
        }

        /// <summary>
        /// The warnings from the model importer.
        /// </summary>
        protected internal List<ImporterWarning> Warnings { get; } = new();

        /// <summary>
        /// Represents types of warning from the model importer.
        /// </summary>
        public enum WarningType
        {
            /// <summary>
            /// No error.
            /// </summary>
            None = 0,

            /// <summary>
            /// Information. Execution should run without errors.
            /// </summary>
            Info = 1,

            /// <summary>
            /// Warning. Execution should run, but may have issues with precision or speed.
            /// </summary>
            Warning = 2,

            /// <summary>
            /// Error. Execution won't run.
            /// </summary>
            Error = 3
        }

        /// <summary>
        /// Represents the data structure for a warning from the model importer.
        /// </summary>
        public class ImporterWarning
        {
            /// <summary>
            /// A message.
            /// </summary>
            public string Message { get; }

            /// <summary>
            /// The severity of a warning.
            /// </summary>
            public WarningType MessageSeverity { get; }

            /// <summary>
            /// Initializes and returns an instance of `ImporterWarning`.
            /// </summary>
            /// <param name="severity">The severity of the warning as a `WarningType`</param>
            /// <param name="msg">The message text of the warning</param>
            public ImporterWarning(string msg, WarningType severity)
            {
                Message = msg;
                MessageSeverity = severity;
            }
        }
    }

    /// <summary>
    /// Represents an exception during the import of a model.
    /// </summary>
    class ImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `ImportException`.
        /// </summary>
        /// <param name="message">message</param>
        protected ImportException(string message)
            : base(message) { }
    }

    /// <summary>
    /// Represents an exception during the import of a layer.
    /// </summary>
    class LayerImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `LayerImportException`.
        /// </summary>
        /// <param name="message">message</param>
        protected LayerImportException(string message)
            : base(message) { }
    }
}
