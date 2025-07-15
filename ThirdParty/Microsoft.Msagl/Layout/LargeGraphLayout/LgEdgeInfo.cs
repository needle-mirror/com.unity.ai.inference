using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Msagl.Core.Geometry;
using Microsoft.Msagl.Core.Geometry.Curves;
using Microsoft.Msagl.Core.Layout;

namespace Microsoft.Msagl.Layout.LargeGraphLayout {
    /// <summary>
    ///
    /// </summary>
    internal class LgEdgeInfo:LgInfoBase {
        /// <summary>
        /// to string for debugging
        /// </summary>
        /// <returns></returns>
        public override string ToString() {
            return String.Format("zoom lvl={0:F2}", ZoomLevel);
        }



        /// <summary>
        ///
        /// </summary>
        /// <param name="edge"></param>
        public LgEdgeInfo(Edge edge) {
            Edge = edge;
            ZoomLevel = int.MaxValue;
        }

        /// <summary>
        ///
        /// </summary>
        public Edge Edge { get; set; }

//        /// <summary>
//        /// those need to be set to correctly draw an edge of the level: ActiveGeometries include ICurves and Arrowheads ( labels todo?)
//        /// </summary>
//        public List<EdgePartialGeometryOnLevel> EdgeGeometriesByLevels; //EdgeGeometriesByLevels[i] corresponds to level i,

    }
}