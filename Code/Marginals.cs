//
// Marginals.cs
//
// Author:
//       Tom Diethe <tom.diethe@bristol.ac.uk>
//
// Copyright (c) 2015 University of Bristol
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace ActiveTransfer
{
    using System.Linq;
    using MicrosoftResearch.Infer.Distributions;
    using SphereEngine;

    /// <summary>
    /// Marginals.
    /// </summary>
    public class Marginals
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Marginals"/> class.
        /// </summary>
        public Marginals()
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Marginals"/> class.
        /// </summary>
        /// <param name="marginals">Marginals.</param>
        public Marginals(Marginals marginals)
        {
            Weights = marginals.Weights == null ? null : marginals.Weights.Select(ia => DistributionArrayHelpers.Copy(ia).ToArray()).ToArray();
            WeightMeans = marginals.WeightMeans == null ? null : DistributionArrayHelpers.Copy(marginals.WeightMeans).ToArray();
            WeightPrecisions = marginals.WeightPrecisions == null ? null : DistributionArrayHelpers.Copy(marginals.WeightPrecisions).ToArray();
        }

        /// <summary>
        /// Gets or sets the weights.
        /// </summary>
        public Gaussian[][] Weights { get; set; }

        /// <summary>
        /// Gets or sets the weight means.
        /// </summary>
        public Gaussian[] WeightMeans { get; set; }

        /// <summary>
        /// Gets or sets the weight precisions.
        /// </summary>
        public Gamma[] WeightPrecisions { get; set; }
    }
}

