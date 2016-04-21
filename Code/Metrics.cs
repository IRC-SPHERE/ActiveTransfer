//
// Metrics.cs
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
    using System;
    using System.Linq;
    using MicrosoftResearch.Infer.Distributions;
    using SphereEngine;

    /// <summary>
    /// Metrics.
    /// </summary>
    public class Metrics
    {
        /// <summary>
        /// The log prob of truth.
        /// </summary>
        private double[] logProbOfTruth;

        /// <summary>
        /// The accuracy.
        /// </summary>
        private double[] accuracy;

        /// <summary>
        /// The true labels.
        /// </summary>
        private bool[] trueLabels;

        /// <summary>
        /// The estimates.
        /// </summary>
        private Bernoulli[] estimates;

        /// <summary>
        /// Gets or sets the true labels.
        /// </summary>
        /// <value>The true labels.</value>
        public bool[] TrueLabels
        {
            get {
                return trueLabels;
            }
            set {
                trueLabels = value;
                logProbOfTruth = null;
                accuracy = null;
            }
        }

        /// <summary>
        /// Gets or sets the estimates.
        /// </summary>
        /// <value>The estimates.</value>
        public Bernoulli[] Estimates
        {
            get {
                return estimates;
            }
            set {
                estimates = value;
                logProbOfTruth = null;
                accuracy = null;
            }
        }

        /// <summary>
        /// Gets the predictions.
        /// </summary>
        /// <value>The predictions.</value>
        public Prediction[] Predictions { get { return TrueLabels.Zip(Estimates, (t, e) => new Prediction { Truth = t, Estimate = e }).ToArray(); } }

        /// <summary>
        /// Gets the estimated labels.
        /// </summary>
        /// <value>The estimated labels.</value>
        public double[] EstimatedLabels { get { return Predictions == null || Predictions.Length == 0 ? null : Estimates.Select(ia => ia.GetProbTrue()).ToArray(); } }

        /// <summary>
        /// Gets the mean squared error.
        /// </summary>
        /// <value>The mean squared error.</value>
        public double MeanSquaredError { get { return Predictions == null || Predictions.Length == 0 ? double.NaN : Predictions.Average(p => Math.Pow((p.Truth ? 1.0 : 0.0) - p.ConditionalDensity, 2)); } }

        /// <summary>
        /// Gets the sum of the log prob of truth.
        /// </summary>
        /// <value>The log prob of truth.</value>
        public double SumLogProbOfTruth { get { return Predictions == null || Predictions.Length == 0 ? double.NaN : Predictions.Sum(p => p.LogProbOfTruth); } }

        /// <summary>
        /// Gets the average accuracy.
        /// </summary>
        /// <value>The accuracy.</value>
        public double AverageAccuracy { get { return Predictions == null || Predictions.Length == 0 ? double.NaN : Predictions.Average(p => p.Correct ? 1.0 : 0.0); } }

        /// <summary>
        /// Gets or sets the name.
        /// </summary>
        /// <value>The name.</value>
        public string Name { get; set; }

        /// <summary>
        /// Gets the cumulative log prob of truth.
        /// </summary>
        /// <value>The cumulative log prob of truth.</value>
        public double[] CumulativeLogProbOfTruth { get { return Predictions == null || Predictions.Length == 0 ? null : Predictions.CumulativeSum(p => p.LogProbOfTruth).ToArray(); } }

        /// <summary>
        /// Gets the cumulative accuracy.
        /// </summary>
        /// <value>The cumulative accuracy.</value>
        public double[] CumulativeAccuracy { get { return Predictions == null || Predictions.Length == 0 ? null : Predictions.CumulativeAverage(p => Convert.ToDouble(p.Correct)).ToArray(); } }

        /// <summary>
        /// Gets the cumulative brier score.
        /// </summary>
        /// <value>The cumulative brier score.</value>
        public double[] CumulativeBrierScore { get { return Predictions == null || Predictions.Length == 0 ? null : Predictions.CumulativeAverage(p => p.BrierScore).ToArray(); } }

        /// <summary>
        /// Gets the brier score.
        /// </summary>
        /// <value>The brier score.</value>
        public double BrierScore { get { return Predictions == null || Predictions.Length == 0 ? double.NaN : Predictions.Average(ia => ia.BrierScore); } }

        /// <summary>
        /// Prints the summary.
        /// </summary>
        public void PrintSummary()
        {
            // Console.WriteLine("Class distribution: {0:N2}", (double)TrueLabels.Count(ia => ia) / TrueLabels.Length);
            // Console.WriteLine("MSE {0:N2}, Error rate {1:N2}, Log prob of truth {2:N2}", MeanSquaredError, ErrorRate, LogProbOfTruth);
            Console.WriteLine("{3:20}: MSE {0:N2}, Error rate {1:N2}, Log prob of truth {2:N2}", MeanSquaredError, 1 - AverageAccuracy, SumLogProbOfTruth, Name);
        }

        /// <summary>
        /// Prints the cumulative summary.
        /// </summary>
        public void PrintCumulativeSummary()
        {
            Console.WriteLine("Log_prob_{0} = [{1}]", Name, String.Join(",", CumulativeLogProbOfTruth.Select(ia => ia.ToString("N2"))));
            Console.WriteLine("Accuracy_{0} = [{1}]", Name, String.Join(",", CumulativeAccuracy.Select(ia => ia.ToString("N2"))));
        }

        /// <summary>
        /// Prediction.
        /// </summary>
        public class Prediction
        {
            /// <summary>
            /// Gets or sets a value indicating whether this <see cref="ActiveTransfer.Metrics+Prediction"/> is truth.
            /// </summary>
            /// <value><c>true</c> if truth; otherwise, <c>false</c>.</value>
            public bool Truth { get; set; }

            /// <summary>
            /// Gets or sets the estimate.
            /// </summary>
            /// <value>The estimate.</value>
            public Bernoulli Estimate { get; set; }

            /// <summary>
            /// Gets the conditional density.
            /// </summary>
            /// <value>The conditional density.</value>
            public double ConditionalDensity { get { return Estimate.GetMean(); } } // GetLogProbTrue(); } }

            /// <summary>
            /// Gets the log prob of truth.
            /// </summary>
            /// <value>The log prob of truth.</value>
            public double LogProbOfTruth { get { return Estimate.GetLogProb(Truth); } }

            /// <summary>
            /// Gets a value indicating whether this <see cref="ActiveTransfer.Metrics+Prediction"/> is correct.
            /// </summary>
            /// <value><c>true</c> if correct; otherwise, <c>false</c>.</value>
            public bool Correct { get { return Truth == (ConditionalDensity > 0.5); } }

            /// <summary>
            /// Gets the brier score.
            /// </summary>
            /// <value>The brier score.</value>
            public double BrierScore { get { return Math.Pow((Truth ? 1.0 : 0.0) - ConditionalDensity, 2); } }
        }
    }
}

