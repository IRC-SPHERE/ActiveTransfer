//
// MetricsCollection.cs
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
using System;
using System.Collections.Generic;
using System.Linq;
using SphereEngine;

namespace ActiveTransfer
{
	/// <summary>
	/// Metrics collection.
	/// </summary>
	public class MetricsCollection
	{
		/// <summary>
		/// The metrics.
		/// </summary>
		private readonly IList<Metrics> metrics = new List<Metrics>();

		/// <summary>
		/// Add the specified metrics.
		/// </summary>
		/// <param name="metrics">Metrics.</param>
		/// <param name="printSummary">If set to <c>true</c> print summary.</param>
		public void Add(Metrics metrics, bool printSummary = false)
		{
			this.metrics.Add(metrics);
			if (printSummary)
			{
				metrics.PrintSummary();
			}
		}
        
        /// <summary>
        /// Gets the minimum length
        /// </summary>
        public int MinimumLength { get { return metrics.Min(ia => ia.Predictions.Length); } }

		/// <summary>
		/// Gets or sets the average log prob.
		/// </summary>
		/// <value>The average log prob.</value>
		public IList<double> AverageLogProb { get; set; }

		/// <summary>
		/// Gets or sets the std dev log prob.
		/// </summary>
		/// <value>The std dev log prob.</value>
		public IList<double> StdDevLogProb { get; set; }

		/// <summary>
		/// Gets or sets the average accuracy.
		/// </summary>
		/// <value>The average accuracy.</value>
		public IList<double> AverageAccuracy { get; set; }

		/// <summary>
		/// Gets or sets the std dev accuracy.
		/// </summary>
		/// <value>The std dev accuracy.</value>
		public IList<double> StdDevAccuracy { get; set; }

		/// <summary>
		/// Gets or sets the average brier score.
		/// </summary>
		/// <value>The average brier score.</value>
		public IList<double> AverageBrierScore { get; set; }

		/// <summary>
		/// Gets or sets the std dev brier score.
		/// </summary>
		/// <value>The std dev brier score.</value>
		public IList<double> StdDevBrierScore { get; set; } 

		/// <summary>
		/// Recomputes the aggregate metrics.
		/// </summary>
		public void RecomputeAggregateMetrics()
		{
			int numberOfInstances = metrics.Min(ia => ia.Predictions.Length);

			AverageLogProb = new double[numberOfInstances];
			StdDevLogProb = new double[numberOfInstances];
			AverageAccuracy = new double[numberOfInstances];
			StdDevAccuracy = new double[numberOfInstances];
			AverageBrierScore = new double[numberOfInstances];
			StdDevBrierScore = new double[numberOfInstances];

			for (int i = 0; i < numberOfInstances; i++)
			{
				AverageLogProb[i] = metrics.Average(ia => ia.CumulativeLogProbOfTruth[i]);
				StdDevLogProb[i] = metrics.StandardDeviation(ia => ia.CumulativeLogProbOfTruth[i]);
				AverageAccuracy[i] = metrics.Average(ia => ia.CumulativeAccuracy[i]);
				StdDevAccuracy[i] = metrics.StandardDeviation(ia => ia.CumulativeAccuracy[i]);
				AverageBrierScore[i] = metrics.Average(ia => ia.CumulativeBrierScore[i]);
				StdDevBrierScore[i] = metrics.StandardDeviation(ia => ia.CumulativeBrierScore[i]);
			}
		}
	}
}

