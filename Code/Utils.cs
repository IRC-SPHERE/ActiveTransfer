//
// Utils.cs
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
	using System.Collections.Generic;
	using SphereEngine;
	using System.Linq;
	using MicrosoftResearch.Infer.Distributions;
	using PythonPlotter;

	public static class Utils
	{
		/// <summary>
		/// Creates the learners.
		/// </summary>
		/// <returns>The learners.</returns>
		/// <param name="dataSet">Data set.</param>
		/// <typeparam name="TLearner">The type of the learner.</typeparam>
		public static IList<IActiveLearner> CreateLearners<TLearner>(DataSet dataSet, BinaryModel trainModel, BinaryModel testModel, BinaryModel evidenceModel, bool reverse = false)
			where TLearner : IActiveLearner
		{
			var learners = new List<IActiveLearner>();
			for (int i = 0; i < dataSet.NumberOfResidents; i++)
			{
				var learner = (IActiveLearner)Activator.CreateInstance<TLearner>();
				learner.DataSet = dataSet.GetSubSet(i);
				if (learner is IReversableLearner && reverse)
				{
					((IReversableLearner)learner).Reversed = true;
				}
        
				if (learner is ActiveLearner)
				{
					var activeLearner = (ActiveLearner)learner;
					activeLearner.TrainModel = trainModel;
					activeLearner.TestModel = testModel;
					activeLearner.RiskMatrix = new[,] { { 0.0, 1.0 }, { 1.0, 0.0 } };
					activeLearner.Gains = Enumerable.Range(0, 2).Select(ii => 1.0 / 3.0).ToArray();
					activeLearner.Costs = Enumerable.Range(0, 2).Select(ii => 1.0).ToArray();
				}
				else if (learner is ActiveEvidence)
				{
					var activeLearner = (ActiveEvidence)learner;
					activeLearner.TrainModel = trainModel;
					activeLearner.TestModel = testModel;
                    activeLearner.EvidenceModel = evidenceModel;
				}

				learners.Add(learner);
			}

			return learners;
		}

		/// <summary>
		/// Plots the cumulative metrics.
		/// </summary>
		/// <param name="experiments">Experiments.</param>
		/// <param name="showPlots">If set to <c>true</c> show plots.</param>
		public static void PlotCumulativeMetrics(IList<Experiment> experiments, string title, bool showPlots = true)
		{
			// Plot cumulative log prob
			// PlotCumulativeMetric(experiments, "Log Prob of Truth", ia => ia.AverageLogProb, ia => ia.StdDevLogProb, null, LegendPosition.Best, showPlots);

			// Plot cumulative accuracy
			PlotCumulativeMetric(experiments, title, "Accuracy", ia => ia.AverageAccuracy, ia => ia.StdDevAccuracy, 5, new Tuple<double, double>(0, 1), LegendPosition.LowerRight, showPlots);

			// Plot cumulative brier score
			// PlotCumulativeMetric(experiments, "Brier Score", ia => ia.AverageBrierScore, ia => ia.StdDevBrierScore, new Tuple<double, double>(0, 1), LegendPosition.LowerRight, showPlots);
		}

		/// <summary>
		/// Plots the cumulative metric.
		/// </summary>
		/// <param name="experiments">Experiments.</param>
		/// <param name="metricName">Metric name.</param>
		/// <param name="selector">Selector.</param>
		/// <param name="errorSelector">Error selector.</param>
        /// <param name="skip">Number to skip at start.</param>
		/// <param name="yLim">Y limits.</param>
		/// <param name="legendPosition">Legend position.</param>
		/// <param name="showPlots">If set to <c>true</c> show plots.</param>
		public static void PlotCumulativeMetric(
            IList<Experiment> experiments,
			string title,
			string metricName,
			Func<MetricsCollection, IList<double>> selector, 
			Func<MetricsCollection, IList<double>> errorSelector,
            int skip,
			Tuple<double, double> yLim = null,
			LegendPosition legendPosition = LegendPosition.Best,
			bool showPlots = true)
		{
            // Plot cumulative log prob
			var plotter = new Plotter
				{
					Title = "Cumulative " + metricName,
					XLabel = "# instances",
					YLabel = metricName,
					Series = experiments.Select(
						ia => new ErrorLineSeries { 
                            X = Enumerable.Range(0, ia.Metrics.MinimumLength).Select(x => x + 1.0).Skip(skip), 
                            Y = selector(ia.Metrics).Skip(skip), 
                            ErrorValues = errorSelector(ia.Metrics).Skip(skip), 
                            Label = ia.Name 
                            }).ToArray(),
					YLim = yLim,
					LegendPosition = legendPosition,
					ScriptName = title.ToLower().Replace(" ", "-") + "-cumulative-" + metricName.ToLower(), // + DateTime.UtcNow.ToString("s"),
					Show = showPlots
				};

			plotter.Plot();
		}

		/// <summary>
		/// Plots the holdout metrics.
		/// </summary>
		/// <param name="experiments">Experiments.</param>
		/// <param name="showPlots">If set to <c>true</c> show plots.</param>
		public static void PlotHoldoutMetrics(IList<Experiment> experiments, string title, string metricName = "", bool showPlots = true)
		{
			// Plot holdout accuracy
            PlotHoldoutMetric(
                experiments,
                title,
                "Accuracy" + (string.IsNullOrEmpty(metricName) ? string.Empty : string.Format(" ({0})", metricName)), 
                ia => ia.AverageAccuracy, 
                ia => ia.StdDevAccuracy, 
                new Tuple<double, double>(0, 1), 
                LegendPosition.LowerRight,
                showPlots);
            
            // Plot holdout brier score
            // PlotHoldoutMetric(
            //     experiments, 
            //     "Brier" + (string.IsNullOrEmpty(name) ? string.Format(" ({0})", name) : string.Empty), 
            //     ia => ia.AverageBrierScore, 
            //     ia => ia.StdDevBrierScore, 
            //     new Tuple<double, double>(0, 0.5), 
            //     LegendPosition.LowerRight,
            //     showPlots);
        }

		/// <summary>
		/// Plots the holdout metric.
		/// </summary>
		/// <param name="experiments">Experiments.</param>
		/// <param name="metricName">Metric name.</param>
		/// <param name="selector">Selector.</param>
		/// <param name="errorSelector">Error selector.</param>
		/// <param name="yLim">Y lim.</param>
		/// <param name="legendPosition">Legend position.</param>
		/// <param name="showPlots">If set to <c>true</c> show plots.</param>
		public static void PlotHoldoutMetric(
			IList<Experiment> experiments,
            string title,
			string metricName,
			Func<HoldoutMetricsCollection, IList<double>> selector, 
			Func<HoldoutMetricsCollection, IList<double>> errorSelector,
			Tuple<double, double> yLim = null,
			LegendPosition legendPosition = LegendPosition.Best,
			bool showPlots = true)
		{
			// Plot cumulative log prob
			var plotter = new Plotter
				{
					Title = "Hold-out " + metricName,
					XLabel = "# instances",
					YLabel = metricName,
					Series = experiments.Select(
						ia => new ErrorLineSeries { X = selector(ia.HoldoutMetrics), ErrorValues = errorSelector(ia.HoldoutMetrics), Label = ia.Name }).ToArray(),
					YLim = yLim,
					LegendPosition = legendPosition,
					ScriptName = title.ToLower().Replace(" ", "-") + "-holdout-" + metricName.ToLower(), // + DateTime.UtcNow.ToString("s"),
					Show = showPlots
				};

			plotter.Plot();
		}

		/// <summary>
		/// Prints the weights.
		/// </summary>
		/// <param name="names">Names.</param>
		/// <param name="posteriorWeights">Posterior weights.</param>
		/// <param name="weights">Weights.</param>
		public static void PrintWeights(IList<string> names, Gaussian[][] posteriorWeights, double[][] weights) // , Gaussian[] posteriorThresholds)
		{
			for (int i = 0; i < weights.Length; i++)
			{
				for (int j = 0; j < weights[i].Length; j++)
				{
					Console.WriteLine("{0}, Feature {1}, Posterior {2}, True {3}",
						names[i], j, posteriorWeights[i][j], weights[i][j]);
				}

				//				Console.WriteLine("Resident {0}, Threshold, Posterior {1}", i, posteriorThresholds[i]);
			}	
		}

		/// <summary>
		/// Prints the weights.
		/// </summary>
		/// <param name="names">Names.</param>
		/// <param name="posteriorWeights">Posterior weights.</param>
		/// <param name="weights">Weights.</param>
		public static void PrintWeights(IList<string> names, Gaussian[][][] posteriorWeights, Gaussian[][][] weights)
		{
			for (int i = 0; i < weights.Length; i++)
			{
				for (int j = 0; j < weights[i].Length; j++)
				{
					for (int k = 0; k < weights[i][j].Length; k++)
					{
						Console.WriteLine("{0}, Activity {1}, Feature {2}, Posterior {3}, True {4}",
							names[i], j, k, posteriorWeights[i][j][k], weights[i][j][k]);
					}
				}
			}	
		}

		/// <summary>
		/// Prints the weight priors.
		/// </summary>
		/// <param name="posteriors">Posteriors.</param>
		/// <param name="communityWeights">Community weights.</param>
		public static void PrintWeightPriors(Marginals posteriors, Gaussian[] communityWeights)
		{
			for (int i = 0; i < posteriors.WeightMeans.Length; i++)
			{
				Console.WriteLine("Feature {0} Posterior Mean {1}, Posterior Precision {2}, Weight mean {3}, weight precision {4}",
					i, posteriors.WeightMeans[i], posteriors.WeightPrecisions[i], communityWeights[i].GetMean(), communityWeights[i].Precision);
			}
		}

		/// <summary>
		/// Prints the predictions.
		/// </summary>
		/// <param name="names">Names.</param>
		/// <param name="posteriorActivities">Posterior activities.</param>
		/// <param name="labels">Labels.</param>
		public static void PrintPredictions(IList<string> names, Discrete[][] posteriorActivities, int[][] labels)
		{
			for (int i = 0; i < posteriorActivities.Length; i++)
			{
				for (int j = 0; j < posteriorActivities[i].Length; j++)
				{
					Console.WriteLine("{0} {1}, True {2}, Predicted {3}", i, names[j], labels[i][j], posteriorActivities[i][j].GetMode());
				}
			}
		}

		/// <summary>
		/// Prints the prediction.
		/// </summary>
		/// <param name="index">Index.</param>
		/// <param name="posteriorActivity">Posterior activity.</param>
		/// <param name="label">If set to <c>true</c> label.</param>
		/// <param name="score">Score.</param>
		public static void PrintPrediction(int index, Bernoulli posteriorActivity, bool label, double score)
		{
			Console.WriteLine("{0,3}, Score: {1,4:N2}, Bayes Point {2:N2}, Label {3,5}, Predicted {4,5}, Correct {5,5}", 
				index, score, posteriorActivity.GetMean(), label, posteriorActivity.GetMode(), label == posteriorActivity.GetMode());
		}

		/// <summary>
		/// Prints the predictions.
		/// </summary>
		/// <param name="posteriorActivities">Posterior activities.</param>
		/// <param name="labels">Labels.</param>
		/// <param name="scores">Scores.</param>
		public static void PrintPredictions(IList<Bernoulli> posteriorActivities, IList<bool> labels, IList<double> scores)
		{
			for (int i = 0; i < posteriorActivities.Count; i++)
			{
				PrintPrediction(i, posteriorActivities[i], labels[i], scores[i]);
			}
		}

		/// <summary>
		/// Plots the posteriors.
		/// </summary>
		/// <param name="weights">Weights.</param>
		/// <param name="trueWeights">True weights.</param>
		/// <param name="title">Title.</param>
		public static void PlotPosteriors(Gaussian[] weights, double[] trueWeights, string title, string ylabel = "resident", bool show = true)
		{
			var weightMeans = weights.Select(ia => ia.GetMean());
			var weightStdDevs = weights.Select(ia => Math.Sqrt(1.0 / ia.GetVariance()));

			var series = trueWeights == null
				? new[]
					{ 
						new BarSeries<double> { Label = "Posterior", Width = 0.35, Horizontal = true, IndependentValues = Enumerable.Range(0, weights.Length).Select(ia => (double)ia), DependentValues = weightMeans, ErrorValues = weightStdDevs },
					}
				: new[]
					{ 
						new BarSeries<double> { Label = "Posterior", Width = 0.35, Horizontal = true, IndependentValues = Enumerable.Range(0, weights.Length).Select(ia => (double)ia), DependentValues = weightMeans, ErrorValues = weightStdDevs },
						new BarSeries<double> { Label = "True", Width = 0.35, Horizontal = true, Color = "r", IndependentValues = Enumerable.Range(0, weights.Length).Select(ia => (double)ia + 0.35), DependentValues = trueWeights }
					};

			var plotter = new Plotter
				{
					Title = title,
					XLabel = "weight",
					YLabel = ylabel,
					Series = series,
					ScriptName = title,
					Show = show
				};

			plotter.Plot();
		}

		/// <summary>
		/// Plots the posteriors.
		/// </summary>
		/// <param name="means">Means.</param>
		/// <param name="precisions">Precisions.</param>
		/// <param name="trueWeights">True weights.</param>
		/// <param name="title">Title.</param>
		/// <param name="ylabel">Ylabel.</param>
		public static void PlotPosteriors(Gaussian[] means, Gamma[] precisions, double[] trueWeights, string title, string ylabel = "resident", bool show = true)
		{
			PlotPosteriors(means.Zip(precisions, (m, p) => new Gaussian(m.GetMean(), 1.0 / p.GetMean())).ToArray(), trueWeights, title, ylabel, show);
		}

		/// <summary>
		/// Plots the posteriors.
		/// </summary>
		/// <param name="precisions">Precisions.</param>
		/// <param name="truePrecisions">True precisions.</param>
		/// <param name="title">Title.</param>
		public static void PlotPosteriors(Gamma[] precisions, double[] truePrecisions, string title, string ylabel = "resident", bool show = true)
		{
			var weightMeans = precisions.Select(ia => ia.GetMean());
			var weightStdDevs = precisions.Select(ia => Math.Sqrt(ia.GetVariance()));

			var series = truePrecisions == null
				? new[]
					{ 
						new BarSeries<double> { Label = "Posterior", Width = 0.35, Horizontal = true, IndependentValues = Enumerable.Range(0, precisions.Length).Select(ia => (double)ia), DependentValues = weightMeans, ErrorValues = weightStdDevs },
					}
				: new[]
					{ 
						new BarSeries<double> { Label = "Posterior", Width = 0.35, Horizontal = true, IndependentValues = Enumerable.Range(0, precisions.Length).Select(ia => (double)ia), DependentValues = weightMeans, ErrorValues = weightStdDevs },
						new BarSeries<double> { Label = "True", Width = 0.35, Horizontal = true, Color = "r", IndependentValues = Enumerable.Range(0, precisions.Length).Select(ia => (double)ia + 0.35), DependentValues = truePrecisions }
					};

			var plotter = new Plotter
				{
					Title = title,
					XLabel = "weight",
					YLabel = ylabel,
					Series = series,
					ScriptName = title,
					Show = show
				};

			plotter.Plot();
		}


		/// <summary>
		/// Plots the posteriors.
		/// </summary>
		/// <param name="weights">Weights.</param>
		/// <param name="trueWeights">True weights.</param>
		public static void PlotPosteriors(Gaussian[][] weights, double[][] trueWeights, bool show = true)
		{
			for (int i = 0; i < weights.Length; i++)
			{
				PlotPosteriors(weights[i], trueWeights[i], string.Format("Resident {0} Weights", i), "feature", show);
			}
		}
	}
}

