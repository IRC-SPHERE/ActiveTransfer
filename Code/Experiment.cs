
//
// Experiments.cs
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
using MicrosoftResearch.Infer.Distributions;
using SphereEngine;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ActiveTransfer
{
	/// <summary>
	/// Experiment.
	/// </summary>
	public class Experiment
	{
		/// <summary>
		/// Gets or sets the train model.
		/// </summary>
		public BinaryModel TrainModel { get; set; }

		/// <summary>
		/// Gets or sets the test model.
		/// </summary>
		public BinaryModel TestModel { get; set; }

        /// <summary>
		/// Gets or sets the evidence model.
		/// </summary>
		public BinaryModel EvidenceModel { get; set;}

		/// <summary>
		/// Gets or sets the active learners.
		/// </summary>
		public IList<IActiveLearner> ActiveLearners { get; set; }

		/// <summary>
		/// Gets or sets the metrics.
		/// </summary>
		public MetricsCollection Metrics { get; set; }

		/// <summary>
		/// Gets or sets the holdout metrics.
		/// </summary>
		public HoldoutMetricsCollection HoldoutMetrics { get; set; }

		/// <summary>
		/// Gets or sets the name.
		/// </summary>
		public string Name { get; set; }

		/// <summary>
		/// Gets or sets the posteriors.
		/// </summary>
		public Marginals Posteriors { get; set; }

		/// <summary>
		/// Gets or sets the individual posteriors.
		/// </summary>
		public IList<Marginals> IndividualPosteriors { get; set; }

		/// <summary>
		/// Gets or sets the posterior activities.
		/// </summary>
		public Bernoulli[][] PosteriorActivities { get; set; }

		/// <summary>
		/// Gets or sets the holdout posterior activities.
		/// </summary>
		public Bernoulli[][][] HoldoutPosteriorActivities { get; set; }

		/// <summary> 
		/// Runs the batch.
		/// </summary>
		/// <param name="dataSet">Data set.</param>
		/// <param name="priors">Priors.</param>
		public void RunBatch(DataSet dataSet, Marginals priors, int niter = 1)
		{
//			Posteriors = priors; 
//			for (int rr = 0; rr < dataSet.NumberOfResidents; ++rr)
//			{
//				Posteriors = TrainModel.Train(dataSet.GetSubSet(rr), Posteriors);
//			}
			Posteriors = TrainModel.Train(dataSet, priors, niter);
		}

		/// <summary>
		/// Runs the online experiment.
		/// </summary>
		/// <param name="dataSet">Data set.</param>
		/// <param name="holdoutSet">Holdout set.</param>
		/// <param name="priors">Priors.</param>
		public void RunOnline(DataSet dataSet, DataSet holdoutSet, Marginals priors)
		{
			using (new CodeTimer("Running online experiment: " + Name))
			{
				Console.WriteLine();

				Metrics = new MetricsCollection();
				HoldoutMetrics = new HoldoutMetricsCollection { Metrics = new Metrics[dataSet.NumberOfResidents][] };

				PosteriorActivities = new Bernoulli[dataSet.NumberOfResidents][];
				HoldoutPosteriorActivities = new Bernoulli[dataSet.NumberOfResidents][][];
				IndividualPosteriors = new Marginals[dataSet.NumberOfResidents];

				var accuracy = new double[dataSet.NumberOfResidents][];

				for (int i = 0; i < dataSet.NumberOfResidents; i++)
				{
					var collection = new List<Metrics>();
					HoldoutPosteriorActivities[i] = new Bernoulli[dataSet.NumberOfInstances[i]][];
					accuracy[i] = new double [dataSet.NumberOfInstances[i]];

					IndividualPosteriors[i] = new Marginals(priors);
					PosteriorActivities[i] = new Bernoulli[dataSet.NumberOfInstances[i]];

					for (int j = 0; j < dataSet.NumberOfInstances[i]; j++)
					{
						var datum = dataSet.GetSubSet(i, j);
						PosteriorActivities[i][j] = TestModel.Test(datum, IndividualPosteriors[i])[0][0];
						HoldoutPosteriorActivities[i][j] = TestModel.Test(holdoutSet.GetSubSet(i), IndividualPosteriors[i])[0];

						// Test on holdout set
						var holdoutMetrics = new Metrics { Name = Name, Estimates = HoldoutPosteriorActivities[i][j], TrueLabels = holdoutSet.Labels[i] };
						accuracy[i][j] = holdoutMetrics.AverageAccuracy;

						// PrintPrediction(i, temp[0][0], testLabels[0][i], testScores[0][i]);

						// Now retrain using this label
						IndividualPosteriors[i] = TrainModel.Train(datum, IndividualPosteriors[i], 10);

						collection.Add(holdoutMetrics);
					}

					// PrintPredictions(posteriorActivities.Select(ia => ia[0]).ToArray(), testLabels.Select(ia => ia[0]).ToArray());
					Metrics.Add(new Metrics { Name = Name, Estimates = PosteriorActivities[i], TrueLabels = dataSet.Labels[i] }, true);

					HoldoutMetrics.Metrics[i] = collection.ToArray();

					Console.WriteLine("{0,20}, Resident {1}, Hold out accuracy {2:N2}", Name, i, collection.Average(ia => ia.AverageAccuracy));
				}

				HoldoutMetrics.RecomputeAggregateMetrics();
				Metrics.RecomputeAggregateMetrics();
                
                // Console.WriteLine("Accuracies " + string.Join(", ", accuracy.ColumnAverage().Select(x => x.ToString("N2"))));
                // Console.WriteLine("Std. dev.  " + string.Join(", ", accuracy.ColumnStandardDeviation().Select(x => x.ToString("N2"))));
                // Console.WriteLine("Accuracies " + string.Join(", ", HoldoutMetrics.AverageAccuracy.Select(x => x.ToString("N2"))));
			}
		}

		/// <summary>
		/// Runs the active experiment.
		/// </summary>
		/// <param name="dataSet">Data set.</param>
		/// <param name="holdoutSet">Holdout set.</param>
		/// <param name="numberOfSelections">Number of selections.</param>
		/// <param name="priors">Priors.</param>
		public void RunActive(DataSet dataSet, DataSet holdoutSet, int numberOfSelections, Marginals priors)
		{
			if (ActiveLearners == null)
			{
				throw new InvalidOperationException("Active Learner not provided");
			}

			using (new CodeTimer("Running active experiment: " + Name))
			{
				Console.WriteLine();

				HoldoutMetrics = new HoldoutMetricsCollection { Metrics = new Metrics[dataSet.NumberOfResidents][] };

				// Metrics = new MetricsCollection(numberOfSelections);
				PosteriorActivities = new Bernoulli[dataSet.NumberOfResidents][];
				HoldoutPosteriorActivities = new Bernoulli[dataSet.NumberOfResidents][][];
				IndividualPosteriors = new Marginals[dataSet.NumberOfResidents];

				var accuracy = new double[dataSet.NumberOfResidents][];

				for (int i = 0; i < dataSet.NumberOfResidents; i++)
				{
					HoldoutMetrics.Metrics[i] = new Metrics[numberOfSelections];

					var collection = new List<Metrics>();
					IndividualPosteriors[i] = new Marginals(priors);

					// Test on holdout set
					HoldoutPosteriorActivities[i] = new Bernoulli[numberOfSelections][];
					accuracy[i] = new double[numberOfSelections];

					var dataSetForResident = dataSet.GetSubSet(i);
					var holdoutSetForResident = holdoutSet.GetSubSet(i);
                    // ActiveLearners[i].Transfer(i, 1); 

					// var individualPosteriors = new Marginals(priors);
					for (int j = 0; j < numberOfSelections; j++)
					{
						PosteriorActivities[i] = TestModel.Test(dataSetForResident, IndividualPosteriors[i])[0];
						HoldoutPosteriorActivities[i][j] = TestModel.Test(holdoutSetForResident, IndividualPosteriors[i])[0];

                        if (ActiveLearners[i].Unlabelled.Count == 0)
                        {
                            Console.WriteLine("Empty unlabelled set");
                            break;
                        }
                            
						// int index = ActiveLearner.GetValueOfInformation(i).ArgMax();
						int index;
						double val;
                        ActiveLearners[i].GetArgMaxVOI(PosteriorActivities[i], IndividualPosteriors[i], out index, out val);
                        
            			// Console.WriteLine("Index {0,4}, VOI {1:N4}", index, value);

                        // Now retrain using this label
                        ActiveLearners[i].UpdateModel(index);
      			        //IndividualPosteriors [i] = TrainModel.Train( dataSet.GetSubSet(i, ActiveLearners [i].Labelled.ToList()), priors, 10);
     				    IndividualPosteriors[i] = TrainModel.Train(dataSet.GetSubSet(i, index), IndividualPosteriors[i], 50); 

						var metrics = new Metrics { Name = Name, Estimates = HoldoutPosteriorActivities[i][j], TrueLabels = holdoutSet.Labels[i] };
						accuracy[i][j] = metrics.AverageAccuracy;

						collection.Add(metrics);
					}

					// PrintPredictions(posteriorActivities.Select(ia => ia[0]).ToArray(), testLabels.Select(ia => ia[0]).ToArray());
					HoldoutMetrics.Metrics[i] = collection.ToArray();

					Console.WriteLine("{0,20}, Resident {1}, \n\t\tClass ratio       {5}, \n\t\tHold out accuracy {2:N2}, \n\t\tAccuracies {3} \n\t\tBriers     {4}\n", 
						Name, i, collection.Average(ia => ia.AverageAccuracy).ToString( "N2" ), 
                        string.Join(", ", collection.Select(ia => ia.AverageAccuracy.ToString("N2"))), 
                        string.Join(", ", collection.Select( ia => ia.BrierScore.ToString( "N2" ) ) ),
                        holdoutSet.Labels[i].Average().ToString( "N2" )
                        );
				}

				HoldoutMetrics.RecomputeAggregateMetrics();

			}
		}
	}
}

