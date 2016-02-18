//
// ToyDataRunner.cs
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
	using System.Linq;
	using MicrosoftResearch.Infer.Distributions;
	using SphereEngine;

	public static class ToyDataRunner
	{
        const int ActiveSteps = 50;
        const double NoisyExampleProportion = 0.9;
        const int NumberOfFeatures = 10;
        
		/// <summary>
		/// Initializes a new instance of the <see cref="ActiveTransfer.ToyDataRunner"/> class.
		/// </summary>
		/// <param name="trainModel">Train model.</param>
		/// <param name="testModel">Test model.</param>
		public static void Run(BinaryModel trainModel, BinaryModel testModel, bool testTransfer, bool testActive, bool testActiveTransfer)
		{
			var phase1PriorMean = new Gaussian(4, 1);
			var phase1PriorPrecision = new Gamma(1, 1);
			var phase2PriorMean = new Gaussian(4, 1);
			var phase2PriorPrecision = new Gamma(1, 1);
			
			// Generate data for 5 individuals
			var data = new List<ToyData>();
			for (int i = 0; i < 3; i++)
			{
				var toy = new ToyData
					{ 
						// NumberOfInstances = 200,
						// NumberOfHoldoutInstances = i == 0 ? 0 : 1000,
						NumberOfResidents = 5,
						NumberOfFeatures = NumberOfFeatures,
						NumberOfActivities = 2,
						UseBias = false,
						TruePriorMean = i == 0 ? phase1PriorMean : phase2PriorMean,
						TruePriorPrecision = i == 0 ? phase1PriorPrecision : phase2PriorPrecision
					};	

				toy.Generate(i == 2 ? NoisyExampleProportion : 0.0, 200);
				if (i != 0)
				{
					// no need for holdout data in training set
					toy.Generate(0.0, 1000, true);
				}

				data.Add(toy);
			}

			var priors = new Marginals
				{
					WeightMeans = DistributionArrayHelpers.CreateGaussianArray(NumberOfFeatures, 0, 1).ToArray(),
					WeightPrecisions = DistributionArrayHelpers.CreateGammaArray(NumberOfFeatures, 1, 1).ToArray()
				};

			Console.WriteLine("Data Generated");

			// TODO: Create meta-features that allow us to do the first form of transfer learning

			// Train the community model
			Console.WriteLine("Training Community Model");
			var communityExperiment = new Experiment { TrainModel = trainModel, TestModel = testModel, Name = "Community" };
			communityExperiment.RunBatch(data[0].DataSet, priors);
			// PrintWeightPriors(communityExperiment.Posteriors, trainData.CommunityWeights);

			// Utils.PlotPosteriors(communityExperiment.Posteriors.Weights, data[0].Weights);
			// Utils.PlotPosteriors(communityExperiment.Posteriors.WeightMeans, communityExperiment.Posteriors.WeightPrecisions, null, "Community weights", "Feature");

			// return;

			if (testTransfer)
			{
				// Do online learning
				// Console.WriteLine("Testing Online Model");
				var onlineExperiment = new Experiment { TrainModel = trainModel, TestModel = testModel, Name = "Online" };
				onlineExperiment.RunOnline(data[1].DataSet, data[1].HoldoutSet, priors);

				// Do transfer learning
				// Console.WriteLine("Testing Community Model");
				var personalisationExperiment = new Experiment { TrainModel = trainModel, TestModel = testModel, Name = "Community" };
				personalisationExperiment.RunOnline(data[1].DataSet, data[1].HoldoutSet, communityExperiment.Posteriors);

				// Plot cumulative metrics
				Utils.PlotCumulativeMetrics(new[] { onlineExperiment, personalisationExperiment }, "Toy Transfer");
			}
            else
            {
                Console.WriteLine("Skipping Transfer Learning");
            }

			// ACTIVE MODEL
            if (testActive)
            {
                ActiveTransfer(trainModel, testModel, data, "Toy Active", priors);
            }
            else
            {
                Console.WriteLine("Skipping Active Learning");
            }

            if (testActiveTransfer)
            {
                Console.WriteLine("Note that the transfer learning is very effective here, so the active learning doesn't add much");
                ActiveTransfer(trainModel, testModel, data, "Toy Active Transfer", communityExperiment.Posteriors);
            }
			else
            {
                Console.WriteLine("Skipping Active Transfer Learning");
            }

            // Now create different costs for acquiring labels - want to demonstrate that we choose from all 3 possible labels			
		}
        
        /// <summary>
        /// Active Transfer tests.
        /// </summary>
        /// <param name="trainModel">The train model.</param>
        /// <param name="testModel">The test model.</param>
        /// <param name="data">The datasets.</param>
        /// <param name="doTransfer">Whether or not to do transfer learning.</param>
        public static void ActiveTransfer(BinaryModel trainModel, BinaryModel testModel, IList<ToyData> data, string title, Marginals priors)
        {
            var learners = new Dictionary<string, IList<IActiveLearner>>
            {
                { "Random", Utils.CreateLearners<RandomLearner>(data[2].DataSet, trainModel, testModel, null) },
                { "US", Utils.CreateLearners<UncertainActiveLearner>(data[2].DataSet, trainModel, testModel, null) },
                { "CS", Utils.CreateLearners<UncertainActiveLearner>(data[2].DataSet, trainModel, testModel, null, true) },
                { "VOI+", Utils.CreateLearners<ActiveLearner>(data[2].DataSet, trainModel, testModel, null) },
                { "VOI-", Utils.CreateLearners<ActiveLearner>(data[2].DataSet, trainModel, testModel, null, true) }
            };
                
            var experiments = new List<Experiment>();

            foreach (var learner in learners)
            {
                Console.WriteLine("Testing {0} ({1})", title, learner.Key);
                var experiment = new Experiment { TrainModel = trainModel, TestModel = testModel, Name = learner.Key, ActiveLearners = learner.Value };
                experiment.RunActive(data[2].DataSet, data[2].HoldoutSet, ActiveSteps, priors);
                experiments.Add(experiment);
            }

            Utils.PlotHoldoutMetrics(experiments, title, "", true);            
        }
	}
}

