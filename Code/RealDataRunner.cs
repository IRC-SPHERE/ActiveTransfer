//
// RealDataRunner.cs
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
    using SphereEngine;

    public class RealDataRunner
    {
        /// <summary>
        /// Gets or sets the source.
        /// </summary>
        /// <value>The source.</value>
        public DataLoader Source { get; set; }

        /// <summary>
        /// Gets or sets the target.
        /// </summary>
        /// <value>The target.</value>
        public DataLoader Target { get; set; }

        /// <summary>
        /// Gets or sets the active steps.
        /// </summary>
        /// <value>The active steps.</value>
        public int ActiveSteps { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="RealDataRunner"/> add bias.
        /// </summary>
        /// <value><c>true</c> if add bias; otherwise, <c>false</c>.</value>
        public bool AddBias { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="RealDataRunner"/> show plots.
        /// </summary>
        /// <value><c>true</c> if show plots; otherwise, <c>false</c>.</value>
        public bool ShowPlots { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ActiveTransfer.ToyDataRunner"/> class.
        /// </summary>
        /// <param name="trainModel">Train model.</param>
        /// <param name="testModel">Test model.</param>
        public void Run(BinaryModel trainModel, BinaryModel testModel, BinaryModel evidenceModel, bool testVOI, bool testActiveEvidence)
        {
            const int NumberOfResidents = 7;
            const double KeepProportion = 1.0;

            var selectedFeatures = new HashSet<int>(Enumerable.Range(0, 48));

            var ted = Source.GetDataSet(Enumerable.Range(1, 14), AddBias, selectedFeatures, KeepProportion);
            var trd = Target.GetDataSet(Enumerable.Range(1, 25), AddBias, selectedFeatures, KeepProportion);
            // var ted = Source.GetDataSet( Enumerable.Range( 1, 1 ), AddBias, selectedFeatures, KeepProportion );
            // var trd = Target.GetDataSet( Enumerable.Range( 1, 20 ), AddBias, selectedFeatures, KeepProportion );
            // var hod = Target.GetDataSet( Enumerable.Range( 1 + NumberOfResidents * 1, NumberOfResidents ) );

            DataSet testSet;
            DataSet holdoutSet;
            ted.SplitTrainTest(0.5, out testSet, out holdoutSet);

            var NumFeatures = trd.Features.First().First().Count();

            var trainData = new ToyData
            {
                NumberOfResidents = trd.NumberOfResidents,
                NumberOfFeatures = NumFeatures,
                NumberOfActivities = 2,
                UseBias = false,
                DataSet = trd
            };

            var testData = new ToyData
            {
                NumberOfResidents = NumberOfResidents,
                NumberOfFeatures = NumFeatures,
                NumberOfActivities = 2,
                UseBias = false,
                DataSet = testSet,
                HoldoutSet = holdoutSet
            };

            var priors = new Marginals
            {
                WeightMeans = DistributionArrayHelpers.CreateGaussianArray(trainData.NumberOfFeatures, 0.0, 1.0).ToArray(),
                WeightPrecisions = DistributionArrayHelpers.CreateGammaArray(trainData.NumberOfFeatures, 1.0, 1.0).ToArray()
            };

            // TODO: Create meta-features that allow us to do the first form of transfer learning

            // Train the community model
            var communityExperiment = new Experiment
            {
                TrainModel = trainModel,
                TestModel = testModel,
                EvidenceModel = evidenceModel,
                Name = "Community"
            };

            communityExperiment.RunBatch(trainData.DataSet, priors);
            // communityExperiment.Posteriors.WeightPrecisions = priors.WeightPrecisions;

            // if (false)
            // {
            //     Utils.PlotPosteriors(communityExperiment.Posteriors.WeightMeans, communityExperiment.Posteriors.WeightPrecisions, null, "Community weights", "Feature", ShowPlots);
            //     Utils.PlotPosteriors(communityExperiment.Posteriors.WeightMeans, communityExperiment.Posteriors.WeightPrecisions, null, "Community weights (prior precision)", "Feature", ShowPlots);
            // }

            // Print top features
            // var topWeights = communityExperiment.Posteriors.WeightMeans.Zip(communityExperiment.Posteriors.WeightPrecisions, (m, p) => new { m, p }).Select((ia, i) => new { ia, i })
            // 	.OrderByDescending(x => Math.Abs(x.ia.m.GetMean())).ToList();
            // Console.WriteLine("Top 20 weights:\n {0}", string.Join("\n", topWeights.Take(20).Select(pair => string.Format("{0}: {1}", pair.i, new Gaussian(pair.ia.m.GetMean(), pair.ia.p.GetMean())))));

            // //communityExperiment.Posteriors.WeightPrecisions = DistributionArrayHelpers.Copy( priors.WeightPrecisions ).ToArray();
            var sourcePosteriors = new Marginals
            {
                WeightMeans = communityExperiment.Posteriors.WeightMeans,
                WeightPrecisions = priors.WeightPrecisions, //communityExperiment.Posteriors.WeightMeans, 
                Weights = null
            };

            // Select half the features
            /*
			trainData.DataSet.Features = trainData.DataSet.Features.Select(
				ia => ia.Select(
					ib => topWeights.Take(topWeights.Count / 2).Select(pair => ib[pair.i]).ToArray())
					.ToArray())
				.ToArray();

			// Retrain using these weights
			*/

            // if (false)
            // {
            // 	// Do online learning
            // 	var onlineExperiment = new Experiment
            // 	{
            // 		TrainModel = trainModel,
            // 		TestModel = testModel,
            // 		Name = "Online"
            // 	};

            // 	onlineExperiment.RunOnline(testData.DataSet, testData.HoldoutSet, priors);

            // 	// Do transfer learning
            // 	var personalisationExperiment = new Experiment
            // 	{
            // 		TrainModel = trainModel,
            // 		TestModel = testModel,
            // 		Name = "Community"
            // 	};

            // 	personalisationExperiment.RunOnline(testData.DataSet, testData.HoldoutSet, communityExperiment.Posteriors);

            // 	// Plot cumulative metrics
            // 	Utils.PlotCumulativeMetrics(new [] { onlineExperiment, personalisationExperiment }, "Active", ShowPlots);
            // }

            // ACTIVE MODEL
            foreach (var doTransfer in new[] { false, true })
            {
                var experiments = new List<Experiment>();
                var learners = CreateLearners(trainModel, testModel, evidenceModel, testData, testVOI, testActiveEvidence);

                foreach (var learner in learners)
                {
                    Console.WriteLine("Testing Active{0} Learning ({1})", doTransfer ? " Real Transfer" : "Real Online", learner.Key);
                    var experiment = new Experiment
                    {
                        TrainModel = trainModel,
                        TestModel = testModel,
                        Name = learner.Key,
                        ActiveLearners = learner.Value
                    };

                    experiment.RunActive(testData.DataSet, testData.HoldoutSet, ActiveSteps, doTransfer ? sourcePosteriors : priors);
                    experiments.Add(experiment);

                    if (false)
                    {
                        Utils.PlotPosteriors(
                            experiment.IndividualPosteriors[0].WeightMeans,
                            experiment.IndividualPosteriors[0].WeightPrecisions,
                            null,
                            "Posterior weights for " + learner.Key + " " + (doTransfer ? " (transfer)" : ""),
                            "Feature",
                            ShowPlots);
                    }
                }

                Utils.PlotHoldoutMetrics(experiments, doTransfer ? "Real Active Transfer" : "Real Active", "", ShowPlots);
            }
        }

        /// <summary>
        /// Creates the learners
        /// </summary>
        /// <param name="trainModel">Train model.</param>
        /// <param name="testModel">Test model.</param>
        /// <param name="evidenceModel">Evidence Model.</param>
        /// <param name="testData">Test data.</param>
        /// <param name="testVOI">Whether to test VOI.</param>
        /// <param name="testActiveEvidence">Whether to test Active Evidence.</param>
        /// <returns>The learners.</returns>
        public Dictionary<string, IList<IActiveLearner>> CreateLearners(
            BinaryModel trainModel,
            BinaryModel testModel,
            BinaryModel evidenceModel,
            ToyData testData,
            bool testVOI,
            bool testActiveEvidence)
        {
            var learners = new Dictionary<string, IList<IActiveLearner>>
                {
                    { "Random", Utils.CreateLearners<RandomLearner>(testData.DataSet, trainModel, testModel, evidenceModel) },
                    { "US",     Utils.CreateLearners<UncertainActiveLearner>(testData.DataSet, trainModel, evidenceModel, testModel) },
                    // { "ActEv+", Utils.CreateLearners<ActiveEvidence>(testData.DataSet, trainModel, testModel, evidenceModel) },
                    // { "ActEv-", Utils.CreateLearners<ActiveEvidence>(testData.DataSet, trainModel, testModel, evidenceModel, true) },
                    // { "VOI+",   Utils.CreateLearners<ActiveLearner>(testData.DataSet, trainModel, testModel, evidenceModel) },
                    // { "VOI-",   Utils.CreateLearners<ActiveLearner>(testData.DataSet, trainModel, testModel, evidenceModel, true) },
                    // { "CS",     Utils.CreateLearners<UncertainActiveLearner>(testData.DataSet, trainModel, testModel, evidenceModel, true) },
				};

            if (testVOI)
            {
                learners["CS"] = Utils.CreateLearners<UncertainActiveLearner>(testData.DataSet, trainModel, testModel, evidenceModel, true);
                learners["VOI+"] = Utils.CreateLearners<ActiveLearner>(testData.DataSet, trainModel, testModel, evidenceModel);
                learners["VOI-"] = Utils.CreateLearners<ActiveLearner>(testData.DataSet, trainModel, testModel, evidenceModel, true);
            }

            if (testActiveEvidence)
            {
                learners["ActEv+"] = Utils.CreateLearners<ActiveEvidence>(testData.DataSet, trainModel, testModel, evidenceModel);
            }

            return learners;
        }
    }
}

