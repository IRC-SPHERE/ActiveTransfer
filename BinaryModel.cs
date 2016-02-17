//
// BinaryModel.cs
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
// #define USE_PRECOMPILED_ALGORITHM

namespace ActiveTransfer
{
    using System.Linq;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;
    using MicrosoftResearch.Infer;
    using SphereEngine;
    using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;
    using GammaArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gamma, double>;


    /// <summary>
    /// The Model.
    /// </summary>
    public class BinaryModel
    {
        private readonly VariableArray<int> numberOfExamples;
        private readonly Variable<int> numberOfResidents;
        private readonly Variable<int> numberOfFeatures;
        private readonly Variable<double> noisePrecision;
        private readonly VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> featureValues;
        private readonly Variable<GaussianArray> weightPriorMeans;
        private readonly Variable<GammaArray> weightPriorPrecisions;
        private readonly VariableArray<double> weightMeans;
        private readonly VariableArray<double> weightPrecisions;
        private readonly VariableArray<VariableArray<double>, double[][]> weights;
        private readonly VariableArray<VariableArray<bool>, bool[][]> activities;
        private readonly InferenceEngine engine;
        private readonly Variable<bool> evidence;

#if USE_PRECOMPILED_ALGORITHM
		private readonly IGeneratedAlgorithm algorithm;
#endif

        /// <summary>
        /// Initializes a new instance of the <see cref="ActiveTransfer.BinaryModel"/> class.
        /// </summary>
        /// <param name="trainModel">If set to <c>true</c> train model.</param>
        /// <param name="showFactorGraph">If set to <c>true</c> show factor graph.</param>
        /// <param name="debug">If set to <c>true</c> debug.</param>
        /// <param name="useBias">If set to <c>true</c>, add a bias feature.</param>
        public BinaryModel(bool trainModel, bool showFactorGraph = false, bool debug = false)
        {
            evidence = Variable.Bernoulli(0.5).Named("evidence");

            using (Variable.If(evidence))
            {
                numberOfResidents = Variable.New<int>().Named("numberOfResidents").Attrib(new DoNotInfer());
                numberOfFeatures = Variable.New<int>().Named("numberOfFeatures").Attrib(new DoNotInfer());

                var resident = new Range(numberOfResidents).Named("resident");
                var feature = new Range(numberOfFeatures).Named("feature");

                numberOfExamples = Variable.Array<int>(resident).Named("numberOfExamples").Attrib(new DoNotInfer());
                var example = new Range(numberOfExamples[resident]).Named("example").Attrib(new Sequential());

                noisePrecision = Variable.New<double>().Named("noisePrecision").Attrib(new DoNotInfer());

                weightPriorMeans = Variable.New<GaussianArray>().Named("weightPriorMeans").Attrib(new DoNotInfer());
                weightPriorPrecisions = Variable.New<GammaArray>().Named("weightPriorPrecisions").Attrib(new DoNotInfer());

                weightMeans = Variable.Array<double>(feature).Named("weightMeans");
                weightPrecisions = Variable.Array<double>(feature).Named("weightPrecisions");

                weightMeans.SetTo(Variable<double[]>.Random(weightPriorMeans));
                weightPrecisions.SetTo(Variable<double[]>.Random(weightPriorPrecisions));

                weights = Variable.Array(Variable.Array<double>(feature), resident).Named("weights");
                featureValues = Variable.Array(Variable.Array(Variable.Array<double>(feature), example), resident).Named("featureValues").Attrib(new DoNotInfer());

                //			if (!useBias)
                //			{
                //				thresholdPriors = Variable.New<GaussianArray>().Named("thresholdPrior").Attrib(new DoNotInfer());
                //				thresholds = Variable.Array<double>(resident).Named("threshold");
                //				thresholds.SetTo(Variable<double[]>.Random(thresholdPriors));
                //			}

                activities = Variable.Array(Variable.Array<bool>(example), resident).Named("activities");
                // activities[resident][example].AddAttribute(new MarginalPrototype(new Bernoulli()));

                using (Variable.ForEach(resident))
                {
                    var products = Variable.Array(Variable.Array<double>(feature), example).Named("products");
                    var scores = Variable.Array<double>(example).Named("scores");
                    var scoresPlusNoise = Variable.Array<double>(example).Named("scoresPlusNoise");

                    weights[resident][feature] = Variable.GaussianFromMeanAndPrecision(weightMeans[feature], weightPrecisions[feature]);

                    using (Variable.ForEach(example))
                    {
                        using (Variable.ForEach(feature))
                        {
                            products[example][feature] = weights[resident][feature] * featureValues[resident][example][feature];
                        }

                        scores[example] = Variable.Sum(products[example]).Named("score");
                        scoresPlusNoise[example] = Variable.GaussianFromMeanAndPrecision(scores[example], noisePrecision).Named("scorePlusNoise");

                        //					if (useBias)
                        {
                            activities[resident][example] = scoresPlusNoise[example] > 0;
                        }
                        //					else
                        //					{
                        //						var diff = (scoresPlusNoise[example] - thresholds[resident]).Named("diff");
                        //						activities[example][resident] = diff > 0;
                        //					}
                    }
                }
            }

            engine = new InferenceEngine
            {
                Algorithm = new ExpectationPropagation { DefaultNumberOfIterations = trainModel ? 10 : 1 },
                ShowFactorGraph = showFactorGraph,
                ShowProgress = false,
				// BrowserMode = BrowserMode.Never, // debug ? BrowserMode.OnError : BrowserMode.Never,
                ShowWarnings = debug
            };

            if (debug)
            {
                engine.Compiler.GenerateInMemory = false;
                engine.Compiler.WriteSourceFiles = true;
                engine.Compiler.IncludeDebugInformation = true;
                engine.Compiler.CatchExceptions = true;
            }

#if USE_PRECOMPILED_ALGORITHM
			numberOfResidents.ObservedValue = default(int);
			numberOfExamples.ObservedValue = default(int);
			numberOfFeatures.ObservedValue = default(int);
			noisePrecision.ObservedValue = default(double);
			featureValues.ObservedValue = default(double[][][]);
			weightPriorMeans.ObservedValue = default(DistributionStructArray<Gaussian, double>); // (DistributionStructArray<Gaussian, double>)Distribution<double>.Array(default(Gaussian[]));
			weightPriorPrecisions.ObservedValue = default(DistributionStructArray<Gamma, double>); // (DistributionStructArray<Gamma, double>)Distribution<double>.Array(default(Gamma[]));
			activities.ObservedValue = default(bool[][]);

			if (trainModel)
			{
				activities.AddAttribute(new DoNotInfer());
				algorithm = engine.GetCompiledInferenceAlgorithm(new IVariable[] { weights, weightMeans, weightPrecisions });
			}
			else
			{
				activities.AddAttribute(QueryTypes.Marginal);
				algorithm = engine.GetCompiledInferenceAlgorithm(new IVariable[] { activities });
			}
#endif
        }

        public void SetObservedVariables(double[][][] featureValues, GaussianArray priorWeightMeans, GammaArray priorWeightPrecisions, bool[][] labels = null)
        {
#if !USE_PRECOMPILED_ALGORITHM
            numberOfResidents.ObservedValue = featureValues.Length;
            numberOfExamples.ObservedValue = featureValues.Select(ia => ia.Length).ToArray();
            numberOfFeatures.ObservedValue = featureValues[0][0].Length;
            noisePrecision.ObservedValue = 1;

            if (labels != null)
            {
                activities.ObservedValue = labels;
            }

            this.featureValues.ObservedValue = featureValues;

            weightPriorMeans.ObservedValue = DistributionArrayHelpers.Copy(priorWeightMeans);
            weightPriorPrecisions.ObservedValue = DistributionArrayHelpers.Copy(priorWeightPrecisions);
#else
			algorithm.SetObservedValue(numberOfResidents.Name, featureValues.Length);
			algorithm.SetObservedValue(numberOfExamples.Name, featureValues[0].Length);
			algorithm.SetObservedValue(numberOfFeatures.Name, featureValues[0][0].Length);
			algorithm.SetObservedValue(noisePrecision.Name, 1.0);
			algorithm.SetObservedValue(this.featureValues.Name, featureValues);
			algorithm.SetObservedValue(weightPriorMeans.Name, DistributionArrayHelpers.Copy(priorWeightMeans));
			algorithm.SetObservedValue(weightPriorPrecisions.Name, DistributionArrayHelpers.Copy(priorWeightPrecisions));

			if (labels != null)
			{
				algorithm.SetObservedValue(activities.Name, labels);
			}
#endif
        }

        /// <summary>
        /// Train the specified dataSet and priors for the specified number of iterations.
        /// </summary>
        /// <param name="dataSet">Data set.</param>
        /// <param name="priors">Priors.</param>
        /// <param name="numberOfIterations">Number of iterations.</param>
        public Marginals Train(DataSet dataSet, Marginals priors, int numberOfIterations = 10)
        {
            SetObservedVariables(
                dataSet.Features,
                DistributionArrayHelpers.Copy(priors.WeightMeans),
                DistributionArrayHelpers.Copy(priors.WeightPrecisions),
                dataSet.Labels);

#if !USE_PRECOMPILED_ALGORITHM
            engine.Algorithm.DefaultNumberOfIterations = numberOfIterations;
            var posteriorWeights = engine.Infer<Gaussian[][]>(weights);
            var posteriorWeightMeans = engine.Infer<Gaussian[]>(weightMeans);
            var posteriorWeightPrecisions = engine.Infer<Gamma[]>(weightPrecisions);
#else
			algorithm.Execute(numberOfIterations);
			var posteriorWeights = algorithm.Marginal<Gaussian[][]>(weights.Name);
			var posteriorWeightMeans = algorithm.Marginal<Gaussian[]>(weightMeans.Name);
			var posteriorWeightPrecisions = algorithm.Marginal<Gamma[]>(weightPrecisions.Name);
#endif

            return new Marginals { Weights = posteriorWeights, WeightMeans = posteriorWeightMeans, WeightPrecisions = posteriorWeightPrecisions };
        }

        /// <summary>
        /// Test the specified dataSet and priors.
        /// </summary>
        /// <param name="dataSet">Data set.</param>
        /// <param name="priors">Priors.</param>
        public Bernoulli[][] Test(DataSet dataSet, Marginals priors)
        {
            SetObservedVariables(
                dataSet.Features,
                DistributionArrayHelpers.Copy(priors.WeightMeans),
                DistributionArrayHelpers.Copy(priors.WeightPrecisions));

#if !USE_PRECOMPILED_ALGORITHM
            var posteriorActivities = engine.Infer<Bernoulli[][]>(activities);
#else
			algorithm.Execute(1);
			var posteriorActivities = algorithm.Marginal<Bernoulli[][]>(activities.Name);
#endif

            return posteriorActivities;
        }

        public Bernoulli ComputeEvidence(DataSet dataSet, Marginals priors)
        {
            SetObservedVariables(
                      dataSet.Features,
                      DistributionArrayHelpers.Copy(priors.WeightMeans),
                      DistributionArrayHelpers.Copy(priors.WeightPrecisions),
                      dataSet.Labels);

            engine.Algorithm.DefaultNumberOfIterations = 1;

            return engine.Infer<Bernoulli>(evidence);
        }

#if !USE_PRECOMPILED_ALGORITHM
        public static BinaryModel CreateTrainModel(bool showFactorGraph = false, bool debug = false)
        {
            return new BinaryModel(true, showFactorGraph, debug);
        }

        public static BinaryModel CreateTestModel(bool showFactorGraph = false, bool debug = false)
        {
            return new BinaryModel(false, showFactorGraph, debug);
        }
#endif
    }
}

