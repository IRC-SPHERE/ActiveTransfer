//
// MultiClassModel.cs
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
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;
    using MicrosoftResearch.Infer;
    using SphereEngine;

    using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;
    using GammaArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gamma, double>;

    /// <summary>
    /// The Multiclass Model.
    /// </summary>
    public class MultiClassModel
    {
        private readonly Variable<int> numberOfExamples;
        private readonly Variable<int> numberOfResidents;
        private readonly Variable<int> numberOfActivities;
        private readonly Variable<int> numberOfFeatures;
        private readonly Variable<double> noisePrecision;
        private readonly VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> featureValues;
        private readonly Variable<GaussianArray> weightPriorMeans;
        private readonly Variable<GammaArray> weightPriorPrecisions;
        private readonly VariableArray<double> weightMeans;
        private readonly VariableArray<double> weightPrecisions;
        private readonly VariableArray<VariableArray<VariableArray<double>, double[][]>, double[][][]> weights;
        private readonly VariableArray<VariableArray<int>, int[][]> activities;

        private readonly InferenceEngine engine;

        /// <summary>
        /// Initializes a new instance of the <see cref="MultiClassModel"/> class.
        /// </summary>
        /// <param name="trainModel">If set to <c>true</c> train model.</param>
        /// <param name="showFactorGraph">If set to <c>true</c> show factor graph.</param>
        /// <param name="debug">If set to <c>true</c> debug.</param>
        public MultiClassModel(bool trainModel, bool showFactorGraph = false, bool debug = false)
        {
            numberOfExamples = Variable.New<int>().Named("numberOfExamples").Attrib(new DoNotInfer());
            numberOfResidents = Variable.New<int>().Named("numberOfResidents").Attrib(new DoNotInfer());
            numberOfActivities = Variable.New<int>().Named("numberOfActivities").Attrib(new DoNotInfer());
            numberOfFeatures = Variable.New<int>().Named("numberOfFeatures").Attrib(new DoNotInfer());

            var resident = new Range(numberOfResidents).Named("resident");
            var activity = new Range(numberOfActivities).Named("activity");
            var feature = new Range(numberOfFeatures).Named("sensor");
            var example = new Range(numberOfExamples).Named("example").Attrib(new Sequential());

            noisePrecision = Variable.New<double>().Named("noisePrecision").Attrib(new DoNotInfer());

            weightPriorMeans = Variable.New<GaussianArray>().Named("weightPriorMeans").Attrib(new DoNotInfer());
            weightPriorPrecisions = Variable.New<GammaArray>().Named("weightPriorPrecisions").Attrib(new DoNotInfer());

            weightMeans = Variable.Array<double>(activity).Named("weightMeans");
            weightPrecisions = Variable.Array<double>(activity).Named("weightPrecisions");

            weightMeans.SetTo(Variable<double[]>.Random(weightPriorMeans));
            weightPrecisions.SetTo(Variable<double[]>.Random(weightPriorPrecisions));

            weights = Variable.Array(Variable.Array(Variable.Array<double>(feature), activity), resident).Named("weights");
            weights[resident][activity][feature] = Variable.GaussianFromMeanAndPrecision(
                            weightMeans[activity], weightPrecisions[activity]).ForEach(resident, feature);

            featureValues = Variable.Array(Variable.Array(Variable.Array<double>(feature), resident), example).Named("featureValues").Attrib(new DoNotInfer());

            activities = Variable.Array(Variable.Array<int>(resident), example).Named("activities");
            activities.SetValueRange(activity);

            using (Variable.ForEach(resident))
            {
                DefineSymmetryBreaking(weights[resident], activity, feature);

                using (Variable.ForEach(example))
                {
                    var score = BpmUtils.ComputeClassScores(weights[resident], featureValues[example][resident], noisePrecision);
                    if (!trainModel)
                    {
                        activities[example][resident] = Variable.DiscreteUniform(activity);
                    }

                    BpmUtils.ConstrainMaximum(activities[example][resident], score);
                }
            }

            if (trainModel)
            {
                activities.AddAttribute(new DoNotInfer());
            }

            engine = new InferenceEngine
            {
                Algorithm = new ExpectationPropagation { DefaultNumberOfIterations = trainModel ? 10 : 1 },
                ShowFactorGraph = showFactorGraph,
                ShowProgress = debug,
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
        }

        /// <summary>
        /// Defines the symmetry-breaking constraints. 
        /// For each feature, the sum of weights is constraint to be constant over all classes.
        /// </summary>
        /// <param name="weights">The random variables over the weights.</param>
        /// <param name="activity">The activity range.</param>
        /// <param name="feature">The feature range.</param>
        private void DefineSymmetryBreaking(VariableArray<VariableArray<double>, double[][]> weights, Range activity, Range feature)
        {
            var transposedWeights = Variable.Array(Variable.Array<double>(activity), feature).Named("TransposedWeights");
            var transposedWeightSums = Variable.Array<double>(feature).Named("TransposedWeightSums");

            // Transpose the weights
            transposedWeights[feature][activity] = Variable.Copy(weights[activity][feature]);

            // For each feature, sum the transposed weights over classes
            transposedWeightSums[feature] = Variable.Sum(transposedWeights[feature]);

            // Constrain all sums to be constant
            Variable.ConstrainEqual(transposedWeightSums[feature], 0);
        }

        public void SetObservedVariables(int numberOfActivities, double[][][] featureValues)
        {
            numberOfExamples.ObservedValue = featureValues.Length;
            numberOfResidents.ObservedValue = featureValues[0].Length;
            numberOfFeatures.ObservedValue = featureValues[0][0].Length;
            this.numberOfActivities.ObservedValue = numberOfActivities;
            noisePrecision.ObservedValue = 10;

            this.featureValues.ObservedValue = featureValues;
        }

        public void Train(int numberOfActivities, double[][][] featureValues, int[][] labels,
            out Gaussian[][][] posteriorWeights, out Gaussian[] posteriorWeightMeans, out Gamma[] posteriorWeightPrecisions)
        {
            SetObservedVariables(numberOfActivities, featureValues);

            activities.ObservedValue = labels;

            var priorWeightMeans = new Gaussian[numberOfActivities];
            var priorWeightPrecisions = new Gamma[numberOfActivities];
            for (int i = 0; i < numberOfActivities; i++)
            {
                priorWeightMeans[i] = new Gaussian(0, 1);
                priorWeightPrecisions[i] = new Gamma(4, 0.5);
            }

            weightPriorMeans.ObservedValue = DistributionArrayHelpers.Copy(priorWeightMeans);
            weightPriorPrecisions.ObservedValue = DistributionArrayHelpers.Copy(priorWeightPrecisions);

            posteriorWeights = engine.Infer<Gaussian[][][]>(weights);
            posteriorWeightMeans = engine.Infer<Gaussian[]>(weightMeans);
            posteriorWeightPrecisions = engine.Infer<Gamma[]>(weightPrecisions);
        }

        public void Test(int numberOfActivities, double[][][] featureValues, Gaussian[] priorWeightMeans, Gamma[] priorWeightPrecisions, out Discrete[][] posteriorActivities)
        {
            SetObservedVariables(numberOfActivities, featureValues);

            weightPriorMeans.ObservedValue = DistributionArrayHelpers.Copy(priorWeightMeans);
            weightPriorPrecisions.ObservedValue = DistributionArrayHelpers.Copy(priorWeightPrecisions);


            posteriorActivities = engine.Infer<Discrete[][]>(activities);
        }
    }
}

