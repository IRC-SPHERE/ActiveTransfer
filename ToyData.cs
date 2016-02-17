//
// ToyData.cs
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
	using MicrosoftResearch.Infer.Maths;
	using SphereEngine;
	using GammaArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gamma, double>;
	using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;

	/// <summary>
	/// Toy data.
	/// </summary>
	public class ToyData
	{
		/// <summary>
		/// The prior weight means.
		/// </summary>
		private GaussianArray priorWeightMeans;

		/// <summary>
		/// The prior weight precisions.
		/// </summary>
		private GammaArray priorWeightPrecisions;

		/// <summary>
		/// Gets or sets the number of residents.
		/// </summary>
		/// <value>The number of residents.</value>
		public int NumberOfResidents { get; set; }

		/// <summary>
		/// Gets or sets the number of features.
		/// </summary>
		/// <value>The number of features.</value>
		public int NumberOfFeatures { get; set; }

		/// <summary>
		/// Gets or sets the number of activities.
		/// </summary>
		/// <value>The number of activities.</value>
		public int NumberOfActivities { get; set; }

		/// <summary>
		/// Gets or sets a value indicating whether this <see cref="ActiveTransfer.ToyData"/> use bias.
		/// </summary>
		/// <value><c>true</c> if use bias; otherwise, <c>false</c>.</value>
		public bool UseBias { get; set; }

		/// <summary>
		/// Gets or sets the true prior mean.
		/// </summary>
		/// <value>The true prior mean.</value>
		public Gaussian TruePriorMean { get; set; }

		/// <summary>
		/// Gets or sets the true prior precision.
		/// </summary>
		/// <value>The true prior precision.</value>
		public Gamma TruePriorPrecision { get; set; }

		/// <summary>
		/// Gets the prior weight means.
		/// </summary>
		/// <value>The prior weight means.</value>
		public GaussianArray PriorWeightMeans { get { return priorWeightMeans ?? (priorWeightMeans = DistributionArrayHelpers.CreateGaussianArray(NumberOfFeatures + (UseBias ? 1 : 0), 0, 1)); } }

		/// <summary>
		/// Gets the prior weight precisions.
		/// </summary>
		/// <value>The prior weight precisions.</value>
		public GammaArray PriorWeightPrecisions { get { return priorWeightPrecisions ?? (priorWeightPrecisions = DistributionArrayHelpers.CreateGammaArray(NumberOfFeatures + (UseBias ? 1 : 0), 1, 1)); } }

		/// <summary>
		/// Gets or sets the community weights.
		/// </summary>
		/// <value>The community weights.</value>
		public Gaussian[] CommunityWeights { get; set; }

		/// <summary>
		/// Gets or sets the weights.
		/// </summary>
		/// <value>The weights.</value>
		public double[][] Weights { get; set; }

		/// <summary>
		/// Gets or sets the data set.
		/// </summary>
		/// <value>The data set.</value>
		public DataSet DataSet { get; set; }

		/// <summary>
		/// Gets or sets the holdout set.
		/// </summary>
		/// <value>The holdout set.</value>
		public DataSet HoldoutSet { get; set; }

		/// <summary>
		/// Computes the weights.
		/// </summary>
		public void ComputeWeights()
		{
			if (NumberOfActivities != 2)
			{
				throw new InvalidOperationException("This version of the function is for binary data only");
			}

			Weights = new double[NumberOfResidents][];
			int numFeaturesIncludingBias = NumberOfFeatures + (UseBias ? 1 : 0);

			CommunityWeights = new Gaussian[numFeaturesIncludingBias];
			for (int i = 0; i < numFeaturesIncludingBias; i++)
			{
				CommunityWeights[i] = new Gaussian(TruePriorMean.Sample(), TruePriorPrecision.Sample());
			}


			for (int i = 0; i < NumberOfResidents; i++)
			{
				Weights[i] = new double[numFeaturesIncludingBias];

				// Generate weight per feature, and then sample from that per user, for this to match the model
				for (int j = 0; j < numFeaturesIncludingBias; j++)
				{
					Weights[i][j] = CommunityWeights[j].Sample();
				}
			}
		}

		/// <summary>
		/// Generate the data using the specified noisy example proportion.
		/// </summary>
		/// <param name="noisyExampleProportion">Noisy example proportion.</param>
		/// <param name="holdout">If set to <c>true</c> holdout.</param>
		public void Generate(double noisyExampleProportion, int numberOfInstances, bool holdout = false)
		{
			if (NumberOfActivities != 2)
			{
				throw new InvalidOperationException("This version of the function is for binary data only");
			}

			if (numberOfInstances == 0)
			{
				return;
			}

			if (Weights == null)
			{
				ComputeWeights();
			}

			// int numberOfInstances = holdout ? NumberOfHoldoutInstances : NumberOfInstances;

			var scores = new double[NumberOfResidents][];
			var features = new double[NumberOfResidents][][];
			var labels = new bool[NumberOfResidents][];
			int numFeaturesIncludingBias = NumberOfFeatures + (UseBias ? 1 : 0);

			for (int i = 0; i < NumberOfResidents; i++)
			{
				features[i] = new double[numberOfInstances][];
				scores[i] = new double[numberOfInstances];
				labels[i] = new bool[numberOfInstances];

				// Generate weight per feature, and then sample from that per user, for this to match the model
				for (int j = 0; j < numberOfInstances; j++)
				{
					bool noisyExample = Rand.Double() > noisyExampleProportion;
					features[i][j] = new double[numFeaturesIncludingBias];

					var products = new double[numFeaturesIncludingBias];
					for (int k = 0; k < numFeaturesIncludingBias; k++)
					{
						// double feature = Rand.Double() > noisyExampleProportion ? (double)Rand.Int(2) - 0.5 : 0.0; // Rand.Double() - 0.5 : 0.0;
						double feature = noisyExample ? Rand.Double() - 0.5 : 0.0; // (double)Rand.Int(2);
						features[i][j][k] = (k == NumberOfFeatures) ? -1 : feature;
						products[k] = Weights[i][k] * features[i][j][k];
					}

					scores[i][j] = new Gaussian(products.Sum(), 1).Sample();

					labels[i][j] = scores[i][j] > 0;
				}
			}

			if (holdout)
			{
				HoldoutSet = new DataSet { Features = features, Labels = labels };
			}
			else
			{
				DataSet = new DataSet { Features = features, Labels = labels };
			}
		}
	}
}

