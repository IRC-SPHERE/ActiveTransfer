//
// ActiveLearner.cs
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
using MicrosoftResearch.Infer.Distributions;

using GammaArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gamma, double>;
using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;
using SphereEngine;
using MicrosoftResearch.Infer.Factors;


namespace ActiveTransfer 
{
	/// <summary>
	/// Active learner interface.
	/// </summary>
	public interface IActiveLearner
	{
		/// <summary>
		/// Gets or sets the labelled sets.
		/// </summary>
		/// <value>The labelled.</value>
		HashSet<int> Labelled { get; set; }

    void Transfer( int resident, int num ); 

		/// <summary>
		/// Gets or sets the unlabelled sets.
		/// </summary>
		/// <value>The unlabelled.</value>
		HashSet<int> Unlabelled { get; set; }

		/// <summary>
		/// Gets or sets the data set.
		/// </summary>
		/// <value>The data set.</value>
		DataSet DataSet { get; set; }

		/// <summary>
		/// Gets or sets the active posteriors.
		/// </summary>
		/// <value>The active posteriors.</value>
//		IList<Marginals> ActivePosteriors { get; set; }

		/// <summary>
		/// Calculates the probabilities.
		/// </summary>
		/// <param name="priors">Priors.</param>
		// void CalculateProbabilities(Marginals priors);

		/// <summary>
		/// Gets the argument maxising the Value of information.
		/// </summary>
		/// <param name="activityPosteriors">Activity posteriors.</param>
		/// <param name="priors">Priors.</param>
		/// <param name="argMax">Argument max.</param>
		/// <param name="maxVal">Max value.</param>
		void GetArgMaxVOI(Bernoulli[] activityPosteriors, Marginals priors, out int argMax, out double maxVal);

		/// <summary>
		/// Updates the model.
		/// </summary>
		/// <param name="index">Index.</param>
		void UpdateModel(int index);
	}

	/// <summary>
	/// Reversable learner interface.
	/// </summary>
	public interface IReversableLearner
	{
		/// <summary>
		/// Gets or sets a value indicating whether this <see cref="ActiveTransfer.IReversableLearner"/> is reversed.
		/// </summary>
		/// <value><c>true</c> if reversed; otherwise, <c>false</c>.</value>
		bool Reversed { get; set; }
	}

	/// <summary>
	/// Active learner base.
	/// </summary>
	public class ActiveLearnerBase : IActiveLearner
	{
		/// <summary>
		/// The data set.
		/// </summary>
		public DataSet dataSet;

		/// <summary>
		/// Gets or sets the labelled sets.
		/// </summary>
		/// <value>The labelled.</value>
		public HashSet<int> Labelled { get; set; }

		/// <summary>
		/// Gets or sets the unlabelled sets.
		/// </summary>
		/// <value>The unlabelled.</value>
		public HashSet<int> Unlabelled { get; set; }

		/// <summary>
		/// Gets or sets the data set.
		/// </summary>
		/// <value>The data set.</value>
		public DataSet DataSet
		{
			get
			{
				return dataSet;
			}
			set
			{
				dataSet = value;

				if (dataSet == null)
				{
					Labelled = null;
					Unlabelled = null;
				}
				else
				{
					Labelled = new HashSet<int>();
					Unlabelled = new HashSet<int>(Enumerable.Range(0, DataSet.NumberOfInstances.Max()));
				}
			}
		}

        /// <summary>
        /// Transfer a number of instances from the unlabelled to the labelled set for the given resident.
        /// </summary>
        /// <param name="resident">The index of the resident.</param>
        /// <param name="num">The number to transfer.</param>
        public void Transfer(int resident, int num) 
        {
            if (num == 0)
            {
                return;
            }

            var rng = new Random();

            for (int tt = 0; tt < 2; ++tt) 
            {
                for (int nn = 0; nn < num; ++nn) 
                {
                    var uinds = Unlabelled.OrderBy(x => rng.Next());

                    foreach (var ind in uinds)
                    {
                        if (dataSet.Labels [0] [ind] == ( tt == 0 ? false : true )) 
                        {
                            Unlabelled.Remove(ind);
                            Labelled.Add(ind);
                            break;
                        }
                    }
                }
            }
        }

		/// <summary>
		/// Gets or sets the active posteriors.
		/// </summary>
		/// <value>The active posteriors.</value>
		/// <summary>
		/// Calculates the probabilities.
		/// </summary>
		/// <param name="priors">Priors.</param>
		/// <summary>
		/// Gets the argument maxising the Value of information.
		/// </summary>
		/// <param name="activityPosteriors">Activity posteriors.</param>
		/// <param name="priors">Priors.</param>
		/// <param name="argMax">Argument max.</param>
		/// <param name="maxVal">Max value.</param>
		public virtual void GetArgMaxVOI(Bernoulli[] activityPosteriors, Marginals priors, out int argMax, out double maxVal)
		{
			throw new NotImplementedException();
		}

		/// <summary>
		/// Updates the model.
		/// </summary>
		/// <param name="index">Index.</param>
		public void UpdateModel(int index)
		{
            if (Labelled.Contains(index))
            {
                throw new Exception( "The selected index is already in the labelled set." ); 
            }
            
			Unlabelled.Remove(index); 
			Labelled.Add(index); 
		}
	}

	/// <summary>
	/// Random learner.
	/// </summary>
	public class RandomLearner : ActiveLearnerBase
	{
		/// <summary>
		/// The random number generator.
		/// </summary>
		private Random random;

		/// <summary>
		/// Initializes a new instance of the <see cref="ActiveTransfer.RandomLearner"/> class.
		/// </summary>
		public RandomLearner() : this(0)
		{
		}

		/// <summary>
		/// Initializes a new instance of the <see cref="ActiveTransfer.RandomLearner"/> class.
		/// </summary>
		/// <param name="seed">The random seed.</param>
		public RandomLearner(int seed = 0)
		{
			random = new Random(seed);
		}

		/// <summary>
		/// Gets the argument maxising the Value of information.
		/// </summary>
		/// <param name="activityPosteriors">Activity posteriors.</param>
		/// <param name="priors">Priors.</param>
		/// <param name="argMax">Argument max.</param>
		/// <param name="maxVal">Max value.</param>
		public override void GetArgMaxVOI(Bernoulli[] activityPosteriors, Marginals priors, out int argMax, out double maxVal)
		{
			argMax = Unlabelled.ToArray()[random.Next(Unlabelled.Count)];
			maxVal = 0.0;
		}
	}

	/// <summary>
	/// Uncertain active learner.
	/// </summary>
	public class UncertainActiveLearner : ActiveLearnerBase, IReversableLearner
	{
		/// <summary>
		/// Gets or sets a value indicating whether this <see cref="ActiveTransfer.UncertainActiveLearner"/> is reversed.
		/// </summary>
		/// <value><c>true</c> if reversed; otherwise, <c>false</c>.</value>
		public bool Reversed { get; set; }


		/// <summary>
		/// Gets or sets the active posteriors.
		/// </summary>
		/// <value>The active posteriors.</value>
		/// <summary>
		/// Calculates the probabilities.
		/// </summary>
		/// <param name="priors">Priors.</param>
		/// <summary>
		/// Gets the argument maxising the Value of information.
		/// </summary>
		/// <param name="activityPosteriors">Activity posteriors.</param>
		/// <param name="priors">Priors.</param>
		/// <param name="argMax">Argument max.</param>
		/// <param name="maxVal">Max value.</param>
		public override void GetArgMaxVOI(Bernoulli[] activityPosteriors, Marginals priors, out int argMax, out double maxVal)
		{
			// More efficient to only loop over this once ...
			argMax = -1;
			bool isSet = false;
			maxVal = Reversed ? double.NegativeInfinity : double.PositiveInfinity;

			foreach (var index in Unlabelled)
			{
				double diff = Math.Abs(0.5 - activityPosteriors[index].GetMean());

				if (Reversed)
				{
					if (diff > maxVal || !isSet)
					{
						argMax = index;
						maxVal = diff;
						isSet = true;
					}
				}
				else
				{
					if (diff < maxVal || !isSet)
					{
						argMax = index;
						maxVal = diff;
						isSet = true;
					}
				}
			}
		}
	}

	/// <summary>
	/// Active learner.
	/// </summary>
	public class ActiveLearner : ActiveLearnerBase, IReversableLearner
	{
		/// <summary>
		/// Gets or sets a value indicating whether this <see cref="ActiveTransfer.ActiveLearner"/> is reversed.
		/// </summary>
		/// <value><c>true</c> if reversed; otherwise, <c>false</c>.</value>
		public bool Reversed { get; set; }

			/// <summary>
		/// Gets or sets the gains.
		/// </summary>
		/// <value>The gains.</value>
		public double[] Gains { get; set; }

		/// <summary>
		/// Gets or sets the costs.
		/// </summary>
		/// <value>The costs.</value>
		public double[] Costs { get; set; }
	    
		/// <summary>
		/// The hypothesis activity posteriors.
		/// </summary>
		private Bernoulli[] hypothesisActivityPosteriors;

		/// <summary>
		/// Gets or sets the train model.
		/// </summary>
		/// <value>The train model.</value>
		public BinaryModel TrainModel { get; set; }

		/// <summary>
		/// Gets or sets the test model.
		/// </summary>
		/// <value>The test model.</value>
		public BinaryModel TestModel { get; set; }

		/// <summary>
		/// Gets or sets the risk matrix.
		/// </summary>
		/// <value>The risk matrix.</value>
		public double [,] RiskMatrix { get; set; }

		/// <summary>
		/// Calculates the probabilities.
		/// </summary>
		public virtual void CalculateProbabilities(Marginals priors) 
		{
			hypothesisActivityPosteriors = TestModel.Test(DataSet, priors)[0];
		}

		/// <summary>
		/// Gets the probability of the given index.
		/// </summary>
		/// <returns>The probability of.</returns>
		/// <param name="index">Index.</param>
		private Bernoulli GetProbabilityOf(int index, Marginals priors)
		{
			return TestModel.Test(DataSet.GetSubSet(index), priors)[0][0];
		}


		private double JL_ii(int index) 
		{ 
			return DataSet.Labels[0][index]
					? RiskMatrix[1,0] * (1.0 - hypothesisActivityPosteriors[index].GetMean())
					: RiskMatrix[0,1] * hypothesisActivityPosteriors[index].GetMean();
		}

		private double JL(HashSet<int> labelled) 
		{
			return labelled.Sum(index => JL_ii(index));
		}

		private double JU_ii(int index) 
		{
			double prob = hypothesisActivityPosteriors[index].GetMean();
			return (RiskMatrix[1,0] + RiskMatrix[0,1]) * (1.0 - prob) * prob; 
		}

		private double JU(HashSet<int> unlabelled) 
		{
			return unlabelled.Sum(index => JU_ii(index));
		}

		private double JAll() 
		{
			return JU(Unlabelled) + JL(Labelled);
		}

		private double JMean() 
		{
			return JAll() / (Unlabelled.Count + Labelled.Count); 
		}

		private double JAll_j(int index, Bernoulli[] activityPosteriors, Marginals priors)
		{
			// var prevProbs = activityPosteriors.Select(ia => new Bernoulli(ia)).ToArray();
			hypothesisActivityPosteriors = activityPosteriors.Select(ia => new Bernoulli(ia)).ToArray();

			// Get datum
			// var datum = DataSet.GetSubSet(0, index);
			bool trueLabel = DataSet.Labels[0][index];

			// Create copies of the Labelled an Unlabelled sets
			var labelled = new HashSet<int>(Labelled);
			var unlabelled = new HashSet<int>(Unlabelled);

			labelled.Add(index); 
			unlabelled.Remove(index); 

			// datum.Labels[0][0] = true;
			DataSet.Labels[0][index] = true;

			// Learn as if positive
			// var positivePosteriors = TrainModel.Train(datum, priors, 1);
			Marginals positivePosteriors = priors;

			try
			{
				positivePosteriors = TrainModel.Train(DataSet.GetSubSet(0, index), priors, 1);
			}
			catch (ImproperMessageException)
			{
				// As fallback use priors
			}

			// recompute probabilities
			CalculateProbabilities(positivePosteriors);

			var jjposl = JL(labelled); 
			var jjposu = JU(unlabelled); 
			var Jjpos = (JAll()) * (1.0 - hypothesisActivityPosteriors[index].GetMean()); 

			// Restore posteriors

			labelled.Add(index); 
			unlabelled.Remove(index); 

			// datum.Labels[0][0] = false;
			DataSet.Labels[0][index] = false;

			// Learn as if negative
			// var negativePosteriors = TrainModel.Train(datum, priors, 1);
			Marginals negativePosteriors = priors;
			try
			{
				negativePosteriors = TrainModel.Train(DataSet.GetSubSet(0, index), priors, 1);
			}
			catch (ImproperMessageException)
			{
				// As fallback use priors
			}

			// recompute probabilities
			CalculateProbabilities(negativePosteriors);

			var jjnegl = JL(labelled); 
			var jjnegu = JU(unlabelled);
      var Jjneg = ( JAll() ) * ( hypothesisActivityPosteriors [index].GetMean() ); 

			// restore posteriors
			// activityPosteriors = prevProbs;
			DataSet.Labels[0][index] = trueLabel;

			var voi = Jjpos + Jjneg; 

			return voi; 
		}

		private double Cj( double p, int index ) 
		{
			return Costs[1] * p + Costs[0] * (1.0 - p); 
		}

		private double VOI(double jall, int index, Bernoulli[] activityPosteriors, Marginals priors) 
		{
			var prob = activityPosteriors[index].GetMean();
			var voij = JAll_j(index, activityPosteriors, priors); 
			return (jall - voij) - Cj(prob, index); 
		}

//		/// <summary>
//		/// Gets the value of information.
//		/// </summary>
//		/// <value>The value of information.</value>
//		public double[] GetValueOfInformation()
//		{
//			double jall = JAll();
//			return Unlabelled.Select(index => VOI(jall, index)).ToArray();
//		}

		/// <summary>
		/// Gets the argument maxising the Value of information.
		/// </summary>
		/// <param name="activityPosteriors">Activity posteriors.</param>
		/// <param name="priors">Priors.</param>
		/// <param name="argMax">Argument max.</param>
		/// <param name="maxVal">Max value.</param>
		public override void GetArgMaxVOI(Bernoulli[] activityPosteriors, Marginals priors, out int argMax, out double maxVal)
		{
			hypothesisActivityPosteriors = activityPosteriors.Select(ia => new Bernoulli(ia)).ToArray();
			double jall = JAll(); 

			var unlabelled = Unlabelled.ToArray();
			argMax = -1; 
			maxVal = Reversed ? double.PositiveInfinity : double.NegativeInfinity; 

			var signs = new double[unlabelled.Length];
			var vois = new double[unlabelled.Length];

			for (int i = 0; i < unlabelled.Length; i++)
			{
				var index = unlabelled[i];
				vois[i] = VOI(jall, index, activityPosteriors, priors);
				signs[i] = Math.Sign(vois[i]) / 2.0 + 0.5;
				//Console.Write( "." ); 
				//Console.WriteLine( "y_true: {0}", labels[0][ind] );
				//Console.WriteLine( "y_hat: {0}", probs[ind] );
				//Console.WriteLine( "VOJ_{0}: {1}", ind, voi ); 
				//Console.WriteLine(); 
				if (Reversed)
				{
					if (vois[i] < maxVal || argMax < 0)
					{
						maxVal = vois[i];
						argMax = index;
					}
				}
				else
				{
					if (vois[i] > maxVal || argMax < 0)
					{
						maxVal = vois[i];
						argMax = index;
					}
				}
			}

			//Console.WriteLine(); 
			//Console.WriteLine( "\n+ivity: {0}", signs.Average() ); 
		}


		public void VOITest( int numActivelySelected, Marginals priors) 
		{
			var onlineEstimates = new List<Bernoulli>(); 
			var onlineTargets = new List<bool>(); 

			Metrics metrics = null;

			for (int jj = 0; jj < numActivelySelected; ++jj)
			{
				CalculateProbabilities(priors); 

				//Console.WriteLine( "\nJL: {0}", JL() );
				//Console.WriteLine( "JU: {0}", JU() );

				int argMax;
				double maxVal;
				GetArgMaxVOI(hypothesisActivityPosteriors, priors, out argMax, out maxVal);

				Unlabelled.Remove(argMax); 
				Labelled.Add(argMax);

				UpdateModel(argMax); 


				onlineEstimates.Add(GetProbabilityOf(argMax, priors));
				onlineTargets.Add(DataSet.Labels[0][argMax]); 

				metrics = new Metrics
					{ 
						Name = "active", 
						Estimates = onlineEstimates.Select(ia => new Bernoulli(ia)).ToArray(), 
						TrueLabels = onlineTargets.ToArray()
					};

				// metrics.PrintSummary();
			}

			if (Unlabelled.Any())
			{
				CalculateProbabilities(priors); 
				foreach (var index in Unlabelled)
				{
					onlineEstimates.Add(hypothesisActivityPosteriors[index]); 
					onlineTargets.Add(DataSet.Labels[0][index]);
				}

				metrics = new Metrics
					{ 
						Name = "active", 
						Estimates = onlineEstimates.Select(ia => new Bernoulli(ia)).ToArray(), 
						TrueLabels = onlineTargets.ToArray()
					};
			}

			if (metrics != null)
			{
				metrics.PrintSummary();
			}
		}
	}
}