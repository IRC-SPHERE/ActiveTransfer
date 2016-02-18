//
// DataSet.cs
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

namespace ActiveTransfer
{
	/// <summary>
	/// Data set. Currently fixed to binary labels.
	/// </summary>
	public class DataSet
	{
		/// <summary>
		/// Gets or sets the features.
		/// </summary>
		/// <value>The features.</value>
		public double[][][] Features { get; set; }

		/// <summary>
		/// Gets or sets the labels.
		/// </summary>
		/// <value>The labels.</value>
		public bool[][] Labels { get; set; }

		/// <summary>
		/// Gets the number of residents.
		/// </summary>
		/// <value>The number of residents.</value>
		public int NumberOfResidents { get { return Features == null ? -1 : Features.Length; } } 

		/// <summary>
		/// Gets the number of instances.
		/// </summary>
		/// <value>The number of instances.</value>
		public int [] NumberOfInstances { 
      get {
        return Features == null
          ? null
          : Features.Select( ff => ff.Length ).ToArray(); 
      } 
    }

		/// <summary>
		/// Gets the number of features.
		/// </summary>
		/// <value>The number of features.</value>
		public int NumberOfFeatures { get { return Features == null ? -1 : Features[0][0].Length; } }

		/// <summary>
		/// Gets the number of activities.
		/// </summary>
		/// <value>The number of activities.</value>
		public int NumberOfActivities { get { return 2; } }

		/// <summary>
		/// Gets the sub set.
		/// </summary>
		/// <returns>The sub set.</returns>
		/// <param name="resident">Resident.</param>
		public DataSet GetSubSet(int resident)
		{
			return new DataSet { Features = new[] { Features[resident] }, Labels = new[] { Labels[resident] } };
		}

		/// <summary>
		/// Gets the sub-set.
		/// </summary>
		/// <returns>The sub set.</returns>
		/// <param name="resident">Resident.</param>
		/// <param name="index">Index.</param>
		public DataSet GetSubSet(int resident, int index)
		{
			return GetSubSet(resident, new[] { index });
		}

		/// <summary>
		/// Gets the sub-set.
		/// </summary>
		/// <returns>The sub set.</returns>
		/// <param name="resident">Resident.</param>
		/// <param name="indices">Indices.</param>
		public DataSet GetSubSet(int resident, IList<int> indices)
		{
			return GetSubSet(new[] { resident }, indices);
		}

		/// <summary>
		/// Gets the subset.
		/// </summary>
		/// <returns>The subset.</returns>
		/// <param name="residents">Residents.</param>
		/// <param name="indices">Indices.</param>
		public DataSet GetSubSet(IList<int> residents, IList<int> indices)
		{
			var dataSet = new DataSet { Features = new[] { new double[indices.Count][] }, Labels = new[] { new bool[indices.Count] } };

			for (int i = 0; i < residents.Count; i++)
			{
				for (int j = 0; j < indices.Count; j++)
				{
					int resident = residents[i];
					int index = indices[j];
					dataSet.Features[i][j] = Features[resident][index];
					dataSet.Labels[i][j] = Labels[resident][index];
				}
			}

			return dataSet;
		}

		public void SplitTrainTest(double trainProportion, out DataSet trainSet, out DataSet testSet)
		{
			if (trainProportion < 0.0 || trainProportion > 1.0)
			{
				throw new ArgumentOutOfRangeException("trainProportion");
			}

			var counts = Labels.Select(ia => (int)Math.Ceiling(trainProportion * (double)ia.Length)).ToArray();

			trainSet = new DataSet { Features = Features.Select((ia, i) => ia.Take(counts[i]).ToArray()).ToArray(), Labels = Labels.Select((ia, i) => ia.Take(counts[i]).ToArray()).ToArray() };
			testSet  = new DataSet { Features = Features.Select((ia, i) => ia.Skip(counts[i]).ToArray()).ToArray(), Labels = Labels.Select((ia, i) => ia.Skip(counts[i]).ToArray()).ToArray() };
		}
	}
}

