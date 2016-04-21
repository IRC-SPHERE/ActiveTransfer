//
// DataLoader.cs
//
// Author:
//       Tom Diethe <tom.diethe@bristol.ac.uk>
//
// Copyright (c) 2016 University of Bristol
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

    /// <summary>
    /// Data loader.
    /// </summary>
    public class DataLoader
    {
        public List<int> s { get; set; }

        public List<bool> y { get; set; }

        public List<List<double>> x { get; set; }

        public int N
        {
            get 
            {
                return y.Count();
            }
        }

        public int NumSubjects
        {
            get 
            {
                return (new HashSet<int>(s)).Count();
            }
        }

        public int NumFeatures
        {
            get 
            {
                return x.First().Count();
            }
        }

        public DataSet GetDataSet(IEnumerable<int> subjects, bool addbias, HashSet<int> selected, double keepProportion = 1.0)
        {
            //var rng = new Random( 12345 );

            var features = new double[subjects.Count()][][];
            var labels = new bool[subjects.Count()][];

            var inds = new Dictionary<int, int>();
            for (int ss = 0; ss < subjects.Count(); ++ss)
                inds[subjects.ElementAt(ss)] = ss;

            for (var ss = 0; ss < subjects.Count(); ++ss)
            {
                var feats = new List<double[]>();
                var labs = new List<bool>();

                for (int ii = 0; ii < N; ++ii)
                {
                    if (inds.ContainsKey(s[ii]) && inds[s[ii]] == ss)
                    {
                        var thisfeat = new List<double>();

                        for (int ff = 0; ff < x[ii].Count(); ++ff)
                            if (selected == null || !selected.Any() || selected.Contains(ff))
                                thisfeat.Add(x[ii][ff]);

                        if (addbias)
                            thisfeat.Add(1.0);

                        feats.Add(thisfeat.ToArray());
                        labs.Add(y[ii]);
                    }
                }

                //var order = Enumerable.Range( 0, labs.Count() ).OrderBy( ii => rng.NextDouble() );
                var nKeep = Convert.ToInt32(keepProportion * feats.Count());
                //features [ss] = order.Select( ii => feats[ii] ).ToArray();
                //labels [ss]   = order.Select( ii => labs[ii] ).ToArray();
                features[ss] = feats.Take(nKeep).ToArray();
                labels[ss] = labs.Take(nKeep).ToArray();
            }

            var dataset = new DataSet
            {
                Features = features,
                Labels = labels
            };

            return dataset;
        }
    }
}
