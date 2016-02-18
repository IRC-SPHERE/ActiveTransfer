using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ActiveTransfer {
  public class DataLoader {
    public List<int> s {
      set;
      get;
    }

    public List<bool> y {
      set;
      get;
    }

    public List<List<double>> x {
      set;
      get;
    }

    public int N {
      get {
        return y.Count();
      }
    }

    public int NumSubjects {
      get {
        return ( new HashSet<int>( s ) ).Count();
      }
    }

    public int NumFeatures {
      get {
        return x.First().Count();
      }
    }

    public DataSet GetDataSet( IEnumerable<int> subjects, bool addbias, HashSet<int> selected, double keepProportion = 1.0 ) {
      var rng = new Random( 12345 );

      var features = new double [subjects.Count()] [][];
      var labels = new bool [subjects.Count()] [];

      var inds = new Dictionary<int, int>();
      for ( int ss = 0; ss < subjects.Count(); ++ss )
        inds [subjects.ElementAt( ss )] = ss;

      for ( var ss = 0; ss < subjects.Count(); ++ss ) {
        var feats = new List<double []>();
        var labs = new List<bool>();

        for ( int ii = 0; ii < N; ++ii ) {
          if ( inds.ContainsKey( s [ii] ) && inds [s [ii]] == ss ) {
            var thisfeat = new List<double>();

            for ( int ff = 0; ff < x [ii].Count(); ++ff )
		      if ( selected == null || !selected.Any() || selected.Contains( ff ) )
                thisfeat.Add( x [ii] [ff] ); 

              if ( addbias )
                thisfeat.Add( 1.0 ); 

            feats.Add( thisfeat.ToArray() );
            labs.Add( y [ii] );
          }
        }

        var order = Enumerable.Range( 0, labs.Count() ).OrderBy( ii => rng.NextDouble() );
        var nKeep = Convert.ToInt32( keepProportion * feats.Count() );
        //features [ss] = order.Select( ii => feats[ii] ).ToArray();
        //labels [ss]   = order.Select( ii => labs[ii] ).ToArray();
        features [ss] = feats.Take( nKeep ).ToArray();
        labels [ss]   = labs .Take( nKeep ).ToArray();
      }

      var dataset = new DataSet {
        Features = features,
        Labels = labels
      };

      return dataset;
    }
  }
}
