using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using Casas;
using SphereEngine;

using GaussianArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gaussian, double>;
using GammaArray = MicrosoftResearch.Infer.Distributions.DistributionStructArray<MicrosoftResearch.Infer.Distributions.Gamma, double>;



namespace ActiveTransfer {
  public class Active {
    public double [] Gain;
    public double [] C;
    public double [,] R;

    public Gaussian [] [] activePosteriorWeights;
    public Gaussian [] activePosteriorWeightMeans;
    public Gamma [] activePosteriorWeightPrecisions;
    
    public BinaryModel trainModel;
    public BinaryModel testModel;

    public int NActivities; 

    public HashSet<int> L; 
    public HashSet<int> U; 

    public double [] [] [] features;
    public bool [] [] labels;
    public double [] probs; 

    public Metrics metrics {
      private set; 
      get; 
    }



    public int N {
      get {
        return L.Count(); 
      } 
    }

    public int M {
      get {
        return U.Count(); 
      }
    }





    public Active( int NActivities, double [,] R, double [][][] features, bool [][] labels ) {
      Gain = Enumerable.Range( 0, 2 ).Select( ii => 1.0 / 3.0 ).ToArray();
      C = Enumerable.Range( 0, 2 ).Select( ii => 1.0 ).ToArray();

      this.R = R;
      this.NActivities = NActivities; 

      this.features = features; 
      this.labels = labels; 

      probs = new double [labels[0].Count()];

      U = new HashSet<int>( Enumerable.Range( 0, labels[0].Count() ) ); 
      L = new HashSet<int>(); 
    }

    public void Transfer( int num ) { 
      if ( num == 0 )
        return; 
      
      var rng = new Random(); 

      for ( int tt = 0; tt < NActivities; ++tt ) {
        for ( int nn = 0; nn < num; ++nn ) {
          var uinds = U.OrderBy( x => rng.Next() );

          foreach ( var ind in uinds ) {
            if ( labels[0][ind] == ( tt == 0 ? false : true ) ) {
              U.Remove( ind ); 
              L.Add( ind ); 

              break; 
            }
          }
        }
      }

      UpdateModel( L ); 
    }




    public void CalculateProbabilities() {
      Bernoulli [] [] temp;

      testModel.Test( 
        Enumerable.Range( 0, M + N ).Select( ind => new [] { features [0] [ind] } ).ToArray(), 
        activePosteriorWeightMeans, 
        activePosteriorWeightPrecisions, 
        out temp );

      probs = temp.Select( tt => tt[0].GetMean() ).ToArray(); 
    }


    public double probof( int ind ) {
      Bernoulli [] [] temp;

      testModel.Test( 
        new [] { new [] { features [0] [ind] } }, 
        activePosteriorWeightMeans, 
        activePosteriorWeightPrecisions, 
        out temp );
      
      return temp [0] [0].GetMean();
    }

    public void UpdateModel( IEnumerable<int> inds ) {
      if ( inds.Count() == 0 )
        return; 
      
      trainModel.Train( 
        inds.Select( ind => new[] { features[0][ind] } ).ToArray(),
        inds.Select( ind => new[] { labels[0][ind] } ).ToArray(),
        activePosteriorWeightMeans, 
        activePosteriorWeightPrecisions,
        out activePosteriorWeights, 
        out activePosteriorWeightMeans, 
        out activePosteriorWeightPrecisions ); 
    }

    public void UpdateModel( int ind ) {
      trainModel.Train( 
        new[] { new[] { features[0][ind] } },
        new[] { new[] { labels[0][ind] } },
        activePosteriorWeightMeans, 
        activePosteriorWeightPrecisions,
        out activePosteriorWeights, 
        out activePosteriorWeightMeans, 
        out activePosteriorWeightPrecisions ); 

      U.Remove( ind ); 
      L.Add( ind ); 
    }





    public double JL_ii( bool isPositive, double prob ) {
      return isPositive 
        ? R[1, 0] * ( 1.0 - prob )
        : R[0, 1] * prob;
    }

    public double JL_ii( int ind ) { 
      return JL_ii( labels[0][ind], probs[ind] ); 
    }

    public double JL() {
      var jj = 0.0; 

      foreach ( var ind in new SortedSet<int>( L ) ) 
        jj += JL_ii( ind ); 

      return jj; 
    }





    public double JU_ii( double prob ) {
      return ( R[1, 0] + R[0, 1] ) * ( 1.0 - prob ) * prob; 
    }

    public double JU_ii( int ind ) {
      return JU_ii( probs[ind] ); 
    }

    public double JU() {
      double jj = 0.0; 

      foreach ( var ind in new SortedSet<int> ( U ) )
        jj += JU_ii( ind ); 

      return jj; 
    }





    public double JAll() {
      return JU() + JL();
    }

    public double JMean() {
      return JAll() / ( M + N ); 
    }






    public double JAll_j( double p, int ind ) {
      var trueLabel = labels[0][ind]; 
      var prevProbs = probs.Select( pp => pp ).ToArray(); 

//      var oldPosteriorWeightMeans = DistributionArrayHelpers.Copy( activePosteriorWeightMeans ).ToArray();
//      var oldPosteriorWeightPrecisions = DistributionArrayHelpers.Copy( activePosteriorWeightPrecisions ).ToArray(); 



      L.Add( ind ); 
      U.Remove( ind ); 

      labels[0][ind] = true; 

//      activePosteriorWeightMeans = DistributionArrayHelpers.Copy( oldPosteriorWeightMeans ).ToArray(); 
//      activePosteriorWeightPrecisions = DistributionArrayHelpers.Copy( oldPosteriorWeightPrecisions ).ToArray(); 
//      UpdateModel( ind ); 
//      CalculateProbabilities(); 

      //var Jjpos = ( JL() + JU() ) * ( 1.0 - probs[ind] ); 
      var jjposl = JL(); 
      var jjposu = JU(); 
      var Jjpos = ( jjposl + jjposu ) * ( 1.0 - probs[ind] ); 



      L.Add( ind ); 
      U.Remove( ind ); 

      labels[0][ind] = false; 

//      activePosteriorWeightMeans = DistributionArrayHelpers.Copy( oldPosteriorWeightMeans ).ToArray(); 
//      activePosteriorWeightPrecisions = DistributionArrayHelpers.Copy( oldPosteriorWeightPrecisions ).ToArray(); 
//      UpdateModel( ind ); 
//      CalculateProbabilities(); 

      //var Jjneg = ( JL() + JU() ) * ( probs[ind] ); 
      var jjnegl = JL(); 
      var jjnegu = JU(); 
      var Jjneg = ( jjnegl + jjnegu ) * ( probs[ind] ); 


      U.Add( ind ); 
      L.Remove( ind ); 
      probs = prevProbs; 
      labels[0][ind] = trueLabel; 

//      activePosteriorWeightMeans = DistributionArrayHelpers.Copy( oldPosteriorWeightMeans ).ToArray(); 
//      activePosteriorWeightPrecisions = DistributionArrayHelpers.Copy( oldPosteriorWeightPrecisions ).ToArray(); 

      var voi = Jjpos + Jjneg; 

      return voi; 
    }

    public double Cj( double p, int ind ) {
      return C[0] * p + C[1] * ( 1.0 - p ); 
    }

    public double VOI( double jall, int ind ) {
      var p = probs[ind]; 

      var voij = JAll_j( p, ind ); 
      return ( jall - voij ) - Cj( p, ind ); 
    }



    public void VOITest( int numActivelySelected ) {
      var onlineEstimates = new List<double>(); 
      var onlineTargets = new List<bool>(); 

      for ( int jj = 0; jj < numActivelySelected; ++jj ) {
        CalculateProbabilities(); 

        //Console.WriteLine( "\nJL: {0}", JL() );
        //Console.WriteLine( "JU: {0}", JU() );

        double jall = JAll(); 

        var inds = U.Select( uu => uu ).ToArray();
        int maxind = -1; 
        double maxval = -1.0; 

        var signs = new List<double>(); 

        foreach ( var ind in inds ) {
          var voi = VOI( jall, ind ); 
          signs.Add( Math.Sign( voi ) / 2.0 + 0.5 ); 

          //Console.Write( "." ); 
          //Console.WriteLine( "y_true: {0}", labels[0][ind] );
          //Console.WriteLine( "y_hat: {0}", probs[ind] );
          //Console.WriteLine( "VOJ_{0}: {1}", ind, voi ); 
          //Console.WriteLine(); 

          if ( voi > maxval || maxind < 0) {
            maxval = voi; 
            maxind = ind; 
          }
        }
        //Console.WriteLine(); 

        //Console.WriteLine( "\n+ivity: {0}", signs.Average() ); 

        U.Remove( maxind ); 
        L.Add( maxind );

        UpdateModel( maxind ); 


        onlineEstimates.Add( probof( maxind ) );
        onlineTargets.Add( labels[0][maxind] ); 

        metrics = new Metrics { 
          Name = "active", 
          Estimates = onlineEstimates.Select( oo => new Bernoulli( oo ) ).ToArray(), 
          TrueLabels = onlineTargets.ToArray()
        };

        //metrics.PrintSummary();

      }

      if ( U.Count() > 0 ) {
        CalculateProbabilities(); 
        foreach ( var ind in U ) {
          onlineEstimates.Add( probs[ind] ); 
          onlineTargets.Add( labels[0][ind] );
        }

        metrics = new Metrics { 
          Name = "active", 
          Estimates = onlineEstimates.Select( oo => new Bernoulli( oo ) ).ToArray(), 
          TrueLabels = onlineTargets.ToArray()
        };
      }

      metrics.PrintSummary();
    }













    public void asdf() {
      for ( int i = 0; i < N; ++i ) {
        trainModel.Train( new [] { new [] { features [0] [i] } }, new [] { new [] { labels [0] [i] } },
          activePosteriorWeightMeans, activePosteriorWeightPrecisions,
          out activePosteriorWeights, out activePosteriorWeightMeans, out activePosteriorWeightPrecisions ); // , out posteriorThresholds);
      }
    }

    public void asdfasdf() {
      Bernoulli [] [] activeActivities = { new Bernoulli [N] };

      for ( int i = 0; i < N; i++ ) {
        Bernoulli [] [] temp;
        testModel.Test( 
          new [] { new [] { features [0] [i] } }, 
          activePosteriorWeightMeans, 
          activePosteriorWeightPrecisions, 
          out temp );
        activeActivities [0] [i] = temp [0] [0];


        // Now retrain using this label
        trainModel.Train( new [] { new [] { features [0] [i] } }, new [] { new [] { labels [0] [i] } },
          activePosteriorWeightMeans, 
          activePosteriorWeightPrecisions,
          out activePosteriorWeights,
          out activePosteriorWeightMeans, 
          out activePosteriorWeightPrecisions );
      }

      var activeMetrics = new Metrics {
        Estimates = activeActivities [0],
        TrueLabels = labels [0]
      };

      activeMetrics.PrintSummary();
    }
  }
}




/*

      var c = 1.0;
      var C = new double [NActivities, NActivities] {
        { 0.0, c }, 
        { c, 0.0 }
      };

      var activeLearner = new Active( NActivities, C );


      activeLearner.activePosteriorWeightMeans = communityPosteriorWeightMeans;
      activeLearner.activePosteriorWeightPrecisions = communityPosteriorWeightPrecisions;

      activeLearner.trainModel = trainModel;
      activeLearner.testModel = testModel;

      activeLearner.features = testFeatures;
      activeLearner.labels = testLabels;

      activeLearner.asdfasdf();
 */