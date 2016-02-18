using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Factors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SphereEngine;

namespace ActiveTransfer {
  public class ActiveEvidence : ActiveLearnerBase, IReversableLearner {
    private Random rng; 

    /// <summary>
    /// The hypothesis activity posteriors.
    /// </summary>
    public bool Reversed{ get; set; }

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
    /// 
    /// </summary>
    public BinaryModel EvidenceModel {
      get; set; }

    public ActiveEvidence() {
      rng = new Random( 12345 ); 
    }

    /// <summary>
    /// Calculates the probabilities.
    /// </summary>
    public void CalculateProbabilities(Marginals priors) 
    {
      hypothesisActivityPosteriors = TestModel.Test(DataSet, priors)[0];
    }

    private double ExpectedEvidence(int index, Marginals priors)
    {
      var niter = 1; 

      var pp = hypothesisActivityPosteriors[index]; 
      bool trueLabel = DataSet.Labels[0][index];


      var labelled = new HashSet<int>( Labelled ); 
      labelled.Add( index ); 

      //evidenceData.Labels[0] = evidenceData.Labels[0].Select( ll => !ll ).ToArray(); 


      // Learn as if positive
      DataSet.Labels[0][index] = true;

      Marginals positivePosteriors = priors;

      try
      {
        if ( Reversed )
          positivePosteriors = priors;
        else
          positivePosteriors = TrainModel.Train(DataSet.GetSubSet(0, index), priors, niter);
      }
      catch (ImproperMessageException)
      {
        // As fallback use priors
      }

      var positivePriorEvidence = EvidenceModel.ComputeEvidence( DataSet.GetSubSet( 0, labelled.ToList() ), priors ); 
      var positivePostrEvidence = EvidenceModel.ComputeEvidence( DataSet.GetSubSet( 0, labelled.ToList() ), positivePosteriors ); 



      // Learn as if negative
      DataSet.Labels[0][index] = false;

      Marginals negativePosteriors = priors;

      try
      {
        if ( Reversed )
          negativePosteriors = priors;
        else
          negativePosteriors = TrainModel.Train(DataSet.GetSubSet(0, index), priors, niter);
      }
      catch (ImproperMessageException)
      {
        // As fallback use priors
      }

      var negativePriorEvidence = EvidenceModel.ComputeEvidence( DataSet.GetSubSet( 0, labelled.ToList() ), priors ); 
      var negativePostrEvidence = EvidenceModel.ComputeEvidence( DataSet.GetSubSet( 0, labelled.ToList() ), negativePosteriors ); 



      DataSet.Labels[0][index] = trueLabel;
      


      var returns = new List<double>(); 

      returns.Add( 
         ( positivePriorEvidence.LogOdds ) / 
         ( negativePriorEvidence.LogOdds )
      ); 
      //return Math.Max( returns.Last(), 1.0 / returns.Last() );

      returns.Add( 
         ( positivePostrEvidence.LogOdds ) / 
         ( negativePostrEvidence.LogOdds )
      ); 
      //return Math.Max( returns.Last(), 1.0 / returns.Last() );

      returns.Add( 
         ( positivePriorEvidence.LogOdds + positivePostrEvidence.GetLogProbTrue() ) / 
         ( negativePriorEvidence.LogOdds + negativePostrEvidence.GetLogProbTrue() )  
      ); 
      return Math.Max( returns.Last(), 1.0 / returns.Last() );
    }


    public override void GetArgMaxVOI( MicrosoftResearch.Infer.Distributions.Bernoulli [] activityPosteriors, Marginals priors, out int argMax, out double maxVal ) {
      CalculateProbabilities( priors ); 

      var evidences = new Dictionary<int, double>(); 
      
      var sortedUnlabelled = Unlabelled
        .OrderBy( _ => rng.NextDouble() )
        .OrderBy( uu => Math.Abs( hypothesisActivityPosteriors[uu].GetMean() - 0.5 ) )
        .Take( 10 )
      ; 

      foreach ( var index in sortedUnlabelled ) { 
        var evidence = ExpectedEvidence( index, priors ); 

        evidences.Add( index, evidence ); 
      }

      var ordered = evidences.OrderBy( ee => ee.Value ); 
      argMax = ordered.First().Key; 
      maxVal = ordered.First().Value;
    }
  }
}