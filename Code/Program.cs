//
// Program.cs
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
    using System.IO;
    using MicrosoftResearch.Infer.Maths;
    using Newtonsoft.Json;

    /// <summary>
    /// Main class.
    /// </summary>
    public class MainClass
    {
        const bool ShowFactorGraph = false;
        const bool Debug = false;

        /// <summary>
        /// The train model.
        /// </summary>
        private static readonly BinaryModel trainModel = BinaryModel.CreateTrainModel(ShowFactorGraph, Debug);

        /// <summary>
        /// The test model.
        /// </summary>
        private static readonly BinaryModel testModel = BinaryModel.CreateTestModel(false, Debug);

        private static readonly BinaryModel evidenceModel = BinaryModel.CreateTestModel(false, Debug);

        /// <summary>
        /// The entry point of the program, where the program control starts and ends.
        /// </summary>
        public static void Main()
        {
            // Demo of plotting utility
#if FALSE
			{
				var x = Enumerable.Range(0, 200).Select(ia => (double)ia / 100.0);
				var y = x.Select(ia => Math.Sin(2.0 * ia * Math.PI));
				PythonPlotter.Plot(x, y, "Test figure", "$x$", @"$\sin(2 \pi x)$");
                
			}
#endif

            Rand.Restart(0);

            Console.WriteLine("Run Toy Experiments [Y]/n");

            if (Console.ReadKey(true).Key.ToString().ToLower() != "n")
            {
                Console.WriteLine("Run Transfer [Y]/n");
                bool runTransfer = Console.ReadKey(true).Key.ToString().ToLower() != "n";
                Console.WriteLine("Run Active y/[N]");
                bool runActive = Console.ReadKey(true).Key.ToString().ToLower() == "y";
                Console.WriteLine("Run Active Transfer y/[N]");
                bool runActiveTransfer = Console.ReadKey(true).Key.ToString().ToLower() == "y";

                ToyDataRunner.Run(trainModel, testModel, runTransfer, runActive, runActiveTransfer);
            }

            Console.WriteLine("Run Real Experiments y/[N]");

            if (Console.ReadKey(true).Key.ToString().ToLower() == "y")
            {
                var accelerometerRunner = new RealDataRunner
                {
                    Target = JsonConvert.DeserializeObject<DataLoader>(File.ReadAllText("../../../data/accel/real_target_3.json")),
                    Source = JsonConvert.DeserializeObject<DataLoader>(File.ReadAllText("../../../data/accel/real_source_3.json")),
                    ActiveSteps = 20,
                    ShowPlots = true,
                    AddBias = true
                };

                Console.WriteLine("Run VOI [Y]/n");
                bool runVOI = Console.ReadKey(true).Key.ToString().ToLower() != "n";
                Console.WriteLine("Run Active Evidence y/[N]");
                bool runActiveEvidence = Console.ReadKey(true).Key.ToString().ToLower() == "y";

                accelerometerRunner.Run(trainModel, testModel, evidenceModel, runVOI, runActiveEvidence);
            }
        }
    }
}
