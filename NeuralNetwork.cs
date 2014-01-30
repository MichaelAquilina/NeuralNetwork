using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

namespace ArtificialNeuralNetwork
{
    #region Extra Classes

    public delegate void EpochEventHandler(object sender, EpochEvent args );

    public delegate double ActivationFunction( double NET );

    public class ActivationFunctions
    {
        public static double Sigmoid(double NET)
        {
            double value =  1.0 / (1.0 + Math.Pow(Math.E, -1.0 * NET));

            return value;
        }

        public static double Tangent(double NET)
        {
            double value = Math.Tanh(NET / 2);

            return value;
        }   
    }

    public struct EpochEvent
    {
        public int Epochs;
        public int GoodFacts;
        public int BadFacts;

        public EpochEvent(int Epochs, int GoodFacts, int BadFacts )
        {
            this.Epochs = Epochs;
            this.GoodFacts = GoodFacts;
            this.BadFacts = BadFacts;
        }
    }

    #endregion

    /// <summary>
    /// Artificial Neural Network class capable of training and determining input data
    /// </summary>
    public class NeuralNetwork
    {
        #region Global Variables

        public double LearningRate = 0.25;
        public double ErrorThreshold = 0.25;
        public int EpochsLimit = 20000;

        private Boolean randomTraining = true;
        private int Epochs = 0;
        private Boolean trained = false;

        //activation function to be used by the neural network
        public ActivationFunction ActivationFunction;

        public Matrix InputHidden;
        public Matrix HiddenOutput;

        public Boolean IsTrained
        {
            get { return trained; }
        }
        public Boolean RandomTraining
        {
            get { return randomTraining; }
            set { randomTraining = value; }
        }
        public int EpochsTaken
        {
            get { return Epochs; }
        }

        public event EpochEventHandler EpochCompleted;

        #endregion

        public NeuralNetwork(int InputLayer, int HiddenLayer, int OutputLayer)
        {
            InputHidden = Matrix.RandomMatrix(InputLayer, HiddenLayer);
            HiddenOutput = Matrix.RandomMatrix(HiddenLayer, OutputLayer);

            ActivationFunction = ActivationFunctions.Sigmoid;
        }

        public void ResetEpochsTaken()
        {
            Epochs = 0;
        }

        public void Train(List<LabeledVectorN> TrainingData)
        {
            //used to determine if the ANN is stuck in some local maxima
            int PrevBadFacts = 0;
            int StuckCount = 0;

            int GoodFacts;
            int BadFacts;
            int Epochs = 0;

            VectorN OUT;
            VectorN OUTH;
            VectorN TARGET;
            VectorN ERROR;

            //trash can for saving training data when it is removed during random training
            List<LabeledVectorN> Trash = new List<LabeledVectorN>();
            Random random = new Random();

            do
            {
                GoodFacts = 0;
                BadFacts = 0;
                Epochs++;

                while( TrainingData.Count > 0 )
                {
                    int index = random.Next(TrainingData.Count-1);

                    LabeledVectorN Input = TrainingData[index];
                    TrainingData.Remove(Input);
                    Trash.Add(Input);

                    OUTH = Activate( Input.Vector * InputHidden );
                    OUT = Activate( OUTH * HiddenOutput );

                    TARGET = Input.Label;

                    //calculate the error margin between expected and output values
                    ERROR = TARGET - OUT;

                    //check if back propagation is needed
                    for (int j = 0; j < ERROR.Dimensions; j++)
                    {
                        if (Math.Abs(ERROR[j]) > ErrorThreshold)
                        {
                            BadFacts++;
                            PerformEBP(TARGET, OUT, OUTH, Input.Vector);
                            break;
                        }
                        if (j == ERROR.Dimensions - 1) GoodFacts++;
                    }
                }

                TrainingData = Trash;
                Trash = new List<LabeledVectorN>();

                OnEpochCompleted(new EpochEvent(Epochs, GoodFacts, BadFacts));

                //determine the Minimum value reached
                if (PrevBadFacts == BadFacts)
                    StuckCount++;
                else
                    StuckCount = 0;

                PrevBadFacts = BadFacts;

            } while (BadFacts > 0 && Epochs<EpochsLimit && StuckCount<100);

            if( Epochs<EpochsLimit)
                trained = true;
            
            this.Epochs = Epochs;
        }

        public VectorN DetermineOutput(VectorN Input)
        {
            VectorN OUT;
            
            OUT = Activate( Input * InputHidden );
            OUT = Activate( OUT * HiddenOutput );

            return OUT;
        }

        #region Private Functions 

        private void OnEpochCompleted(EpochEvent args)
        {
            if (EpochCompleted != null)
                EpochCompleted(this, args);
        }

        private VectorN CalculateDeltaOutput(VectorN Target, VectorN Output)
        {
            VectorN Error = Target - Output;
            VectorN Delta = new VectorN(Target.Dimensions);

            //DeltaOut[i] = OUT[i] ( 1 - OUT[i] ) ( TARGET[i] - OUT[i] )
            for (int i = 0; i < Target.Dimensions; i++)
            {
                double value = Output[i]*(1.0-Output[i])*Error[i];
                Delta[i] = value;
            }

            return Delta;
        }

        private VectorN CalculateDeltaHidden(VectorN DeltaOutput, VectorN OutputHidden)
        {
            VectorN Result = new VectorN(OutputHidden.Dimensions);

            //DeltaHidden[i]  = OUT[i] ( 1 - OUT[i] ) SUM( DeltaOut[j] * HiddenOutput[i,j])
            for (int i = 0; i < Result.Dimensions; i++)
            {
                //determine Sum
                double sum = 0;
                for (int j = 0; j < DeltaOutput.Dimensions; j++)
                    sum += DeltaOutput[j] * HiddenOutput[i,j];

                Result[i] = OutputHidden[i] * (1.0 - OutputHidden[i]) * sum;
            }

            return Result;
        }

        //calls the currently set activation function on the values of the input vector
        private VectorN Activate(VectorN Vector)
        {
            for (int i = 0; i < Vector.Dimensions; i++)
                Vector[i] = ActivationFunction(Vector[i]);

            return Vector;
        }

        private void PerformEBP(VectorN Target, VectorN Output, VectorN OutputHidden, VectorN Input)
        {
            VectorN DeltaOutput = CalculateDeltaOutput(Target, Output);

            for (int i = 0; i < HiddenOutput.Rows; i++)
                for (int j = 0; j < HiddenOutput.Columns; j++)
                    HiddenOutput[i, j] += LearningRate * DeltaOutput[j] * OutputHidden[i];

            VectorN DeltaHidden = CalculateDeltaHidden(DeltaOutput, OutputHidden);

            for (int i = 0; i < InputHidden.Rows; i++)
                for (int j = 0; j < InputHidden.Columns; j++)
                    InputHidden[i, j] += LearningRate * DeltaHidden[j] * Input[i];
        }

        #endregion
    }
}
