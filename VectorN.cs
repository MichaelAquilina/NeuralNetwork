using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace ArtificialNeuralNetwork
{
    //N-Dimensional Vector with Double scalar values
    public class VectorN
    {
        private double[] Values;

        public int Dimensions 
        {
            get { return Values.Length; }
        }

        public double this[int index]
        {
            get { return Values[index]; }
            set { Values[index] = value; }
        }

        public VectorN(int Dimensions)
        {
            Values = new double[Dimensions];
        }

        //returns the index of the max value - a convenience function
        public int MaxValueIndex
        {
            get
            {
                double Max = Double.MinValue;
                int index = -1;
                for (int i = 0; i < Dimensions; i++)
                {
                    if (this[i] > Max)
                    {
                        Max = this[i];
                        index = i;
                    }
                }

                return index;
            }
        }

        #region Overriden Methods

        public override string ToString()
        {
            StringBuilder Builder = new StringBuilder();
            Builder.Append("[ ");

            for (int i = 0; i < this.Dimensions; i++)
            {
                Builder.Append(this[i].ToString("0.0"));
                if( i < Dimensions-1 ) Builder.Append(", ");
            }

            Builder.Append(" ]");
            return Builder.ToString();
        }

        public override int GetHashCode()
        {
            int sum = 0;
            for (int i = 0; i < this.Dimensions; i++)
                sum += (int) this[i];

            return sum;
        }

        public override bool Equals(object obj)
        {
            if (obj is VectorN)
            {
                VectorN vector = (VectorN)obj;
                return this == vector;
            }
            else
                throw new ArgumentException("Can only compare vectors with other vectors");
        }

        #endregion

        #region Operators

        public static VectorN operator *(VectorN vector, double scalar)
        {
            VectorN Output = new VectorN(vector.Dimensions);
            for (int i = 0; i < Output.Dimensions; i++)
                Output[i] = vector[i] * scalar;

            return Output;
        }

        public static double operator *(VectorN vector1, VectorN vector2)
        {
            if (vector1.Dimensions != vector2.Dimensions)
                throw new ArgumentException("Vectors must be of the same dimension");

            double Output = 0;

            for (int i = 0; i < vector1.Dimensions; i++)
                Output += vector1[i] * vector2[i];

            return Output;
        }

        public static VectorN operator +(VectorN vector1, VectorN vector2)
        {
            if (vector1.Dimensions != vector2.Dimensions)
                throw new ArgumentException("Vectors must be of the same dimension");

            VectorN Output = new VectorN( vector1.Dimensions );

            for (int i = 0; i < vector1.Dimensions; i++)
                Output[i] = vector1[i] + vector2[i];

            return Output;
        }

        public static VectorN operator -(VectorN vector1, VectorN vector2)
        {
            if (vector1.Dimensions != vector2.Dimensions)
                throw new ArgumentException("Vectors must be of the same dimension");

            VectorN Output = new VectorN(vector1.Dimensions);

            for (int i = 0; i < vector1.Dimensions; i++)
                Output[i] = vector1[i] - vector2[i];

            return Output;
        }

        public static Boolean operator ==(VectorN vector1, VectorN vector2)
        {
            if (vector1.Dimensions != vector2.Dimensions)
                return false;

            for (int i = 0; i < vector1.Dimensions; i++)
                if (vector1[i] != vector2[i])
                    return false;

            return true;
        }

        public static Boolean operator !=(VectorN vector1, VectorN vector2)
        {
            return (vector1==vector2)==false;
        }

        #endregion

        public static double Distance(VectorN vector1, VectorN vector2)
        {
            if (vector1.Dimensions != vector2.Dimensions)
                throw new ArgumentException("Both vectors need to be of the same dimension");

            double total = 0;
            for (int i = 0; i < vector1.Dimensions; i++)
                total += Math.Pow(vector1[i] - vector2[i], 2);

            return Math.Sqrt(total);
        }

        public static VectorN ConstantVector(double scalar, int dimensions)
        {
            VectorN Output = new VectorN(dimensions);
            for (int i = 0; i < Output.Dimensions; i++)
                Output[i] = scalar;

            return Output;
        }
    }

    //N-Dimensional Vector with a Labeled output value
    public class LabeledVectorN
    {
        public VectorN Label;
        public VectorN Vector;

        public LabeledVectorN(VectorN Vector, VectorN Label)
        {
            this.Vector = Vector;
            this.Label = Label;
        }

        #region Overriden Methods

        public override int GetHashCode()
        {
            return Vector.GetHashCode() + Label.GetHashCode();
        }

        public override bool Equals(object obj)
        {
            if (obj is LabeledVectorN)
            {
                LabeledVectorN vector = (LabeledVectorN)obj;
                return this == vector;
            }
            else
                throw new ArgumentException("Can only compare LabeledVectors with other LabeledVectors");
        }

        #endregion

        #region Operators

        public static Boolean operator ==(LabeledVectorN vector1, LabeledVectorN vector2)
        {
            if (vector1.Vector != vector2.Vector)
                return false;

            if (vector1.Label != vector2.Label)
                return false;

            return true;
        }

        public static Boolean operator !=(LabeledVectorN vector1, LabeledVectorN vector2)
        {
            return (vector1 == vector2) == false;
        }

        #endregion
    }
}
