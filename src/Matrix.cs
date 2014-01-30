using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork
{
    public class Matrix
    {
        public double[,] Values;

        public int Rows {
            get { return Values.GetLength(0); }
        }
        public int Columns {
            get { return Values.GetLength(1); }
        }

        //rows,columns
        public double this[int i, int j]
        {
            get { return Values[i,j]; }
            set { Values[i,j] = value; }
        }
 
        public Matrix( int Rows, int Columns )
        {
            Values = new double[Rows,Columns];
        }

        public static VectorN operator *(VectorN Vector, Matrix M)
        {
            if (Vector.Dimensions != M.Rows)
                throw new ArgumentException("Vector Size must be equal to Matrix Rows Count");

            VectorN Output = new VectorN(M.Columns);

            for (int i = 0; i < M.Columns; i++)
                for (int j = 0; j < M.Rows; j++)
                    Output[i] += M[j, i] * Vector[j];

            return Output;
        }

        public static Matrix operator *(Matrix M1, Matrix M2)
        {
            if (M1.Columns != M2.Rows)
                throw new ArgumentException("Matrix M1 and Matrix M2 cannot be multipled");

            Matrix Output = new Matrix(M1.Rows, M2.Columns);

            for (int i = 0; i < M1.Rows; i++)
            {
                for (int j = 0; j < M1.Columns; j++)
                {
                    for (int k = 0; k < M1.Columns; k++)
                    {
                        Output[i, j] += M1[i, k] * M2[k, j];
                    }
                }
            }

            return Output;
        }

        public static Matrix operator *(Matrix M, double Value)
        {
            Matrix Output = new Matrix(M.Rows, M.Columns);

            Parallel.For(0, M.Rows, i =>
            {
                Parallel.For(0, M.Columns, j =>
                {
                    Output[i, j] = M[i, j] * Value;
                });
            });

            return Output;
        }

        public static Matrix operator +(Matrix M1, Matrix M2)
        {
            if( M1.Rows != M2.Rows || M1.Columns != M1.Columns )
                throw new ArgumentException("Matrix addition needs two matrices of equal height and width" );

            Matrix Output = new Matrix(M1.Rows,M1.Columns);

            for( int i=0; i<M1.Values.GetLength(0); i++ )
            {
                for( int j=0; j<M1.Values.GetLength(1); j++ )
                {
                    Output[i,j] = M1[i,j] + M2[i,j];
                }
            }

            return Output;
        }

        public static Matrix operator -(Matrix M1, Matrix M2)
        {
            if (M1.Rows != M2.Rows || M1.Columns != M1.Columns)
                throw new ArgumentException("Matrix addition needs two matrices of equal height and width");

            Matrix Output = new Matrix(M1.Rows, M1.Columns);

            for (int i = 0; i < M1.Values.GetLength(0); i++)
            {
                for (int j = 0; j < M1.Values.GetLength(1); j++)
                {
                    Output[i, j] = M1[i, j] - M2[i, j];
                }
            }

            return Output;
        }

        public static Matrix IdentityMatrix(int size)
        {
            Matrix Output = new Matrix(size, size);

            for (int i = 0; i < size; i++)
                Output[i, i] = 1;

            return Output;
        }

        public static Matrix RandomMatrix(int Rows, int Columns)
        {
            Matrix Output = new Matrix(Rows, Columns);
            Random random = new Random((int)DateTime.Now.Ticks);

            for (int i = 0; i < Output.Rows; i++)
                for (int j = 0; j < Output.Columns; j++)
                    Output[i, j] = random.NextDouble()*2-1;

            return Output;
        }

        public override string ToString()
        {
            StringBuilder Builder = new StringBuilder();

            for( int i=0; i<Rows; i++)
            {
                for( int j=0; j<Columns; j++ )
                {
                    Builder.Append(this[i, j]);
                    if (j < Columns - 1)
                        Builder.Append(",");
                }
                Builder.Append("\n");
            }

            return Builder.ToString();
        }
    }
}
