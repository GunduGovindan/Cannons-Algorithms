#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include <math.h>
#include <time.h>

//Code for Serial Matrix Multiplication
void MatrixMultiply(int n, double *a, double *b, double *c){
    int i, j, k;
    for (i=0; i<n; i++){
        for (j=0; j<n; j++){
            for (k=0; k<n; k++){
                c[i*n+j] += a[i*n+k]*b[k*n+j];
            }
        }
    }
} 


//Code for Cannons Multiplication
void MatrixMatrixMultiply(int n, double *a, double *b, double *c,MPI_Comm comm){
    int i;
    int nlocal;
    int npes, dims[2], periods[2];
    int myrank, my2drank, mycoords[2];
    int uprank, downrank, leftrank, rightrank, coords[2];
    int shiftsource, shiftdest;
    MPI_Status status;
    MPI_Comm comm_2d;
    MPI_Comm_size(comm, &npes);
    MPI_Comm_rank(comm, &myrank);
    dims[0] = dims[1] = sqrt(npes);
    periods[0] = periods[1] = 1;
    MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);
    MPI_Comm_rank(comm_2d, &my2drank);
    MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);
    MPI_Cart_shift(comm_2d, 0, -1, &rightrank, &leftrank);
    MPI_Cart_shift(comm_2d, 1, -1, &downrank, &uprank);
    nlocal = n/dims[0];
    MPI_Cart_shift(comm_2d, 0, -mycoords[0], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, shiftdest,
    1, shiftsource, 1, comm_2d, &status);
    MPI_Cart_shift(comm_2d, 1, -mycoords[1], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
    shiftdest, 1, shiftsource, 1, comm_2d, &status);
    for (i=0; i<dims[0]; i++) {
        MatrixMultiply(nlocal, a, b, c); /*c=c+a*b*/
        MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE,
        leftrank, 1, rightrank, 1, comm_2d, &status);
        MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
        uprank, 1, downrank, 1, comm_2d, &status);
    }
    MPI_Cart_shift(comm_2d, 0, +mycoords[0], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE,
    shiftdest, 1, shiftsource, 1, comm_2d, &status);

    MPI_Cart_shift(comm_2d, 1, +mycoords[1], &shiftsource, &shiftdest);
    MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
    shiftdest, 1, shiftsource, 1, comm_2d, &status);

    MPI_Comm_free(&comm_2d);
} 




#define ROWS 350
#define COLS 350
#define SIZE 350*350

int main(int argc,char *argv[])
{

    //Reading tensor of the Input Image
    FILE *myFile;
    printf("Here");
    myFile = fopen("image_data_gray_350.txt", "r");

    if (myFile == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    // Determine the number of numbers in the file (assuming one number per line)
    int numNumbers = 0;
    double currentNumber;
    while (fscanf(myFile, "%lf", &currentNumber) == 1) {
        numNumbers++;
    }

    // Allocate memory for the 1D double array
    double numberArray[numNumbers];
    if (numberArray == NULL) {
        printf("Memory allocation failed.\n");
        fclose(myFile);
        return 1;
    }

    // Rewind the file to the beginning
    rewind(myFile);
    // Read the numbers into the array
    for (int i = 0; i < numNumbers; i++) {
        fscanf(myFile, "%lf", &numberArray[i]);
    }

    // Clean up: close the file and free the allocated memory
    fclose(myFile);



    //CREATING KERNEL


    // Create a 2D array with required dimensions
    double twoDArray[ROWS][COLS];
    
    // Initialize the 2D array with kernel values
    int count=0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            if (i==j){
                if (count==2){
                twoDArray[i][j]=1;
                count=0;
                }
                else{
                    count+=1;
                }
                
            }
            else{
                twoDArray[i][j]=0;

            }

        }
    }

    // Flatten the 2D array into a 1D array
    double oneDArray[SIZE];
    int index = 0;
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            oneDArray[index++] = twoDArray[i][j];
        }
    }
    double oneDDArray[SIZE];
    for (int i=0;i<SIZE;i++){
        oneDDArray[i]=0;
    }

    int n=350;

    //Initialize the MPI variables
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Init(&argc, &argv);

    //Start the timer
    double start_time = MPI_Wtime();
    //Call cannons matrix multiplication function
    MatrixMatrixMultiply(n, numberArray,oneDArray,oneDDArray,comm);
    printf("\nResult Matrix (A * B):C\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", oneDDArray[i * n + j]);
        }
        printf("\n");
    }
    MPI_Finalize();
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    printf("Time taken:%f",elapsed_time);

    //Serial Matrix Multiplication Code
    /*printf("SERIAL_____");
    start_time = MPI_Wtime();
    MatrixMultiply(n,a,b,c);
    printf("\nResult Matrix (A * B):C\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%lf ", c[i * n + j]);
        }
        printf("\n");
    }
    end_time = MPI_Wtime();
    elapsed_time = end_time - start_time;
    printf("Time taken:%f",elapsed_time);*/

    //Write the output image into a text file as tensors
    FILE *fp = fopen("output/output_grey.txt", "w");
    for (int i = 0; i < n*n; i++) {
    fprintf(fp, "%.2lf\n", oneDDArray[i]); // Format with 2 decimal places (adjust as needed)
    }
    fclose(fp);
    return 0;
}
