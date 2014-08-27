#include <stdio.h>
#include <stdlib.h>

typedef struct{
  //Add needed fields
   **data;//The actual data of the matrix
  int rows;     //The amount of rows in the matrix
  int cols;     //The amount of coloumns in the matrix
} matrix_t;

typedef matrix_t *Matrix;


matrix_t* new_matrix(int rows, int cols){
  Matrix result = (Matrix) calloc(1, sizeof(matrix_t));

  result->rows = rows;
  result->cols = cols;  
  result->data = (float **) calloc(rows, sizeof(float*));
  result->data[0] = (float *) calloc(rows*cols, sizeof(float));

  for (int i = 0; i < cols; ++i){
    result->data[i] = result->data[i-1] + rows;
  }

  return result;
}


void print_matrix(matrix_t* matrix){
  Matrix *ptr = &matrix;
  for (int i = 0; i < ptr->rows; ++i){
    printf("[%f", ptr[i][0]);
    for (int j = 1; j < ptr->cols; ++j){
      printf(",\t%f", ptr[i][j]);
    }
    printf("]\n");
  }
}


void set_value(matrix_t* matrix, int row, int col, float value){
  matrix->data[row][col] = value;
}


float get_value(matrix_t* matrix, int row, int col){
  return matrix->data[row][col];
}

int is_sparse(matrix_t matrix, float sparse_threshold){
  double zero_elements = 0, non-zeros = 0;

  for (int i = 0; i < matrix->rows; ++i){
    for (int j = 0; j < matrix->cols; ++j){
      if (matrix->data[i][j] == 0.0){
        zero_elements += 1;
      } else{
        non-zeros += 1;
      }
    }
  }

  if ((zero_elements/non-zeros) > sparse_threshold){
    return 1;
  } else{
    return 0;
  }
}


int matrix_multiply(matrix_t* a, matrix_t* b, matrix_t** c){
  if(a->cols != b->rows){ //If the number of coloumns in the first matrix is not equal to the number of rows in the second matrix, 
    **c = 0;              //Set c to zero
    return -1;            //Return -1;
  } else{ //Else, create the matrix "C"
    *c = new_matrix(a->rows, b->cols);
  }

  //First iterate over each element in matrix C.
  for (int i = 0; i < c->rows*c->cols; ++i){
    (*c)->data[i] = 0.0;
    //Then start calculating the dot-product of the j-th row in A, 
    for (int j = 0; j < a->rows; ++j){
      //And k-th coloumn in B
      for (int k = 0; k < a->cols; ++k){
        //Add up the products to make the dot-product in the corresponding C indice.
        (*c)->data[i] += a->data[j][k]*b->data[k][j];
      }
    }
  }

  return 1;
}


void change_size(matrix_t* matrix, int new_rows, int new_cols){
  //Create the new matrix
  Matrix new_mat = new_matrix(new_rows, new_cols);
  //The size of the sub-matrix whose values are kept and copied over to the new matrix
  int min_rows, min_cols;

  //Find the value for the sub-matrix's row-size
  matrix->rows <= new_rows ? min_rows = matrix->rows : min_rows = new_rows;
  //Find the value for the sub-matrix's coloumn-size
  matrix->cols <= new_cols ? min_cols = matrix->cols : min_cols = new_cols;

  //Iterate over the sub-matrix to copy over the values that are to be kept.
  for (int i = 0; i < min_rows; ++i){
    for (int j = 0; j < min_cols; ++j){
      new_mat->data[i][j] = matrix->data[i][j];
    }
  }

  //"Delete" the old matrix by freeing it.
  free_matrix(matrix);
  //Copy over the address of the new matrix to the pointer.
  *matrix = &new_matrix;
}


void free_matrix(matrix_t* matrix){
  free(matrix->data[0]);
  free(matrix->data);
  free(matrix);
}


int main(int argc, char** argv){
  
  // Create and fill matrix m
  matrix_t* m = new_matrix(3,4);
  for(int row = 0; row < 3; row++){
    for(int col = 0; col < 4; col++){
      set_value(m, row, col, row*10+col);
    }
  }
  
  // Create and fill matrix n
  matrix_t* n = new_matrix(4,4);
  for(int row = 0; row < 4; row++){
    for(int col = 0; col < 4; col++){
      set_value(n, row, col, col*10+row);
    }
  }
  
  // Create and fill matrix o
  matrix_t* o = new_matrix(5,5);
  for(int row = 0; row < 5; row++){
    for(int col = 0; col < 5; col++){
      set_value(o, row, col, row==col? 1 : 0);
    }
  }
  // Printing matrices
  printf("Matrix m:\n");
  print_matrix(m);
  /*
  Should print:
  0.00 1.00 2.00 3.00 
  10.00 11.00 12.00 13.00 
  20.00 21.00 22.00 23.00
  */
  
  printf("Matrix n:\n");
  print_matrix(n);
  /*
  Should print:
  0.00 10.00 20.00 30.00 
  1.00 11.00 21.00 31.00 
  2.00 12.00 22.00 32.00 
  3.00 13.00 23.00 33.00 
  */
  
  
  printf("Matrix o:\n");
  print_matrix(o);
  /*
  Should print:
  1.00 0.00 0.00 0.00 0.00 
  0.00 1.00 0.00 0.00 0.00 
  0.00 0.00 1.00 0.00 0.00 
  0.00 0.00 0.00 1.00 0.00 
  0.00 0.00 0.00 0.00 1.00
  */
  
  // Checking if matrices are sparse (more than 75% 0s)
  printf("Matrix m is sparse: %d\n", is_sparse(*m, 0.75)); // Not sparse, should print 0
  printf("Matrix o is sparse: %d\n", is_sparse(*o, 0.75)); // Sparse, should print 1
  
  
  // Attempting to multiply m and o, should not work
  matrix_t* p;
  int error = matrix_multiply(m,o,&p);
  printf("Error (m*o): %d\n", error); // Should print -1 
 
  // Attempting to multiply m and n, should work
  error = matrix_multiply(m,n,&p);
  print_matrix(p);
  /*
  Should print:
  14.00 74.00 134.00 194.00 
  74.00 534.00 994.00 1454.00 
  134.00 994.00 1854.00 2714.00 
  */
  
  // Shrinking m, expanding n
  change_size(m, 2,2);
  change_size(n, 5,5);
  
  printf("Matrix m:\n");
  print_matrix(m);
  /*
  Should print:
  0.00 1.00 
  10.00 11.00 
  */
  printf("Matrix n:\n");
  print_matrix(n);
  /*
  Should print:
  0.00 10.00 20.00 30.00 0.00 
  1.00 11.00 21.00 31.00 0.00 
  2.00 12.00 22.00 32.00 0.00 
  3.00 13.00 23.00 33.00 0.00 
  0.00 0.00 0.00 0.00 0.00
  */
  
  // Freeing memory
  free_matrix(m);
  free_matrix(n);
  free_matrix(o);
  free_matrix(p);
}
