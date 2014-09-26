#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

//For use in multiply()
int min(int a, int b){return a<b ? a:b;}
int max(int a, int b){return a>b ? a:b;}

typedef struct{
    int* row_ptr;
    int* col_ind;
    int n_values;
    int n_row_ptr;
    float* values;
} csr_matrix_t;

typedef struct{
    float* values;  //The actual non-zero values
    int* offset;    //Offsets of diagonals in the array "values"
    int amnt;       //Amount of diagonals
    int* diag_id;   //Position of diagonal line in full matrix
} s_matrix_t;

typedef s_matrix_t* DiagMatrix;
typedef csr_matrix_t* CSRMatrix;

int diag_count(int dim, int n){
    return n*dim -
           ((n*(n+1))/2);
}

void print_raw_csr_matrix(csr_matrix_t* m){
    printf("row_ptr = {");
    for(int i = 0; i < m->n_row_ptr; i++)
        printf("%d ", m->row_ptr[i]);
    printf("}\n");

    printf("col_ind = {");
    for(int i = 0; i < m->n_values; i++)
        printf("%d ", m->col_ind[i]);
    printf("}\n");

    printf("values = {");
    for(int i = 0; i < m->n_values; i++)
        printf("%f ", m->values[i]);
    printf("}\n");
}

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
void print_formated_csr_matrix(csr_matrix_t* m){
    for(int i = 0; i < m->n_row_ptr-1; i++){
        int col = m->row_ptr[i];
        for(int j = 0; j < m->n_row_ptr-1; j++){
            if(j == m->col_ind[col] && col < m->row_ptr[i+1]){
                printf("%s%.2f ", KRED, m->values[col]);
                //printf("%.2f ", m->values[col]);
                col++;
            }
            else{
                printf("%s%.2f ", KNRM, 0.0);
                //printf("%.2f ", KNRM, 0.0);
            }
        }
        printf("%s\n", KNRM);
        //printf("\n");
    }
}

csr_matrix_t* create_csr_matrix(int n_rows, int n_cols, int a, int b, int c, int d, int e){

    csr_matrix_t* matrix = (csr_matrix_t*)malloc(sizeof(csr_matrix_t));

    matrix->row_ptr = (int*)malloc(sizeof(int) * (n_rows+1));
    matrix->n_row_ptr = n_rows+1;

    int ah = a/2;
    int size = diag_count(n_rows, ah);
    size += (diag_count(n_rows, ah+b+c) - diag_count(n_rows, ah+b));
    size += (diag_count(n_rows, ah+b+c+d+e) - diag_count(n_rows, ah+b+c+d));
    size = size*2 + n_rows;

    matrix->col_ind = (int*)malloc(sizeof(int)*size);
    matrix->values = (float*)malloc(sizeof(float)*size);
    matrix->n_values = size;

    int limits[10];
    limits[5] = ah;
    limits[6] = ah + b;
    limits[7] = ah + b + c;
    limits[8] = ah + b + c + d;
    limits[9] = ah + b + c + d + e;
    limits[0] = -limits[9];
    limits[1] = -limits[8];
    limits[2] = -limits[7];
    limits[3] = -limits[6];
    limits[4] = -limits[5];

    limits[5]++;
    limits[6]++;
    limits[7]++;
    limits[8]++;
    limits[9]++;

    int index = 0;
    int index2 = 0;
    int index3 = 0;
    matrix->row_ptr[0] = 0;
    for(int i = 0; i < n_rows; i++){

        int row_width = index;
        for(int j = fmax(0, limits[0]); j < fmax(0, limits[1]); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmax(0, limits[2]); j < fmax(0, limits[3]); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmax(0, limits[4]); j < fmin(limits[5], n_cols); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmin(n_cols, limits[6]); j < fmin(n_cols, limits[7]); j++)
            matrix->col_ind[index++] = j;

        for(int j = fmin(n_cols, limits[8]); j < fmin(n_cols, limits[9]); j++)
            matrix->col_ind[index++] = j;

        row_width = index - row_width;
        matrix->row_ptr[index2+1] = matrix->row_ptr[index2] + row_width;
        index2++;

        for(int j = 0; j < row_width; j++)
            matrix->values[index3++] = (float)rand()/RAND_MAX;

        for(int j = 0; j < 10; j++)
            limits[j]++;
    }

    return matrix;
}

float* create_vector(int n){
    float* v = (float*)malloc(sizeof(float)*n);
    for(int i = 0; i < n; i++){
        v[i] = (float)rand()/RAND_MAX;
    }
    return v;
}

void print_vector(float* v, int n, int orientation){
    for(int i = 0; i < n; i++){
        printf("%f%s", v[i], orientation ? " " : "\n");
    }

    if(orientation)
        printf("\n");
}

double print_time(struct timeval start, struct timeval end){
    long int ms = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    double s = ms/1e6;
    printf("Time (ms): %f s\n", s);
    return s;
}

void multiply_naive(csr_matrix_t* m, float* v, float* r){
    for(int i = 0; i < m->n_row_ptr-1; i++){
        for(int j = m->row_ptr[i]; j < m->row_ptr[i+1]; j++){
            r[i] += v[m->col_ind[j]] * m->values[j];
        }
    }
}

void compare(float* a, float* b, int n){
    int n_errors = 0;
    for(int i = 0; i < n; i++){
        if(fabs(a[i] - b[i]) > 1e-4){
            n_errors++;
            if(n_errors < 40){
                printf("Error at: %d, expected: %f, actual: %f\n", i, a[i], b[i]);
            }
        }
    }
    if (10 <= n_errors){
        printf("%d errors...\n", n_errors);
    }
}

DiagMatrix create_s_matrix(int dim, int a, int b, int c, int d, int e){
    int cntr = 1, a_half = (a-1)/2;
    DiagMatrix result = (DiagMatrix) malloc(sizeof(s_matrix_t));
    result->amnt = a+2*(c+e);
    result->diag_id = (int*) malloc(sizeof(int)*result->amnt);
    result->offset = (int*) malloc(sizeof(int)*(result->amnt+1));
    result->values = (float*) malloc(sizeof(float)*result->offset[result->amnt]);

    //Setting the first diagonal id and offset, the biggest one of them all
    result->offset[0] = 0;
    result->diag_id[0] = 0;
    int smallMat[3] = {a_half, c, e}; //How wide each band of diagonals are (two sets of these bands, above and below diagonal 0)
    int smallOffset[3] = {1, 1+a_half+b, 1+a_half+b+c+d}; //Offset from diagonal 0 on where each diagonal band starts

    //Then setting the diagonal IDs
    for (int i = 0; i < 3; ++i){
        for (int j = 0; j < smallMat[i]; ++j){
            for (int x = -1; x < 2; x += 2){
               result->diag_id[cntr] = (smallOffset[i]+j)*x;
               cntr++;
            }
        }
    }

    //Set the offsets in values for each diagonal
    result->offset[0] = 0; cntr = 0;
    for (int i = 1; i < result->amnt+1; ++i){
        cntr += dim-abs(result->diag_id[i-1]);
        result->offset[i] = cntr;
        //printf("cntr: %d\noffset[%d]: %d\n", cntr, i-1, result->offset[i-1]);
    }

    printf("\nvalue size: %d\n", result->offset[result->amnt]);

    printf("diag_id:\n");
    for (int i = 0; i < result->amnt; ++i){
        printf("%d, ", result->diag_id[i]);
    }

    printf("\noffset:\n");
    for (int i = 0; i < result->amnt+1; ++i){
        printf("%d, ", result->offset[i]);
    }
    printf("\n\n");

    return result;
}

DiagMatrix convert_to_s_matrix(DiagMatrix mat, csr_matrix_t* csr){
    int col, dims = mat->offset[0]+1,
        diag, offset, cntr = 0, diag_index, test = 0;
    for (int row = 0; row < csr->n_row_ptr-1; ++row){//For each row of the matrix (CSR format)
        for (int row_element = csr->row_ptr[row]; row_element < csr->row_ptr[row+1]; ++row_element){//For each non-zero element in row (CSR format)
            col = csr->col_ind[row_element]; diag = (row-col); //Which diagonal does current element belong to?
            offset = min(row, col); //At what number inside the diagonal does this element appear?
            //Find out which number (X) in mat->offset[X] the diagonal is
            for (int i = 0; i < mat->amnt; ++i){
                if(mat->diag_id[i] == diag){
                    test++; diag_index = i; break;
                }
            }

            if (0 == row){
                printf("Transferring %.2f from csr to position %d in diag nr %d\n", csr->values[row_element], offset, diag_index);
                printf("cntr: %d, row: %d, col: %d, offset: %d, diag_index:%d, row_element: %d\n\n",
                    cntr, row, col, offset+mat->offset[diag_index], diag_index, row_element); cntr++;
            }

            //Do transfer of value
            mat->values[mat->offset[diag_index]+offset] = csr->values[row_element];
        }
        printf("\n");
    }
    return mat;
}

void multRes(float* restrict res, float* restrict diag, float* restrict vec,
    int start, int stop, int col_offset, int row_offset){
    for (int i = start; i < stop; ++i){
        res[i-start+row_offset] += vec[i-start+col_offset]*diag[i];
    }
}

void multiply(DiagMatrix m, float* v, float* r){
    int start, end, row_offset, col_offset;
    for (int i = 0; i < m->amnt; ++i){//Iterate over diagonals and calculate constants for inner for-loop
        //do I need to switch the min() and max() below?
        row_offset = min(0, m->diag_id[i]);//If diag starts with values on row 0, displace r
        col_offset = max(0, m->diag_id[i]);//If diag starts with values on a different row than row 0, displace v
        end = m->offset[i+1]; start = m->offset[i];
        multRes(r, m->values, v, start, end, col_offset, row_offset);
    }
}

int main(int argc, char** argv){
    if(argc != 7){
        printf("useage %s dim a b c d e\n", argv[0]);
        exit(-1);
    }

    int a = atoi(argv[2]); int b = atoi(argv[3]); int c = atoi(argv[4]);
    int d = atoi(argv[5]); int e = atoi(argv[6]); int dim = atoi(argv[1]);

    csr_matrix_t* m = create_csr_matrix(dim, dim, a, b, c, d, e);
    print_formated_csr_matrix(m);
    printf("csr->n_values: %d\n", m->n_values);
    //printf("Printing csr->row_ptr:\n");
    for (int i = 0; i < m->n_row_ptr-1; ++i){
        //printf("\nrow_ptr[%d]: %d, row_ptr[%d]: %d\n", i, m->row_ptr[i], i+1, m->row_ptr[i+1]);
        for (int j = m->row_ptr[i]; j < m->row_ptr[i+1]; ++j){
            //printf("%d\t", m->col_ind[j]);
        }
    }

    struct timeval start, end;
    float* v = create_vector(dim);
    float* r1 = (float*)calloc(dim, sizeof(float));
    float* r2 = (float*)calloc(dim, sizeof(float));

    gettimeofday(&start, NULL); multiply_naive(m, v, r1);
    gettimeofday(&end, NULL); print_time(start, end);

    //printf("Entering create_s_matrix\n");
    DiagMatrix s = create_s_matrix(dim, a, b, c, d, e);

    printf("Entering convert_to_s_matrix\n");
    convert_to_s_matrix(s, m);
    printf("Exited convert_to_s_matrix\n\n");


    for (int i = 0; i < s->amnt; ++i){
        printf("Diag nr. %d:\n", i);
        for (int j = s->offset[i]; j < s->offset[i+1]; ++j){
            printf("%.2f ", s->values[j]);
        }
        printf("\n");
    }

    printf("Entering multiply\n");
    gettimeofday(&start, NULL); multiply(s, v, r2);
    gettimeofday(&end, NULL); print_time(start, end);

    compare(r1, r2, dim);
}
