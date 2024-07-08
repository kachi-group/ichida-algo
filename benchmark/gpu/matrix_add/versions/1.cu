#include "../template.cuh"

matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

matrix* new_matrix_d(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->cols = cols;
    cudaMalloc((void**)&(res->data), rows * cols * sizeof(float));
    return res;
}

__global__ void matrix_add(float *a, float*b ,int rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<rows){
        a[idx]+=b[idx];
    }
}

double time(int n) {
    int row=100000;
    matrix* a = new_matrix_d(row, 1);
    matrix* b = new_matrix_d(row, 1);
    cudaStream_t stream1;
    cudaStreamCreate ( &stream1);
    
    int thread=1024;
    int block=((row+thread-1)/thread);

    clock_t start = clock();
    for(int i=0;i<n;i++){
        matrix_add<<<1,1,0,stream1>>>(a->data,b->data,row);
    }
    double seconds = (double)(clock() - (double)start) / CLOCKS_PER_SEC;
    return seconds;
}