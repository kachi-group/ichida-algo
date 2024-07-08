#include "template.cuh"
#include <stdio.h>
#include <time.h>

int main(int argc, char* argv[]) {
    long n;
    if (argc > 1) {
        n = atol(argv[1]);
    } else {
        n = 100000;
    }
    printf("%f", time(n));
}