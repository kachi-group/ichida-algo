void relu(float* dest, int rows) {
    for (int i = 0; i < rows; i++) {
        if (dest[i] < 0)
            dest[i] = 0;
    }
}
