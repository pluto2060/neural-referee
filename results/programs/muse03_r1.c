void compute(const char *input) {
    int n = str_len(input);
    int i = n - 1;
    while (i >= 0) {
        putchar(input[i]);
        i = i - 1;
    }
    putchar('\n');
}
