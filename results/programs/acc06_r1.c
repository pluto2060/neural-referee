void compute(const char *input) {
    int n = 0;
    int i = 0;
    while (input[i]) {
        char c = input[i];
        n = n * 10 + (c - '0');
        i = i + 1;
    }
    print_int(n + 1);
    putchar('\n');
}
