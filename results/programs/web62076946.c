void compute(const char *input) {
    int n = 0;
    int i = 0;
    while (input[i]) {
        char c = input[i];
        if (c >= '0' && c <= '9') n = n + (c - '0');
        i = i + 1;
    }
    print_int(n);
    putchar('\n');
}
