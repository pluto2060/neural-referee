void compute(const char *input) {
    int n = 0;
    int i = 0;
    while (input[i]) {
        if (input[i] == 'e') n = n + 1;
        i = i + 1;
    }
    print_int(n);
    putchar('\n');
}
