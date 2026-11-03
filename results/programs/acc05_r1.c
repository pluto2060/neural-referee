void compute(const char *input) {
    int max = 0;
    int i = 0;
    while (input[i]) {
        char c = input[i];
        int digit = c - '0';
        if (digit > max) max = digit;
        i = i + 1;
    }
    print_int(max);
    putchar('\n');
}
