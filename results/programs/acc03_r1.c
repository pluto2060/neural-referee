void compute(const char *input) {
    int len = str_len(input);
    int i = len - 1;
    while (i >= 0) {
        print_str(input + i);
        putchar('\n');
        i = i - 1;
    }
}
