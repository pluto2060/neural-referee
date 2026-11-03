void compute(const char *input) {
    int n = parse_int(input);
    int sum = 0;
    for (int i = 1; i <= n; i++) {
        sum = sum + i;
    }
    print_int(sum);
    putchar('\n');
}
