void compute(const char *input) {
    int n = parse_int(input);
    int s = 0;
    int i = 1;
    while (i <= n) {
        s = s + i;
        i = i + 1;
    }
    print_int(s);
    putchar('\n');
}
