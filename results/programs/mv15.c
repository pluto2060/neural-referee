void compute(const char *input) {
    int n = parse_int(input);
    int steps = 0;
    int r;
    int half;
    while (n > 1) {
        r = n;
        half = 0;
        while (r > 1) {
            r = r - 2;
            half = half + 1;
        }
        if (r == 0) {
            n = half;
        } else {
            n = n + n + n + 1;
        }
        steps = steps + 1;
    }
    print_int(steps);
    putchar('\n');
}
