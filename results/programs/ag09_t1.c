void compute(const char *input) {
    int n = parse_int(input);
    int steps = 0;
    int q = 0;
    int r = 0;
    while (n != 1) {
        q = 0;
        r = n;
        while (r >= 2) {
            r = r - 2;
            q = q + 1;
        }
        if (r == 0) {
            n = q;
        } else {
            n = n + n + n + 1;
        }
        steps = steps + 1;
    }
    print_int(steps);
    putchar('\n');
}
