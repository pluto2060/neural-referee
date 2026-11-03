void compute(const char *input) {
    int n = parse_int(input);
    int steps = 0;
    while (n > 1) {
        int tmp = n;
        int half = 0;
        while (tmp > 1) {
            tmp = tmp - 2;
            half = half + 1;
        }
        if (tmp == 0) {
            n = half;
        } else {
            n = n + n + n + 1;
        }
        steps = steps + 1;
    }
    print_int(steps);
    putchar('\n');
}
