void compute(const char *input) {
    int m = 0;
    int i = 0;
    while (input[i]) {
        char c = input[i];
        if (c >= '0' && c <= '9') {
            int d = c - '0';
            if (d > m) m = d;
        }
        i = i + 1;
    }
    print_int(m);
    putchar('\n');
}
