void compute(const char *input) {
    char out[42];
    int carry;
    int t;
    int d1;
    int d2;
    int s;
    int start;
    out[41] = 0;
    carry = 0;
    t = 0;
    while (t < 40) {
        d1 = input[39 - t] - '0';
        d2 = input[80 - t] - '0';
        s = d1 + d2 + carry;
        if (s >= 10) {
            s = s - 10;
            carry = 1;
        } else {
            carry = 0;
        }
        out[40 - t] = s + '0';
        t = t + 1;
    }
    out[0] = carry + '0';
    start = 0;
    while (start < 40) {
        if (out[start] == '0') {
            start = start + 1;
        } else {
            break;
        }
    }
    while (start <= 40) {
        putchar(out[start]);
        start = start + 1;
    }
    putchar('\n');
}
