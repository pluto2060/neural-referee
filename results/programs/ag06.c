void compute(const char *input) {
    int L;
    int p;
    int i1;
    int i2;
    int carry;
    int k;
    int d1;
    int d2;
    int s;
    int digit;
    char out[21];
    L = str_len(input);
    p = 0;
    while (input[p] != '+') {
        p = p + 1;
    }
    i1 = p - 1;
    i2 = L - 1;
    carry = 0;
    k = 0;
    while (i1 >= 0 || i2 > p || carry) {
        d1 = 0;
        d2 = 0;
        if (i1 >= 0) {
            d1 = input[i1] - '0';
            i1 = i1 - 1;
        }
        if (i2 > p) {
            d2 = input[i2] - '0';
            i2 = i2 - 1;
        }
        s = d1 + d2 + carry;
        if (s >= 10) {
            digit = s - 10;
            carry = 1;
        } else {
            digit = s;
            carry = 0;
        }
        out[k] = digit + '0';
        k = k + 1;
    }
    while (k > 0) {
        k = k - 1;
        putchar(out[k]);
    }
    putchar('\n');
}
