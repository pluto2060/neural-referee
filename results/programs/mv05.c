void compute(const char *input) {
    int len;
    int pos;
    int i;
    int j;
    int r;
    int carry;
    int res[41];
    int da;
    int db;
    int s;
    int d;
    int k;
    len = str_len(input);
    pos = 0;
    while (pos < len) {
        if (input[pos] == '+') break;
        pos = pos + 1;
    }
    i = pos - 1;
    j = len - 1;
    r = 0;
    carry = 0;
    while (i >= 0 || j > pos) {
        da = 0;
        db = 0;
        if (i >= 0) {
            da = input[i] - '0';
            i = i - 1;
        }
        if (j > pos) {
            db = input[j] - '0';
            j = j - 1;
        }
        s = da + db + carry;
        d = s;
        if (d >= 10) {
            d = d - 10;
            carry = 1;
        } else {
            carry = 0;
        }
        res[r] = d;
        r = r + 1;
    }
    if (carry == 1) {
        res[r] = 1;
        r = r + 1;
    }
    while (r > 1) {
        if (res[r - 1] != 0) break;
        r = r - 1;
    }
    k = r - 1;
    while (k >= 0) {
        putchar(res[k] + '0');
        k = k - 1;
    }
    putchar('\n');
}
