void compute(const char *input) {
    int n = str_len(input);
    int i = 0;
    int j = n - 1;
    int ok = 1;
    while (i < j) {
        if (input[i] != input[j]) {
            ok = 0;
            break;
        }
        i = i + 1;
        j = j - 1;
    }
    if (ok == 1) {
        print_str("YES\n");
    } else {
        print_str("NO\n");
    }
}
