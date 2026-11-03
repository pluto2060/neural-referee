void compute(const char *input) {
    int n = str_len(input);
    int i = 0;
    int j = n - 1;
    while (i < j) {
        char c1 = input[i];
        char c2 = input[j];
        if (c1 != c2) {
            print_str("NO\n");
            return;
        }
        i = i + 1;
        j = j - 1;
    }
    print_str("YES\n");
}
