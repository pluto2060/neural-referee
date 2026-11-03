void compute(const char *input) {
    int n = 0;
    int i = 0;
    while (input[i]) {
        char c = input[i];
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') n = n + 1;
        i = i + 1;
    }
    print_int(n);
    putchar('\n');
}
