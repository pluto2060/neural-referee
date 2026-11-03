void compute(const char *input) {
    int i = 0;
    while (input[i]) {
        char c = input[i];
        if (c >= 'a' && c <= 'z') c = c - 32;
        putchar(c);
        i = i + 1;
    }
    putchar('\n');
}
