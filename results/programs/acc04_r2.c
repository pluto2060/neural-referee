void compute(const char *input) {
    int i = 0;
    while (input[i]) {
        char c = input[i];
        if (c >= 'a' && c <= 'z') c = c - 'a' + 'A';
        print_str(&c);
        i = i + 1;
    }
    putchar('\n');
}
