void compute(const char *input) {
    int sum = 0;
    int i = 0;
    while (input[i]) {
        char c = input[i];
        sum = sum + c - '0';
        i = i + 1;
    }
    print_int(sum);
    putchar('\n');
}
