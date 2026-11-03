void compute(const char *input) {
    int sum = 0;
    int i = 0;
    while (input[i] != '\0') {
        int digit = input[i] - '0';
        while (digit > 0) {
            sum += digit % 2;
            digit /= 2;
        }
        i++;
    }
    print_int(sum);
    print_str("\n");
}
