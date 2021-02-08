#include "fast_double_parser.h"
// To detect issues such as 
// https://github.com/lemire/fast_double_parser/issues/42

fast_double_parser::value128 my_other_full_multiplication(uint64_t value1, uint64_t value2) {
    return fast_double_parser::full_multiplication(value1, value2);
}