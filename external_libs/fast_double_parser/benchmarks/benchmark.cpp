
#include "absl/strings/charconv.h"
#include "absl/strings/numbers.h"
#include "fast_double_parser.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdio.h>
#include <vector>

#include "double-conversion/ieee.h"
#include "double-conversion/string-to-double.h"

double findmax_fast_double_parser(const std::vector<std::string>& s) {
  double answer = 0;
  double x;
  for (const std::string & st : s) {
    bool isok = fast_double_parser::parse_number(st.c_str(), &x);
    if (!isok)
      throw std::runtime_error("bug in findmax_fast_double_parser");
    answer = answer > x ? answer : x;
  }
  return answer;
}


double findmax_strtod(const std::vector<std::string>& s) {
  double answer = 0;
  double x = 0;
  for (const std::string& st : s) {
    char *pr = (char *)st.data();
#ifdef _WIN32
    static _locale_t c_locale = _create_locale(LC_ALL, "C");
    x = _strtod_l(st.data(), &pr, c_locale);
#else
    static locale_t c_locale = newlocale(LC_ALL_MASK, "C", NULL);
    x = strtod_l(st.data(), &pr, c_locale);
#endif
    if ((pr == nullptr) || (pr == st.data())) {
      throw std::runtime_error("bug in findmax_strtod");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}

double findmax_absl(const std::vector<std::string>& s) {
  double answer = 0;
  double x = 0;
  for (const std::string& st : s) {
    bool isok = absl::SimpleAtod(st, &x);
    if (!isok) {
      throw std::runtime_error("bug in findmax_absl");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}

double findmax_absl_from_chars(const std::vector<std::string>& s) {
  double answer = 0;
  double x = 0;
  for (const std::string& st : s) {
    auto res = absl::from_chars(st.data(), st.data() + st.size(), x);
    if (res.ptr == st.data()) {
      throw std::runtime_error("bug in findmax_absl_from_chars");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}

double findmax_doubleconversion(const std::vector<std::string>& s) {
  double answer = 0;
  double x;
  int flags = double_conversion::StringToDoubleConverter::ALLOW_LEADING_SPACES |
              double_conversion::StringToDoubleConverter::ALLOW_TRAILING_JUNK |
              double_conversion::StringToDoubleConverter::ALLOW_TRAILING_SPACES;
  double empty_string_value = 0.0;
  uc16 separator = double_conversion::StringToDoubleConverter::kNoSeparator;
  double_conversion::StringToDoubleConverter converter(
      flags, empty_string_value, double_conversion::Double::NaN(), NULL, NULL,
      separator);
  int processed_characters_count;
  for (const std::string& st : s) {
    x = converter.StringToDouble(st.data(), int(st.size()),
                                 &processed_characters_count);
    if (processed_characters_count == 0) {
      throw std::runtime_error("bug in findmax_doubleconversion");
    }
    answer = answer > x ? answer : x;
  }
  return answer;
}

// ulp distance
// Marc B. Reynolds, 2016-2019
// Public Domain under http://unlicense.org, see link for details.
// adapted by D. Lemire
inline uint64_t f64_ulp_dist(double a, double b) {
  uint64_t ua, ub;
  memcpy(&ua, &a, sizeof(ua));
  memcpy(&ub, &b, sizeof(ub));
  if ((int64_t)(ub ^ ua) >= 0)
    return (int64_t)(ua - ub) >= 0 ? (ua - ub) : (ub - ua);
  return ua + ub + 0x80000000;
}

void validate(const std::vector<std::string>& s) {

  double x, xref;
  int flags = double_conversion::StringToDoubleConverter::ALLOW_LEADING_SPACES |
              double_conversion::StringToDoubleConverter::ALLOW_TRAILING_JUNK |
              double_conversion::StringToDoubleConverter::ALLOW_TRAILING_SPACES;
  double empty_string_value = 0.0;
  uc16 separator = double_conversion::StringToDoubleConverter::kNoSeparator;
  double_conversion::StringToDoubleConverter converter(
      flags, empty_string_value, double_conversion::Double::NaN(), NULL, NULL,
      separator);
  int processed_characters_count;
  for (const std::string& st : s) {
#ifdef _WIN32
    static _locale_t c_locale = _create_locale(LC_ALL, "C");
    xref = _strtod_l(st.data(), nullptr, c_locale);
#else
    static locale_t c_locale = newlocale(LC_ALL_MASK, "C", NULL);
    xref = strtod_l(st.data(), nullptr, c_locale);
#endif
    x = converter.StringToDouble(st.data(), int(st.size()),
                                 &processed_characters_count);
    if (xref != x) {
      std::cerr << "double conversion disagrees" << std::endl;
      printf("double conversion: %.*e\n", DBL_DIG + 1, x);
      printf("reference: %.*e\n", DBL_DIG + 1, xref);
      printf("string: %s\n", st.c_str());
      printf("f64_ulp_dist = %d\n", (int)f64_ulp_dist(x, xref));
      throw std::runtime_error("double conversion disagrees");
    }
    absl::from_chars(st.data(), st.data() + st.size(), x);
    if (xref != x) {
      std::cerr << "abseil from_chars disagrees" << std::endl;
      printf("abseil from_chars: %.*e\n", DBL_DIG + 1, x);
      printf("reference: %.*e\n", DBL_DIG + 1, xref);
      printf("string: %s\n", st.c_str());
      printf("f64_ulp_dist = %d\n", (int)f64_ulp_dist(x, xref));
      throw std::runtime_error("abseil from_chars disagrees");
    }
    bool isok = absl::SimpleAtod(st, &x);
    if (!isok) {
      throw std::runtime_error("bug in absl::SimpleAtod");
    }
    if (xref != x) {
      std::cerr << "abseil disagrees" << std::endl;
      printf("abseil: %.*e\n", DBL_DIG + 1, x);
      printf("reference: %.*e\n", DBL_DIG + 1, xref);
      printf("string: %s\n", st.c_str());
      printf("f64_ulp_dist = %d\n", (int)f64_ulp_dist(x, xref));
      throw std::runtime_error("abseil disagrees");
    }
    isok = fast_double_parser::parse_number(st.c_str(), &x);
    if (!isok) {
      printf("fast_double_parser refused to parse %s\n", st.c_str());
      throw std::runtime_error("fast_double_parser refused to parse");
    }
    if (xref != x) {
      std::cerr << "fast_double_parser disagrees" << std::endl;
      printf("fast_double_parser: %.*e\n", DBL_DIG + 1, x);
      printf("reference: %.*e\n", DBL_DIG + 1, xref);
      printf("string: %s\n", st.c_str());
      printf("f64_ulp_dist = %d\n", (int)f64_ulp_dist(x, xref));
      throw std::runtime_error("fast_double_parser disagrees");
    }
  }
}

void printvec(const std::vector<unsigned long long>& evts, size_t volume) {
  printf("%.2f cycles  %.2f instr  %.4f branch miss  %.2f cache ref %.2f cache "
         "miss \n",
         evts[0] * 1.0 / volume, evts[1] * 1.0 / volume, evts[2] * 1.0 / volume,
         evts[3] * 1.0 / volume, evts[4] * 1.0 / volume);
}

void process(const std::vector<std::string>& lines, size_t volume) {
  double volumeMB = volume / (1024. * 1024.);
  // size_t howmany = lines.size();
  std::chrono::high_resolution_clock::time_point t1, t2;
  double dif, ts;
  for (size_t i = 0; i < 3; i++) {
    if (i > 0)
      printf("=== trial %zu ===\n", i);

    t1 = std::chrono::high_resolution_clock::now();
    ts = findmax_fast_double_parser(lines);
    t2 = std::chrono::high_resolution_clock::now();
    if (ts == 0)
      printf("bug\n");
    dif = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    if (i > 0)
      printf("fast_double_parser  %.2f MB/s\n", volumeMB * 1000000000 / dif);
    t1 = std::chrono::high_resolution_clock::now();
    ts = findmax_strtod(lines);
    t2 = std::chrono::high_resolution_clock::now();
    if (ts == 0)
      printf("bug\n");
    dif = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    if (i > 0)
      printf("strtod         %.2f MB/s\n", volumeMB * 1000000000 / dif);
    t1 = std::chrono::high_resolution_clock::now();
    ts = findmax_absl_from_chars(lines);
    t2 = std::chrono::high_resolution_clock::now();
    if (ts == 0)
      printf("bug\n");
    dif = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    if (i > 0)
      printf("abslfromch     %.2f MB/s\n", volumeMB * 1000000000 / dif);
    t1 = std::chrono::high_resolution_clock::now();
    ts = findmax_absl(lines);
    t2 = std::chrono::high_resolution_clock::now();
    if (ts == 0)
      printf("bug\n");
    dif = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    if (i > 0)
      printf("absl           %.2f MB/s\n", volumeMB * 1000000000 / dif);
    t1 = std::chrono::high_resolution_clock::now();
    ts = findmax_doubleconversion(lines);
    t2 = std::chrono::high_resolution_clock::now();
    if (ts == 0)
      printf("bug\n");
    dif = double(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    if (i > 0)
      printf("double-conv    %.2f MB/s\n", volumeMB * 1000000000 / dif);
    printf("\n\n");
  }
}

void fileload(char *filename) {

  std::ifstream inputfile(filename);
  if (!inputfile) {
    std::cerr << "can't open " << filename << std::endl;
    return;
  }
  std::string line;
  std::vector<std::string> lines;
  lines.reserve(10000); // let us reserve plenty of memory.
  size_t volume = 0;
  while (getline(inputfile, line)) {
    volume += line.size();
    lines.push_back(line);
  }
  std::cout << "read " << lines.size() << " lines " << std::endl;
  validate(lines);
  process(lines, volume);
}

void demo(size_t howmany) {
  std::cout << "parsing random integers in the range [0,1)" << std::endl;
  std::vector<std::string> lines;
  lines.reserve(howmany); // let us reserve plenty of memory.
  size_t volume = 0;
  for (size_t i = 0; i < howmany; i++) {
    double x = (double)rand() / RAND_MAX;
    std::string line = std::to_string(x);
    volume += line.size();
    lines.push_back(line);
  }
  validate(lines);
  process(lines, volume);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    demo(100 * 1000);
    std::cout << "You can also provide a filename: it should contain one "
                 "string per line corresponding to a number"
              << std::endl;
  } else {
    fileload(argv[1]);
  }
}
