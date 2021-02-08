#include "fast_double_parser.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

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

static inline uint64_t rng(uint64_t h) {
  h ^= h >> 33;
  h *= UINT64_C(0xff51afd7ed558ccd);
  h ^= h >> 33;
  h *= UINT64_C(0xc4ceb9fe1a85ec53);
  h ^= h >> 33;
  return h;
}

std::string randomfloats(uint64_t howmany) {
  std::stringstream out;
  uint64_t offset = 1190;
  for (size_t i = 1; i <= howmany; i++) {
    uint64_t x = rng(i + offset);
    double d;
    ::memcpy(&d, &x, sizeof(double));
    // paranoid
    while (!std::isfinite(d)) {
      offset++;
      x = rng(i + offset);
      ::memcpy(&d, &x, sizeof(double));
    }
    out << std::setprecision(DBL_DIG) << d << " ";
  }
  return out.str();
}

void check(double d) {
  std::string s(64, '\0');
  auto written = std::snprintf(&s[0], s.size(), "%.*e", DBL_DIG + 1, d);
  s.resize(written);
  double x;
  const char * isok = fast_double_parser::parse_number(s.data(), &x);
  if (!isok) {
    printf("fast_double_parser refused to parse %s\n", s.c_str());
    throw std::runtime_error("fast_double_parser refused to parse");
  }
  if(isok != s.data() + s.size()) throw std::runtime_error("does not point at the end");
  if (d != x) {
    std::cerr << "fast_double_parser disagrees" << std::endl;
    printf("fast_double_parser: %.*e\n", DBL_DIG + 1, x);
    printf("reference: %.*e\n", DBL_DIG + 1, d);
    printf("string: %s\n", s.c_str());
    printf("f64_ulp_dist = %d\n", (int)f64_ulp_dist(x, d));
    throw std::runtime_error("fast_double_parser disagrees");
  }
}


void check_string(std::string s) {
  double x;
  bool isok = fast_double_parser::parse_number(s.data(), &x);
  if (!isok) {
    printf("fast_double_parser refused to parse %s\n", s.c_str());
    throw std::runtime_error("fast_double_parser refused to parse");
  }
#ifdef _WIN32
  static _locale_t c_locale = _create_locale(LC_ALL, "C");
  double d = _strtod_l(s.data(), nullptr, c_locale);
#else
  static locale_t c_locale = newlocale(LC_ALL_MASK, "C", NULL);
  double d = strtod_l(s.data(), nullptr, c_locale);
#endif
  if (d != x) {
    std::cerr << "fast_double_parser disagrees" << std::endl;
    printf("fast_double_parser: %.*e\n", DBL_DIG + 1, x);
    printf("reference: %.*e\n", DBL_DIG + 1, d);
    printf("string: %s\n", s.c_str());
    printf("f64_ulp_dist = %d\n", (int)f64_ulp_dist(x, d));
    throw std::runtime_error("fast_double_parser disagrees");
  }
}


void unit_tests() {
  for (std::string s : {"7.3177701707893310e+15","1e23", "9007199254740995","7e23"}) {
    check_string(s);
  }
  for (double d : {-65.613616999999977, 7.2057594037927933e+16, 1.0e-308,
                   0.1e-308, 0.01e-307, 1.79769e+308, 2.22507e-308,
                   -1.79769e+308, -2.22507e-308, 1e-308}) {
    check(d);
  }
  uint64_t offset = 1190;
  size_t howmany = 10000000;
  for (size_t i = 1; i <= howmany; i++) {
    if ((i % 10000) == 0) {
      printf(".");
      fflush(NULL);
    }
    uint64_t x = rng(i + offset);
    double d;
    ::memcpy(&d, &x, sizeof(double));
    // paranoid
    while ((!std::isnormal(d)) || std::isnan(d) || std::isinf(d)) {
      offset++;
      x = rng(i + offset);
      ::memcpy(&d, &x, sizeof(double));
    }
    check(d);
  }

  printf("Unit tests ok\n");
}
static const double testing_power_of_ten[] = {
    1e-307, 1e-306, 1e-305, 1e-304, 1e-303, 1e-302, 1e-301, 1e-300, 1e-299,
    1e-298, 1e-297, 1e-296, 1e-295, 1e-294, 1e-293, 1e-292, 1e-291, 1e-290,
    1e-289, 1e-288, 1e-287, 1e-286, 1e-285, 1e-284, 1e-283, 1e-282, 1e-281,
    1e-280, 1e-279, 1e-278, 1e-277, 1e-276, 1e-275, 1e-274, 1e-273, 1e-272,
    1e-271, 1e-270, 1e-269, 1e-268, 1e-267, 1e-266, 1e-265, 1e-264, 1e-263,
    1e-262, 1e-261, 1e-260, 1e-259, 1e-258, 1e-257, 1e-256, 1e-255, 1e-254,
    1e-253, 1e-252, 1e-251, 1e-250, 1e-249, 1e-248, 1e-247, 1e-246, 1e-245,
    1e-244, 1e-243, 1e-242, 1e-241, 1e-240, 1e-239, 1e-238, 1e-237, 1e-236,
    1e-235, 1e-234, 1e-233, 1e-232, 1e-231, 1e-230, 1e-229, 1e-228, 1e-227,
    1e-226, 1e-225, 1e-224, 1e-223, 1e-222, 1e-221, 1e-220, 1e-219, 1e-218,
    1e-217, 1e-216, 1e-215, 1e-214, 1e-213, 1e-212, 1e-211, 1e-210, 1e-209,
    1e-208, 1e-207, 1e-206, 1e-205, 1e-204, 1e-203, 1e-202, 1e-201, 1e-200,
    1e-199, 1e-198, 1e-197, 1e-196, 1e-195, 1e-194, 1e-193, 1e-192, 1e-191,
    1e-190, 1e-189, 1e-188, 1e-187, 1e-186, 1e-185, 1e-184, 1e-183, 1e-182,
    1e-181, 1e-180, 1e-179, 1e-178, 1e-177, 1e-176, 1e-175, 1e-174, 1e-173,
    1e-172, 1e-171, 1e-170, 1e-169, 1e-168, 1e-167, 1e-166, 1e-165, 1e-164,
    1e-163, 1e-162, 1e-161, 1e-160, 1e-159, 1e-158, 1e-157, 1e-156, 1e-155,
    1e-154, 1e-153, 1e-152, 1e-151, 1e-150, 1e-149, 1e-148, 1e-147, 1e-146,
    1e-145, 1e-144, 1e-143, 1e-142, 1e-141, 1e-140, 1e-139, 1e-138, 1e-137,
    1e-136, 1e-135, 1e-134, 1e-133, 1e-132, 1e-131, 1e-130, 1e-129, 1e-128,
    1e-127, 1e-126, 1e-125, 1e-124, 1e-123, 1e-122, 1e-121, 1e-120, 1e-119,
    1e-118, 1e-117, 1e-116, 1e-115, 1e-114, 1e-113, 1e-112, 1e-111, 1e-110,
    1e-109, 1e-108, 1e-107, 1e-106, 1e-105, 1e-104, 1e-103, 1e-102, 1e-101,
    1e-100, 1e-99,  1e-98,  1e-97,  1e-96,  1e-95,  1e-94,  1e-93,  1e-92,
    1e-91,  1e-90,  1e-89,  1e-88,  1e-87,  1e-86,  1e-85,  1e-84,  1e-83,
    1e-82,  1e-81,  1e-80,  1e-79,  1e-78,  1e-77,  1e-76,  1e-75,  1e-74,
    1e-73,  1e-72,  1e-71,  1e-70,  1e-69,  1e-68,  1e-67,  1e-66,  1e-65,
    1e-64,  1e-63,  1e-62,  1e-61,  1e-60,  1e-59,  1e-58,  1e-57,  1e-56,
    1e-55,  1e-54,  1e-53,  1e-52,  1e-51,  1e-50,  1e-49,  1e-48,  1e-47,
    1e-46,  1e-45,  1e-44,  1e-43,  1e-42,  1e-41,  1e-40,  1e-39,  1e-38,
    1e-37,  1e-36,  1e-35,  1e-34,  1e-33,  1e-32,  1e-31,  1e-30,  1e-29,
    1e-28,  1e-27,  1e-26,  1e-25,  1e-24,  1e-23,  1e-22,  1e-21,  1e-20,
    1e-19,  1e-18,  1e-17,  1e-16,  1e-15,  1e-14,  1e-13,  1e-12,  1e-11,
    1e-10,  1e-9,   1e-8,   1e-7,   1e-6,   1e-5,   1e-4,   1e-3,   1e-2,
    1e-1,   1e0,    1e1,    1e2,    1e3,    1e4,    1e5,    1e6,    1e7,
    1e8,    1e9,    1e10,   1e11,   1e12,   1e13,   1e14,   1e15,   1e16,
    1e17,   1e18,   1e19,   1e20,   1e21,   1e22,   1e23,   1e24,   1e25,
    1e26,   1e27,   1e28,   1e29,   1e30,   1e31,   1e32,   1e33,   1e34,
    1e35,   1e36,   1e37,   1e38,   1e39,   1e40,   1e41,   1e42,   1e43,
    1e44,   1e45,   1e46,   1e47,   1e48,   1e49,   1e50,   1e51,   1e52,
    1e53,   1e54,   1e55,   1e56,   1e57,   1e58,   1e59,   1e60,   1e61,
    1e62,   1e63,   1e64,   1e65,   1e66,   1e67,   1e68,   1e69,   1e70,
    1e71,   1e72,   1e73,   1e74,   1e75,   1e76,   1e77,   1e78,   1e79,
    1e80,   1e81,   1e82,   1e83,   1e84,   1e85,   1e86,   1e87,   1e88,
    1e89,   1e90,   1e91,   1e92,   1e93,   1e94,   1e95,   1e96,   1e97,
    1e98,   1e99,   1e100,  1e101,  1e102,  1e103,  1e104,  1e105,  1e106,
    1e107,  1e108,  1e109,  1e110,  1e111,  1e112,  1e113,  1e114,  1e115,
    1e116,  1e117,  1e118,  1e119,  1e120,  1e121,  1e122,  1e123,  1e124,
    1e125,  1e126,  1e127,  1e128,  1e129,  1e130,  1e131,  1e132,  1e133,
    1e134,  1e135,  1e136,  1e137,  1e138,  1e139,  1e140,  1e141,  1e142,
    1e143,  1e144,  1e145,  1e146,  1e147,  1e148,  1e149,  1e150,  1e151,
    1e152,  1e153,  1e154,  1e155,  1e156,  1e157,  1e158,  1e159,  1e160,
    1e161,  1e162,  1e163,  1e164,  1e165,  1e166,  1e167,  1e168,  1e169,
    1e170,  1e171,  1e172,  1e173,  1e174,  1e175,  1e176,  1e177,  1e178,
    1e179,  1e180,  1e181,  1e182,  1e183,  1e184,  1e185,  1e186,  1e187,
    1e188,  1e189,  1e190,  1e191,  1e192,  1e193,  1e194,  1e195,  1e196,
    1e197,  1e198,  1e199,  1e200,  1e201,  1e202,  1e203,  1e204,  1e205,
    1e206,  1e207,  1e208,  1e209,  1e210,  1e211,  1e212,  1e213,  1e214,
    1e215,  1e216,  1e217,  1e218,  1e219,  1e220,  1e221,  1e222,  1e223,
    1e224,  1e225,  1e226,  1e227,  1e228,  1e229,  1e230,  1e231,  1e232,
    1e233,  1e234,  1e235,  1e236,  1e237,  1e238,  1e239,  1e240,  1e241,
    1e242,  1e243,  1e244,  1e245,  1e246,  1e247,  1e248,  1e249,  1e250,
    1e251,  1e252,  1e253,  1e254,  1e255,  1e256,  1e257,  1e258,  1e259,
    1e260,  1e261,  1e262,  1e263,  1e264,  1e265,  1e266,  1e267,  1e268,
    1e269,  1e270,  1e271,  1e272,  1e273,  1e274,  1e275,  1e276,  1e277,
    1e278,  1e279,  1e280,  1e281,  1e282,  1e283,  1e284,  1e285,  1e286,
    1e287,  1e288,  1e289,  1e290,  1e291,  1e292,  1e293,  1e294,  1e295,
    1e296,  1e297,  1e298,  1e299,  1e300,  1e301,  1e302,  1e303,  1e304,
    1e305,  1e306,  1e307,  1e308};


void issue13() {
  std::string a = "0";
  double x;
  bool ok = fast_double_parser::parse_number(a.c_str(), &x);
  if(!ok) throw std::runtime_error("could not parse zero.");
  if(x != 0) throw std::runtime_error("zero does not map to zero.");
  std::cout << "zero maps to zero" << std::endl;
}

void issue40() {
  //https://tools.ietf.org/html/rfc7159
  // A fraction part is a decimal point followed by one or more digits.
  std::string a = "0.";
  double x;
  bool ok = fast_double_parser::parse_number(a.c_str(), &x);
  if(ok) throw std::runtime_error("We should not parse '0.'");
}

void issue32() {
  std::string a = "-0";
  double x;
  bool ok = fast_double_parser::parse_number(a.c_str(), &x);
  if(!ok) throw std::runtime_error("could not parse -zero.");
  if(x != 0) throw std::runtime_error("-zero does not map to zero.");
  std::cout << "zero maps to zero" << std::endl;
}

void issue23() {
  std::string a = "0e+42949672970";
  double x;
  bool ok = fast_double_parser::parse_number(a.c_str(), &x);
  if(!ok) throw std::runtime_error("could not parse zero.");
  if(x != 0) throw std::runtime_error("zero does not map to zero.");
  std::cout << "zero maps to zero" << std::endl;
}

void issue23_2() {
  std::string a = "5e0012";
  double x;
  const char * ok = fast_double_parser::parse_number(a.c_str(), &x);
  if(!ok) throw std::runtime_error("could not parse zero.");
  if(ok != a.c_str() + a.size()) throw std::runtime_error("does not point at the end.");
  if(x != 5e12) throw std::runtime_error("cannot parse 5e0012.");
  std::cout << "can parse 5e0012" << std::endl;
}

inline void Assert(bool Assertion) {
  if (!Assertion)
    throw std::runtime_error("bug");
}
bool basic_test_64bit(std::string vals, double val) {
  std::cout << " parsing "  << vals << std::endl;
  double result_value;
  const char * ok = fast_double_parser::parse_number(vals.c_str(), & result_value);
  if (!ok) {
    std::cerr << " I could not parse " << vals << std::endl;
    return false;
  }
  if(ok != vals.c_str() + vals.size()) throw std::runtime_error("does not point at the end.");
  if (std::isnan(val)) {
    if (!std::isnan(result_value)) {
      std::cerr << "not nan" << result_value << std::endl;
      return false;
    }
  } else if (result_value != val) {
    std::cerr << "I got " << std::hexfloat << result_value << " but I was expecting " << val
              << std::endl;
    std::cerr << std::dec;
    std::cerr << "string: " << vals << std::endl;
    return false;
  }
  std::cout << std::hexfloat << result_value << " == " << val << std::endl;
  std::cout << std::dec;

  return true;
}
int main() {
  const int evl_method = FLT_EVAL_METHOD;
  printf("FLT_EVAL_METHOD = %d\n", evl_method);
  bool is_pow_correct{1e-308 == std::pow(10,-308)};
  if(!is_pow_correct) {
    printf("It appears that your system has bad floating-point support since 1e-308 differs from std::pow(10,-308).\n");
    printf("Aborting further tests.");
    return EXIT_SUCCESS;
  } 
  Assert(basic_test_64bit("1090544144181609348835077142190",0x1.b8779f2474dfbp+99));
  Assert(basic_test_64bit("4503599627370496.5", 4503599627370496.5));
  Assert(basic_test_64bit("4503599627370497.5", 4503599627370497.5));
  Assert(basic_test_64bit("0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000044501477170144022721148195934182639518696390927032912960468522194496444440421538910330590478162701758282983178260792422137401728773891892910553144148156412434867599762821265346585071045737627442980259622449029037796981144446145705102663115100318287949527959668236039986479250965780342141637013812613333119898765515451440315261253813266652951306000184917766328660755595837392240989947807556594098101021612198814605258742579179000071675999344145086087205681577915435923018910334964869420614052182892431445797605163650903606514140377217442262561590244668525767372446430075513332450079650686719491377688478005309963967709758965844137894433796621993967316936280457084866613206797017728916080020698679408551343728867675409720757232455434770912461317493580281734466552734375", 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000044501477170144022721148195934182639518696390927032912960468522194496444440421538910330590478162701758282983178260792422137401728773891892910553144148156412434867599762821265346585071045737627442980259622449029037796981144446145705102663115100318287949527959668236039986479250965780342141637013812613333119898765515451440315261253813266652951306000184917766328660755595837392240989947807556594098101021612198814605258742579179000071675999344145086087205681577915435923018910334964869420614052182892431445797605163650903606514140377217442262561590244668525767372446430075513332450079650686719491377688478005309963967709758965844137894433796621993967316936280457084866613206797017728916080020698679408551343728867675409720757232455434770912461317493580281734466552734375));
  Assert(basic_test_64bit("0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022250738585072008890245868760858598876504231122409594654935248025624400092282356951787758888037591552642309780950434312085877387158357291821993020294379224223559819827501242041788969571311791082261043971979604000454897391938079198936081525613113376149842043271751033627391549782731594143828136275113838604094249464942286316695429105080201815926642134996606517803095075913058719846423906068637102005108723282784678843631944515866135041223479014792369585208321597621066375401613736583044193603714778355306682834535634005074073040135602968046375918583163124224521599262546494300836851861719422417646455137135420132217031370496583210154654068035397417906022589503023501937519773030945763173210852507299305089761582519159720757232455434770912461317493580281734466552734375", 0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000022250738585072008890245868760858598876504231122409594654935248025624400092282356951787758888037591552642309780950434312085877387158357291821993020294379224223559819827501242041788969571311791082261043971979604000454897391938079198936081525613113376149842043271751033627391549782731594143828136275113838604094249464942286316695429105080201815926642134996606517803095075913058719846423906068637102005108723282784678843631944515866135041223479014792369585208321597621066375401613736583044193603714778355306682834535634005074073040135602968046375918583163124224521599262546494300836851861719422417646455137135420132217031370496583210154654068035397417906022589503023501937519773030945763173210852507299305089761582519159720757232455434770912461317493580281734466552734375));
  issue40();
  issue23();
  issue32();
  issue23_2();
  unit_tests();
  for (int p = -306; p <= 308; p++) {
    if (p == 23)
      p++;
    printf(".");
    fflush(NULL);
    bool success;
    double d = fast_double_parser::compute_float_64(p, 1, false, &success);
    if (!success) {
      printf("failed to parse\n");
      printf(" 10 ^ %d ", p);

      return EXIT_FAILURE;
    }
    if (d != testing_power_of_ten[p + 307]) {
      printf(" 10 ^ %d ", p);

      printf("bad parsing\n");
      printf("got: %.*e\n", DBL_DIG + 1, d);
      printf("reference: %.*e\n", DBL_DIG + 1, testing_power_of_ten[p + 307]);

      return EXIT_FAILURE;
    }
  }
  for (int p = -306; p <= 308; p++) {
    double d;
    std::string s = "1e"+std::to_string(p);
    std::cout << "parsing " << s << std::endl;
    const char * isok = fast_double_parser::parse_number(s.c_str(), &d);
    if (!isok) {
      printf("fast_double_parser refused to parse %s\n", s.c_str());
      throw std::runtime_error("fast_double_parser refused to parse");
    }
    if(isok != s.c_str() + s.size()) throw std::runtime_error("does not point at the end");
    if (d != testing_power_of_ten[p + 307]) {
      std::cerr << "fast_double_parser disagrees" << std::endl;
      printf("fast_double_parser: %.*e\n", DBL_DIG + 1, d);
      printf("reference: %.*e\n", DBL_DIG + 1, testing_power_of_ten[p + 307]);
      printf("string: %s\n", s.c_str());
      throw std::runtime_error("fast_double_parser disagrees");
    }
  }
  std::cout << std::endl;
  std::cout << "All ok" << std::endl;
  printf("Good!\n");
  return EXIT_SUCCESS;
}
