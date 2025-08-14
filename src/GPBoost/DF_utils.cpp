/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 - 2025 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/DF_utils.h>
#include <LightGBM/utils/log.h>
using LightGBM::Log;

namespace GPBoost {

    double normalQF(double p) {
        CHECK(p > 0.0 && p < 1.0);
        double r, val;
        const double q = p - 0.5;
        if (std::abs(q) <= .425) {
            r = .180625 - q * q;
            val =
                q * (((((((r * 2509.0809287301226727 +
                    33430.575583588128105) * r + 67265.770927008700853) * r +
                    45921.953931549871457) * r + 13731.693765509461125) * r +
                    1971.5909503065514427) * r + 133.14166789178437745) * r +
                    3.387132872796366608)
                / (((((((r * 5226.495278852854561 +
                    28729.085735721942674) * r + 39307.89580009271061) * r +
                    21213.794301586595867) * r + 5394.1960214247511077) * r +
                    687.1870074920579083) * r + 42.313330701600911252) * r + 1);
        }
        else {
            if (q > 0) {
                r = 1 - p;
            }
            else {
                r = p;
            }

            r = std::sqrt(-std::log(r));

            if (r <= 5)
            {
                r += -1.6;
                val = (((((((r * 7.7454501427834140764e-4 +
                    .0227238449892691845833) * r + .24178072517745061177) *
                    r + 1.27045825245236838258) * r +
                    3.64784832476320460504) * r + 5.7694972214606914055) *
                    r + 4.6303378461565452959) * r +
                    1.42343711074968357734)
                    / (((((((r *
                        1.05075007164441684324e-9 + 5.475938084995344946e-4) *
                        r + .0151986665636164571966) * r +
                        .14810397642748007459) * r + .68976733498510000455) *
                        r + 1.6763848301838038494) * r +
                        2.05319162663775882187) * r + 1);
            }
            else { /* very close to  0 or 1 */
                r += -5;
                val = (((((((r * 2.01033439929228813265e-7 +
                    2.71155556874348757815e-5) * r +
                    .0012426609473880784386) * r + .026532189526576123093) *
                    r + .29656057182850489123) * r +
                    1.7848265399172913358) * r + 5.4637849111641143699) *
                    r + 6.6579046435011037772)
                    / (((((((r *
                        2.04426310338993978564e-15 + 1.4215117583164458887e-7) *
                        r + 1.8463183175100546818e-5) * r +
                        7.868691311456132591e-4) * r + .0148753612908506148525)
                        * r + .13692988092273580531) * r +
                        .59983220655588793769) * r + 1);
            }

            if (q < 0.0) {
                val = -val;
            }
        }

        return val;
    }

    double digamma(double x) {
        static double c = 8.5;
        static double euler_mascheroni = 0.57721566490153286060;
        double r;
        double value;
        double x2;
        CHECK(x > 0);
        //
        //  Use approximation for small argument.
        //
        if (x <= 0.000001)
        {
            value = -euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x;
            return value;
        }
        //
        //  Reduce to DIGAMA(X + N).
        //
        value = 0.0;
        x2 = x;
        while (x2 < c)
        {
            value = value - 1.0 / x2;
            x2 = x2 + 1.0;
        }
        //
        //  Use Stirling's (actually de Moivre's) expansion.
        //
        r = 1.0 / x2;
        value = value + log(x2) - 0.5 * r;

        r = r * r;

        value = value
            - r * (1.0 / 12.0
                - r * (1.0 / 120.0
                    - r * (1.0 / 252.0
                        - r * (1.0 / 240.0
                            - r * (1.0 / 132.0)))));

        return value;
    }//end digamma

    double trigamma(double x) {
        double a = 0.0001;
        double b = 5.0;
        double b2 = 0.1666666667;
        double b4 = -0.03333333333;
        double b6 = 0.02380952381;
        double b8 = -0.03333333333;
        double value;
        double y;
        double z;
        CHECK(x > 0);
        z = x;
        //
        //  Use small value approximation if X <= A.
        //
        if (x <= a)
        {
            value = 1.0 / x / x;
            return value;
        }
        //
        //  Increase argument to ( X + I ) >= B.
        //
        value = 0.0;

        while (z < b)
        {
            value = value + 1.0 / z / z;
            z = z + 1.0;
        }
        //
        //  Apply asymptotic formula if argument is B or greater.
        //
        y = 1.0 / z / z;

        value = value + 0.5 *
            y + (1.0
                + y * (b2
                    + y * (b4
                        + y * (b6
                            + y * b8)))) / z;

        return value;
    }//end trigamma

    double tetragamma(double x) {
        CHECK(x > 0.0);
        const double tiny = 1e-4;   // tiny-x shortcut
        const double threshold = 8.0;    // switch to asymptotic series
        /* tiny-x limit: phi''(x) approx -2/x^3 */
        if (x <= tiny)
            return -2.0 / (x * x * x);
        /* Shift upward until z >= threshold, using
           phi''(x) = phi''(x+1) - 2/x^3   */
        double z = x;
        double value = 0.0;
        while (z < threshold) {
            value -= 2.0 / (z * z * z);
            z += 1.0;
        }
        /* Bernoulli based asymptotic series */
        double z2 = z * z;
        double z3 = z2 * z;
        double z4 = z2 * z2;
        double z6 = z4 * z2;
        double z8 = z4 * z4;
        double z10 = z8 * z2;

        value += -1.0 / z2
            - 1.0 / z3
            - 0.5 / z4
            + 1.0 / (6.0 * z6)
            - 1.0 / (6.0 * z8)
            + 3.0 / (10.0 * z10);

        return value;
    }

}  // namespace GPBoost
