/*!
* This file is part of GPBoost a C++ library for combining
*	boosting with Gaussian process and mixed effects models
*
* Copyright (c) 2020 Fabio Sigrist. All rights reserved.
*
* Licensed under the Apache License Version 2.0. See LICENSE file in the project root for license information.
*/
#include <GPBoost/utils.h>

namespace GPBoost {

	void SortVectorsDecreasing(double* a, int* b, int n) {
		int j, k, l;
		double v;
		for (j = 1; j <= n - 1; j++) {
			k = j;
			while (k > 0 && a[k] < a[k - 1]) {
				v = a[k]; l = b[k];
				a[k] = a[k - 1]; b[k] = b[k - 1];
				a[k - 1] = v; b[k - 1] = l;
				k--;
			}
		}
	}

}  // namespace GPBoost
