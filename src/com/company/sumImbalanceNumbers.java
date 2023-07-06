package com.company;

import java.util.HashSet;
import java.util.Set;

/*
Input: nums = [2,3,1,4]
Output: 3
Explanation: There are 3 subarrays with non-zero imbalance numbers:
- Subarray [3, 1] with an imbalance number of 1.
- Subarray [3, 1, 4] with an imbalance number of 1.
- Subarray [1, 4] with an imbalance number of 1.
The imbalance number of all other subarrays is 0. Hence, the sum of imbalance numbers of all the subarrays of nums is 3.
 */

/*
Solution =

Each time, check the array start at A[i],
and add A[i + 1], A[i + 2] ... one by one.
At first, the current imbalance number cur = 0,
when add A[j],
check if A[j] + 1 and A[j] - 1 already in the subarray.

If A[j] + 1 not in seen and A[j] - 1 not in seen,
A[j] will make imbalance number increment cur += 1.

If A[j] + 1 in seen and A[j] - 1 in seen,
A[j] will make imbalance number decrement cur -= 1.

Then update result with current imbalance number res += cur.

 */
public class sumImbalanceNumbers {
    public int sumImbalanceNumbers(int[] A) {
        int res = 0, n = A.length;
        for (int i = 0; i < n; ++i) {
            Set<Integer> s = new HashSet<>();
            s.add(A[i]);
            int cur = 0;
            for (int j = i + 1; j < n; ++j) {
                if (!s.contains(A[j])) {
                    int d = 1;
                    if (s.contains(A[j] - 1)) d--;
                    if (s.contains(A[j] + 1)) d--;
                    cur += d;
                    s.add(A[j]);
                }
                res += cur;
            }
        }
        return res;
    }
}
