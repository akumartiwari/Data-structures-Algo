package com.company;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class DPWithBitmask {
    String s;
    //ind -> current digit
    //mask -> (1 -> digit is used), (0 -> digit is not used)
    //greater -> represents whether the current formed number is smaller or not
    long[][][] dp = new long[10][1 << 10][2];//

    /*
     * PROBLEM: Count Special Numbers (LeetCode 2376)
     * Count integers in [1, n] where all digits are distinct (no repeated digit).
     *
     * ALGORITHM: Digit DP with bitmask – counts numbers with distinct digits shorter than n's length,
     *            then adds digit-DP count for numbers of the same length as n.
     * TC: O(D * 2^10 * 10) where D = number of digits in n | SC: O(D * 2^10 * 2)
     *
     * Example: n = 20 → Output: 19 (all integers 1–20 except 11 have distinct digits)
     */
    //Author: Anand
    public int countSpecialNumbers(int n) {
        int d = 0;
        int number = n;

        while (number > 0) {
            d++;
            number /= 10;
        }

        s = String.valueOf(n);

        for (long[][] td : dp) {
            for (long[] p : td) {
                Arrays.fill(p, -1);
            }
        }

        // count of numbers having distinct digits and digits less than the number of digits in n
        // numbers of size 1 = 9 ways
        // numbers of size 2 = 9 * 9 ways
        // numbers of size 3 = 9 * 9 * 8 ways
        // numbers of size 4 = 9 * 9 * 8 * 7 ways

        long ans = 0L;
        for (int i = 1; i < d; ++i) {
            long curr = 1;
            for (int j = 1, l = 9; j <= i; ++j) {
                if (j <= 2) curr *= l;
                else curr *= (--l);
            }
            ans += curr;
        }
        ans += dfs(0, 0, 1);
        return (int) ans;
    }

    /*
     * PROBLEM: Count Special Numbers – bitmask DFS traversal (Helper)
     * Recursively fills digits of the number being formed, tracking which digits are already used via a bitmask.
     *
     * ALGORITHM: Digit DP with bitmask memoization
     *   ind     – current digit position (0-indexed from most significant)
     *   mask    – bitmask where bit d=1 means digit d is already used
     *   greater – 1 if the number formed so far is already strictly less than n (free to use any digit);
     *             0 means we are still tight with n's prefix
     * TC: O(D * 2^10 * 2 * 10) | SC: O(D * 2^10 * 2)
     */
    private long dfs(int ind, int mask, int greater) {
        // base case
        if (ind == s.length()) return 1;

        if (dp[ind][mask][greater] != -1) return dp[ind][mask][greater];

        long ans = 0;
        for (int d = 0; d <= 9; ++d) {
            // if curr digit is taken OR d=0 for size=1 then skip
            if ((ind == 0 && d == 0) || (mask & 1 << d) != 0) continue;
            // if current digit is smaller than original number at index then we can take it
            if (d < s.charAt(ind) - '0')
                ans += dfs(ind + 1, mask | 1 << d, 0);

                // curr digit is same than original number at index then we can take it
            else if (d == s.charAt(ind) - '0')
                ans += dfs(ind + 1, mask | 1 << d, greater);

                // curr digit is greater than original number at index and earlier it was smaller than we can take it
            else if (d > s.charAt(ind) - '0' && greater == 0) ans += dfs(ind + 1, mask | 1 << d, greater);
        }

        return dp[ind][mask][greater] = ans;
    }


    /*
     * PROBLEM: Max Product of Two Elements Whose Bitwise AND is Zero (LeetCode 2044)
     * Find the maximum product nums[i] * nums[j] over all pairs (i < j) such that nums[i] & nums[j] == 0.
     *
     * ALGORITHM: Brute-force enumeration of all pairs with bitwise AND check
     * TC: O(N^2) | SC: O(1)
     */
    //Solve using DP with bitmask
    public long maxProduct(int[] nums) {
        long mp = 0L;
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                if ((nums[i] & nums[j]) == 0) {
                    mp = Math.max(mp, (long) nums[i] * nums[j]);
                }
            }
        }

        return mp;
    }

}
