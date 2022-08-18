package com.company;

import java.util.Arrays;

public class DPWithBitmask {
    String s;
    //ind -> current digit
    //mask -> (1 -> digit is used), (0 -> digit is not used)
    //greater -> represents whether the current formed number is smaller or not
    long[][][] dp = new long[10][1 << 10][2];//

    /*
    Input: n = 20
    Output: 19
    Explanation: All the integers from 1 to 20, except 11, are special. Thus, there are 19 special integers.
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
}
