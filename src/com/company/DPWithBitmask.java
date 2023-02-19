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


    /*
    Input: nums = [3,4,4,5]
    Output: 3
    Explanation: There are 3 square-free subsets in this example:
            - The subset consisting of the 0th element [3]. The product of its elements is 3, which is a square-free integer.
            - The subset consisting of the 3rd element [5]. The product of its elements is 5, which is a square-free integer.
            - The subset consisting of 0th and 3rd elements [3,5]. The product of its elements is 15, which is a square-free integer.
    It can be proven that there are no more than 3 square-free subsets in the given array.
    */
    //DO with DP 6 bitmask
    int MOD = 1_00000_0000 + 7;

    public int squareFreeSubsets(int[] nums) {
        Map<String, Integer> dp = new HashMap<>();
        return helper(nums, 0, 1L, dp);
    }


    private int helper(int[] nums, int ind, long prod, Map<String, Integer> dp) {
        if (ind >= nums.length) return 0;
        int t = 0, nt = 0;

        String key = prod + "-" + ind;
        if (dp.containsKey(key)) return dp.get(key);
        // take
        if (isSquareFree(prod * nums[ind])) {
            t += (1 + helper(nums, ind + 1, prod * nums[ind], dp)) % MOD;
        }
        nt += helper(nums, ind + 1, prod, dp);
        int ans = (t + nt) % MOD;
        dp.put(key, ans);
        return ans;
    }

    //function that checks if the given number is square free or not
    private boolean isSquareFree(long num) {
        //finds the remainder
        if (num % 2 == 0)
            //divides the given number by 2
            num = num / 2;
        //if the number again completely divisible by 2, the number is not a square free number
        if (num % 2 == 0)
            return false;
        //num  must be odd at the moment, so we have increment i by 2
        for (int i = 3; i <= Math.sqrt(num); i = i + 2) {
            //checks i is a prime factor or not
            if (num % i == 0) {
                num = num / i;
                //if the number again divides by i, it cannot be a square free number
                if (num % i == 0)
                    return false;
            }
        }
        return true;
    }
}
