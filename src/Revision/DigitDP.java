package com.company;

import java.util.HashMap;
import java.util.Map;


public class DigitDP {


    final int mod = 1_000_000_007;

    /*
     * PROBLEM: Stepping Numbers in Range (LeetCode 2801)
     * Count all stepping numbers in the inclusive range [low, high], where adjacent digits differ by exactly 1.
     *
     * ALGORITHM: Digit DP with memoization
     * TC: O(N * D * 10) where N = length of high, D = digits | SC: O(N * D)
     *
     * Example: low = "1", high = "11" → Output: 10 (1,2,...,9,10 are all stepping numbers in range)
     */
    public int countSteppongStone(String low, String high) {
        Map<String, Integer> dp = new HashMap<>();
        return helper(low, high, new StringBuilder(), high.length() - 1, dp);
    }

    /*
     * PROBLEM: Stepping Numbers in Range – recursive counter (Helper)
     * Recursively builds valid stepping numbers digit by digit up to the bound defined by high.
     *
     * ALGORITHM: Digit DP with top-down memoization and backtracking
     * TC: O(N * 10) per state, O(N^2 * 10) overall | SC: O(N^2) for memo map
     */
    private int helper(String low, String high, StringBuilder sb, int ind, Map<String, Integer> dp) {
        // base case
        if (ind < 0) {
            if (sb.length() > 0) return 1;
            return 0;
        }

        int cnt = 0;
        String key = sb.toString() + "-" + ind;
        if (dp.containsKey(key)) return dp.get(key);

        // Traverse through all digits of high
        for (int i = ind; i >= 0; i--) {
            //take the curr digit
            int curr = high.charAt(i);
            for (int j = 0; j < curr; j++) {
                if (sb.length() == 0 || Math.abs(sb.charAt(sb.length() - 1) - j) == 1) {
                    sb.append(j);
                    cnt = (cnt + helper(low, high, sb, ind - 1, dp)) % mod;
                    //backtrack
                    sb.deleteCharAt(sb.length() - 1);
                }
                //skip i.e. not-take the current digit
                cnt = (cnt + helper(low, high, sb, ind - 1, dp)) % mod;
            }
        }

        dp.put(key, cnt);
        return cnt;
    }
}
