package com.company;

import java.util.Arrays;

public class DP {

    // Author: Anand
    // Solution using DP
    // TC = O(n)
    // Add memoization to  improve TC
    public int fib(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, -1);
        return eval(n, dp);
    }

    private int eval(int n, int[] dp) {
        if (n == 0) return 0;
        if (n == 1) return 1;

        if (dp[n] != -1) return dp[n];
        return dp[n] = fib(n - 1) + fib(n - 2);
    }

    // DP to tabulation:-= Bottom-up approach
    // TC = O(n), SC = O(n)
    public static int frogJump(int n, int heights[]) {
        int[] dp = new int[n];
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            int left = dp[i - 1] + Math.abs(heights[i] - heights[i - 1]);

            int right = Integer.MAX_VALUE;
            if (i > 1) {
                right = dp[i - 2] + Math.abs(heights[i] - heights[i - 2]);
            }

            dp[i] = Math.min(left, right);
        }
        return dp[n - 1];
    }

    // TODO: Completed this code
    public int numberOfWays(String corridor) {
        long[][] dp = new long[100005][3];
        for (long[] longs : dp) {
            Arrays.fill(longs, -1);
        }
        return (int) help(0, corridor, 0, dp);

    }

    private long help(int idx, String s, int cnt, long[][] dp) {
        // base cases
        if (cnt > 2) return 0;
        //if we reach the end of string and have found 2 seats in left then return 1 else 0
        if (idx == s.length()) {
            cnt = 2;
            return cnt;
        }
        // dp condition
        if (dp[idx][cnt] != -1)
            return dp[idx][cnt];

        long ans = 0;

        // check present element is plant or seat
        long MOD = 1_0000_00000 + 7;
        if (s.charAt(idx) == 'P') {
            ans = (ans + help(idx + 1, s, cnt, dp)) % MOD;
            if (cnt == 2) // chek if already we have 2 seats in left, so skip this and initialise cnt
                ans = (ans + help(idx + 1, s, 0, dp)) % MOD;
        } else {
            ans = (ans + help(idx + 1, s, cnt + 1, dp) % MOD);
            if (cnt == 2) {    // chek if already we have 2 seats in left, then consider this seat for next Division
                ans = (ans + help(idx + 1, s, 1, dp)) % MOD;
            }
        }
        return dp[idx][cnt] = ans;
    }
}

