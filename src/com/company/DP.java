package com.company;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

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

}

