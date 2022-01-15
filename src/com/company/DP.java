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


    /*

    [0,1,1,1,0,0,1,1,0], [0,1,1,1,0,0,1,1,0] ->  [1,1,0,0,0,0,1]
    n = 9/2=4
    cntc  = 2

    [1,1,0,0,1]

    [1,1,1,0,0,1,0,1,1,0]

    nums=[0,1,0,1,1,0,0]
    total=3
    n = 7
    i=0;
    ptr=1




     */
    // TC = O(n)
    // Author: Anand
    public static int minSwaps(int[] nums) {
        int n = nums.length;

        int total = 0;
        for (int num : nums) if (num == 1) total++;

        int sbArray = Integer.MAX_VALUE;
        int ptr = 0;

        // Create all subarrays of length=total
        for (int i = 0; i <= n - total; i++) {
            if (i == 0) {
                for (int j = i; j < total; j++) {
                    if (nums[j] == 0) ptr++;
                }
            } else {
                if (nums[i - 1] == 0) ptr--;
                if (nums[i - 1 + total] == 0) ptr++;
            }
            sbArray = Math.min(sbArray, ptr);
        }

        int[] newArray = Arrays.copyOf(nums, 2 * n);
        System.arraycopy(nums, 0, newArray, n, n);

        n = 2 * n;
        ptr = 0;

        // Create all subarrays of length=total
        for (int i = 0; i <= n - total; i++) {
            if (i == 0) {
                for (int j = i; j < total; j++) {
                    if (newArray[j] == 0) ptr++;
                }
            } else {
                if (newArray[i - 1] == 0) ptr--;
                if (newArray[i - 1 + total] == 0) ptr++;
            }
            sbArray = Math.min(sbArray, ptr);
        }

        return sbArray;
    }

}

