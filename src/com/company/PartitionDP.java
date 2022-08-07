package com.company;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class PartitionDP {

    // MCM Recursive approach
    // Algorithm
    // Consider A = [a, b], B = [b, c] so the total number of operations in A*B = a * b * c
    // TC = O(n2*n) // {i,j} -> total numbe rof possible states is n2 and 1 loop is running n times
    // SC = O(n2) + stack space
    public static int matrixMultiplication(int[] arr, int N) {
        int[][] dp = new int[N][N];
        for (int[] d : dp) Arrays.fill(d, -1);
        return helper(arr, 1, N - 1, dp);
    }

    private static int helper(int[] arr, int i, int j, int[][] dp) {
        // base case
        if (i == j) return 0;

        if (dp[i][j] != -1) return dp[i][j];
        // Try out all possible cases
        int ans = Integer.MAX_VALUE;
        for (int k = i; k < j; k++)
            ans = Math.min(ans, arr[i - 1] * arr[k] * arr[j] + helper(arr, i, k, dp) + helper(arr, k + 1, j, dp));
        //return mininum
        return dp[i][j] = ans;
    }


    // MCM Pattern
    // TC = O(n2*n) // {i,j} -> total numbe rof possible states is n2 and 1 loop is running n times
    // SC = O(n2)
    // Create a DP of size i,j
    // Copy the base case
    // Write down changing parameter in opposite fashion
    // Copy the recurrence
    // return dp[1][N-1] // First recursive call
    // TC = O(n3), SC = O(n2)
    public static int matrixMultiplicationTD(int[] arr, int N) {

        int[][] dp = new int[N][N];
        for (int i = 1; i < N; i++) dp[i][i] = 0;

        for (int i = N - 1; i >= 1; i--) {
            for (int j = i + 1; j < N; j++) {
                int ans = Integer.MAX_VALUE;
                for (int k = i; k < j; k++) {

                    ans = Math.min(ans, arr[i - 1] * arr[k] * arr[j] + dp[i][k] + dp[k + 1][j]);

                }
                dp[i][j] = ans;
            }
        }

        return dp[1][N - 1];

    }


    // Author: Anand
    // Partition DP based problem
    // TC = O(n3), SC = O(n2) + Stack space
    public int minCost(int n, int[] cuts) {
        List<Integer> cutsl = new ArrayList<>();
        cutsl.add(0);
        cutsl.addAll(Arrays.stream(cuts).boxed().collect(Collectors.toList()));
        cutsl.add(n);
        Collections.sort(cutsl);

        int[][] dp = new int[cuts.length + 1][cuts.length + 1];

        for (int[] d : dp) Arrays.fill(d, -1);
        return mcf(cutsl, 1, cuts.length, dp);
    }

    private int mcf(List<Integer> cutsl, int i, int j, int[][] dp) {
        // base case
        if (i > j) return 0;

        if (dp[i][j] != -1) return dp[i][j];
        int cost = Integer.MAX_VALUE;

        for (int k = i; k <= j; k++) {
            cost = Math.min(cost, cutsl.get(j + 1) - cutsl.get(i - 1) + mcf(cutsl, i, k - 1, dp) + mcf(cutsl, k + 1, j, dp));
        }
        return dp[i][j] = cost;
    }


    // Steps for Top-Down DP
    // Create a DP of size i,j
    // Copy the base case
    // Write down changing parameter in opposite fashion
    // if i > j --> continue (As we have already initialised them with value 0)
    // Copy the recurrence
    // return dp[1][c] // First recursive call
    // TC = O(n3), SC = O(n2)
    public static int costTD(int n, int c, int cuts[]) {
        List<Integer> cutsl = new ArrayList<>();
        cutsl.add(0);
        cutsl.addAll(Arrays.stream(cuts).boxed().collect(Collectors.toList()));
        cutsl.add(n);
        Collections.sort(cutsl);

        int[][] dp = new int[c + 2][c + 2];

        for (int i = c; i >= 1; i--) {
            for (int j = 1; j <= c; j++) {
                if (i > j) continue;
                int cost = Integer.MAX_VALUE;
                for (int ind = i; ind <= j; ind++) {
                    cost = Math.min(cost, cutsl.get(j + 1) - cutsl.get(i - 1) + dp[i][ind - 1] + dp[ind + 1][j]);
                }
                dp[i][j] = cost;
            }
        }

        return dp[1][c];
    }


    //Author: Anand
    /*
    The idea is to  go from last remaining ballon bursted to the first one
    In this wasy we will be able to make sub-problems non-overlapping
    and solve them recursively
    for eg:-
     [b1 b2 b3 b4]
    Lets say :-
    Last ballon busted can be any one of them --> Hence loop iteration is done
    Assume b3 was bursted we are left with [b1 b2] [b4] as sub-problems
    Now in the 2nd last step possiblity can be any 1 of them {b1 b3 } {b2 b3} {b3 b4}
    and Hence we can say that b1 b2 is nowhere depeendene on  each other and
    non-overllaping sub-problems observed.
    Cost = arr[b3]*arr[b1]*arr[b4] =~ nl.get(i - 1) * nl.get(ind) * nl.get(j + 1)
    Solve sub-problems recursively
    Return max cost
    TC =~ O(n3), SC = O(n2) + auxillary Stack space
     */
    public int maxCoins(int[] nums) {
        List<Integer> nl = new ArrayList<>();
        nl.add(1);
        nl.addAll(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        nl.add(1);
        int[][] dp = new int[nums.length + 1][nums.length + 1];
        for (int[] d : dp) Arrays.fill(d, -1);
        return f(nl, 1, nums.length, dp);
    }

    private int f(List<Integer> nl, int i, int j, int[][] dp) {
        // base case
        if (i > j) return 0;

        if (dp[i][j] != -1) return dp[i][j];
        int cost = Integer.MIN_VALUE;
        for (int ind = i; ind <= j; ind++) {
            cost = Math.max(cost, nl.get(i - 1) * nl.get(ind) * nl.get(j + 1)
                    + f(nl, i, ind - 1, dp) + f(nl, ind + 1, j, dp));
        }
        return dp[i][j] = cost;
    }

    // TC =~ O(n3), SC = O(n2)
    public int maxCoinsTD(int[] nums) {
        List<Integer> nl = new ArrayList<>();
        nl.add(1);
        nl.addAll(Arrays.stream(nums).boxed().collect(Collectors.toList()));
        nl.add(1);

        int[][] dp = new int[nums.length + 2][nums.length + 2];

        for (int i = nums.length; i >= 1; i--) {
            for (int j = 1; j <= nums.length; j++) {
                // base case
                if (i > j) continue;

                int cost = Integer.MIN_VALUE;
                for (int ind = i; ind <= j; ind++) {
                    cost = Math.max(cost, nl.get(i - 1) * nl.get(ind) * nl.get(j + 1)
                            + dp[i][ind - 1] + dp[ind + 1][j]);
                }
                dp[i][j] = cost;
            }
        }

        return dp[1][nums.length];
    }

    //Author: Anand
    //TC = O(mn), SC = O(mn)
       /*
        The idea is to make horizontal cuts ie, if i make a cut of {i,j} I can further convert it into pieces of (ii, j) and (i-ii, j) and get the max ans
        Repeat above steps for vertical cut
        return max result.
        */
    public long sellingWood(int m, int n, int[][] prices) {
        long[][] memo = new long[m + 1][n + 1];
        for (int i = 0; i <= m; i++) Arrays.fill(memo[i], -1);
        long[][] p = new long[m + 1][n + 1]; // prices of cut of dimnention {i*j}

        for (int[] pri : prices) p[pri[0]][pri[1]] = pri[2];
        return dp(m, n, p, memo);
    }

    private long dp(int i, int j, long[][] p, long[][] memo) {
        // base case
        if (i == 0 || j == 0) return 0;

        if (memo[i][j] != -1) return memo[i][j];
        long ans = p[i][j];

        for (int ii = 1; ii <= i / 2; ii++) {
            ans = Math.max(ans, dp(ii, j, p, memo) + dp(i - ii, j, p, memo));
        }

        for (int jj = 1; jj <= j / 2; jj++) {
            ans = Math.max(ans, dp(i, jj, p, memo) + dp(i, j - jj, p, memo));
        }
        return memo[i][j] = ans;
    }

    /*
    Input: nums = [4,4,4,5,6]
    Output: true
    Explanation: The array can be partitioned into the subarrays [4,4] and [4,5,6].
    This partition is valid, so we return true.
     */
    //Author: Anand
    public boolean validPartition(int[] nums) {

        int n = nums.length;
        int[] dp = new int[nums.length];
        Arrays.fill(dp, -1);
        return ((nums[0] == nums[1] && partition(2, nums, dp)) ||
                ((n > 2 && nums[0] == nums[1] && nums[1] == nums[2]) && partition(3, nums, dp)) ||
                ((n > 2 && nums[0] + 1 == nums[1] && nums[1] + 1 == nums[2]) && partition(3, nums, dp)));
    }

    private boolean partition(int ind, int[] nums, int[] dp) {

        if (ind >= nums.length) return true;

        if (dp[ind] != -1) return dp[ind] == 1;

        if ((ind + 1 < nums.length && (nums[ind] == nums[ind + 1]) && partition(ind + 2, nums, dp))
                ||
                (ind + 2 < nums.length && (nums[ind] == nums[ind + 1] && nums[ind + 1] == nums[ind + 2]) && partition(ind + 3, nums, dp))
                ||
                (ind + 2 < nums.length && (nums[ind] + 1 == nums[ind + 1] && nums[ind + 1] + 1 == nums[ind + 2]) && partition(ind + 3, nums, dp))) {
            dp[ind] = 1;
            return true;
        }
        dp[ind] = 0;
        return false;
    }


}
