package com.company;

import java.lang.reflect.Array;
import java.util.*;

import javafx.util.Pair;

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


    int MOD = 1_000_000_000 + 7;


    // DP based solution
    // TC = O(n * no. of states * no. of different recursive calls)
    public int numberOfWays(String corridor) {
        int n = corridor.length();
        if (n == 1) return 0;
        int cnt = 0;
        for (int i = 0; i < n; i++) cnt += corridor.charAt(i) == 'S' ? 1 : 0 ;

        // If no even seats are available possible arrangements can't be done
        if(cnt % 2 != 0) return 0;

        int[][] dp = new int[n][3];
        for (int i = 0; i < n; i++) Arrays.fill(dp[i], -1);
        return recurse(0, 0, corridor, dp);
    }

    // Let's assume we place barrier before index `ind` everytime
    private int recurse(int ind, int seats, String corridor, int[][] dp) {
        // base case
        if (ind > corridor.length()) return 0;
        if (ind == corridor.length()) {
            if (seats == 2) return 1;
            return 0;
        }

        if (dp[ind][seats] != -1) return dp[ind][seats];
        int calls = 0;

        // if seats == 2 and current is plant then  we have 2 choices either place the barrier or not
        if (seats == 2) {
            if (corridor.charAt(ind) == 'P') {
                // place barrier
                // if you  place a barrier then no of seats will become 0
                calls += recurse(ind + 1, 0, corridor, dp) % MOD;
                // skip
                calls += recurse(ind + 1, seats, corridor, dp) % MOD;
            } else {
                // place barrier
                // if you  place a barrier then no of seats will become 1 (as curr is a seat)
                calls += recurse(ind + 1, 1, corridor, dp) % MOD;
            }
        } else {
            if (corridor.charAt(ind) == 'S') {
                calls += recurse(ind + 1, seats + 1, corridor, dp) % MOD;
            } else
                calls += recurse(ind + 1, seats, corridor, dp) % MOD;
        }
        return dp[ind][seats] = calls;
    }


    // Similar to Frog jump
    // Author: Anand
    //    [0,1,2,3,0]
    //    weekly-contest-236/problems
    public int minSideJump(int[] obstacles) {
        int ans = 0;
        int lane = 2; // lane can have values  = {1 2 3}
        Map<String, Integer> map = new HashMap<>();
        return recurse(obstacles, 0, ans, lane, map);
    }


    private int recurse(int[] obstacles, int i, int ans, int lane, Map<String, Integer> map) {

        // base case
        if (i >= obstacles.length - 1) {
            return ans;
        }

        String key = i + "-" + lane + "-" + ans;

        if (map.containsKey(key)) return map.get(key);

        // obstacles in current lane
        if (obstacles[i + 1] == lane) {
            // 2 possible cases
            // either go up or down
            int left = Integer.MAX_VALUE, right = Integer.MAX_VALUE;
            if (lane == 2) {
                // up
                if (obstacles[i] != lane - 1) {
                    lane--;
                    left = recurse(obstacles, i, ans + 1, lane, map);
                    // backtrack
                    lane++;
                }

                // down
                if (obstacles[i] != lane + 1) {
                    lane++;
                    right = recurse(obstacles, i, ans + 1, lane, map);
                }
            } else if (lane == 3) {
                // up
                if (obstacles[i] != lane - 1) {
                    lane--;
                    left = recurse(obstacles, i, ans + 1, lane, map);
                    // backtrack
                    lane++;
                }

                // up
                if (obstacles[i] != lane - 2) {
                    lane -= 2;
                    right = recurse(obstacles, i, ans + 1, lane, map);
                }
            } else {
                // down
                if (obstacles[i] != lane + 1) {
                    lane += 1;
                    left = recurse(obstacles, i, ans + 1, lane, map);
                    // backtrack
                    lane -= 1;
                }

                // down
                if (obstacles[i] != lane + 2) {
                    lane += 2;
                    right = recurse(obstacles, i, ans + 1, lane, map);
                }
            }

            map.put(key, Math.min(left, right));
            return Math.min(left, right);
        }


        int next = recurse(obstacles, i + 1, ans, lane, map);
        map.put(key, next);
        return next;
    }

    // Author: Anand
    // TC = O(n)
    public int minSideJumpsIterative(int[] obstacles) {
        int n = obstacles.length;
        int[][] dp = new int[n][3];
        dp[0][1] = 0;
        dp[0][0] = 1;
        dp[0][2] = 1;

        for (int i = 1; i < n; i++) {
            for (int j = 0; j < 3; j++) {
                // obstacle in current lane
                if (obstacles[i] == j + 1) {
                    dp[i][j] = 1_00_0000;
                } else {
                    // same jumps as previous
                    dp[i][j] = dp[i - 1][j];
                }
            }

            // cases
            for (int j = 0; j < 3; j++) {
                // no obstacle in current lane
                if (obstacles[i] != j + 1) {
                    // Take possible cases of side jumps as well to consider the minimum moves
                    // if  no obstacle on side jump
                    int lane1 = dp[i][(j + 1) % 3] + 1;
                    int lane2 = dp[i][(j + 2) % 3] + 1;

                    // Get min moves after checking either take side jumps or stay in same lane
                    dp[i][j] = Math.min(Math.min(lane1, lane2), dp[i][j]);
                }
            }
        }
        return Math.min(Math.min(dp[n - 1][0], dp[n - 1][1]), dp[n - 1][2]);
    }
}