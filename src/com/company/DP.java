package com.company;

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