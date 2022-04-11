package com.company;

import java.util.*;

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
        for (int i = 0; i < n; i++) cnt += corridor.charAt(i) == 'S' ? 1 : 0;

        // If no even seats are available possible arrangements can't be done
        if (cnt % 2 != 0) return 0;

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

    // TC = O(n*nCk), SC = O(n*nCk)
    static long maxSum;

    public static long maximumSum(ArrayList<Integer> nums, int k) {
        maxSum = Long.MIN_VALUE;

        int n = nums.size();
        if (n < k) return 0;

        sub(nums, k, 0, 0, Long.MIN_VALUE);
        return maxSum == Long.MIN_VALUE ? -1 : maxSum;
    }

    private static void sub(ArrayList<Integer> nums, int k, int ind, long sum, long prev) {
        // base case
        if (k == 0) {
            maxSum = Math.max(maxSum, sum);
            return;
        }

        if (k < 0 || ind >= nums.size()) {
            return;
        }

        if (nums.get(ind) >= prev && nums.get(ind) > 0) {
            int num = nums.get(ind);
            // take
            sum += num;
            sub(nums, k - 1, ind + 1, sum, nums.get(ind));
            sum -= num;
            sub(nums, k, ind + 1, sum, prev);
        } else sub(nums, k, ind + 1, sum, prev);
    }

    // TOP-DOWN DP
    // TC = O(n2*k)
    // SC = O(n*k)

    public static long maximumSumOptimised(ArrayList<Integer> nums, int k) {

        int n = nums.size();
        if (n < k) return 0;

        long[][] dp = new long[n][k + 1]; // maximumSum of a subsequence of size = k ending at ind = n
        for (int i = 0; i < n; i++) Arrays.fill(dp[i], -1);


        // Fill for subsequence of size = 1 it'll have current/first element only
        for (int i = 0; i < n; i++) dp[i][1] = nums.get(i);

        for (int endIndex = 1; endIndex < n; endIndex++) {
            for (int prev = 0; prev < endIndex; prev++) {

                // valid choice
                if (nums.get(prev) <= nums.get(endIndex)) {
                    for (int size = 2; size <= k; size++) {
                        if (dp[prev][size - 1] != -1) {
                            dp[endIndex][size] = Math.max((dp[prev][size - 1] + nums.get(endIndex)), dp[endIndex][size]);
                        }
                    }
                }
            }
        }

        long max = Long.MIN_VALUE;
        // For all ending index get  the max_value for size=k
        for (int endIndex = 0; endIndex < n; endIndex++) {
            max = Math.max(max, dp[endIndex][k]);
        }
        return max;

    }


    // TODO : Complete it
    static int ans = 0;

    // TC = O(m*n*2), SC=O(n)
    public static int sellMaximum(ArrayList<Integer> desiredSize, ArrayList<Integer> apartmentSize, int k) {
        int n = desiredSize.size();
        int m = apartmentSize.size();

        int[] dp = new int[n]; // dp stores for each desiredSize the maximum no. of apartement that can be sold out
        Arrays.fill(dp, -1);
        sellMax(desiredSize, apartmentSize, k, 0, 0, new ArrayList<>(), dp);

        return ans;
    }

    private static int sellMax(ArrayList<Integer> desiredSize, ArrayList<Integer> apartmentSize, int k, int ind, int ca, List<Integer> taken, int[] dp) {
        int n = desiredSize.size();
        int m = apartmentSize.size();
        // base-cases
        if (ind >= n) {
            ans = Math.max(ans, ca);
            return ans;
        }

        if (dp[ind] != -1) return dp[ind];
        for (int i = 0; i < m; i++) {
            // If curr apartment can be taken
            if ((apartmentSize.get(i) <= desiredSize.get(ind) + k)
                    && (apartmentSize.get(i) >= desiredSize.get(ind) - k)
                    && !taken.contains(apartmentSize.get(i))
            ) {
                // take
                taken.add(apartmentSize.get(i));
                dp[ind] = sellMax(desiredSize, apartmentSize, k, ind + 1, ca + 1, taken, dp);
                // backtrack
                taken.remove(apartmentSize.get(i));
            }
        }
        return dp[ind];
    }

    /*
    Input: s = "1100101"
    Output: 5
    Explanation:
    One way to remove all the cars containing illegal goods from the sequence is to
    - remove a car from the left end 2 times. Time taken is 2 * 1 = 2.
    - remove a car from the right end. Time taken is 1.
    - remove the car containing illegal goods found in the middle. Time taken is 2.
    This obtains a total time of 2 + 1 + 2 = 5.

    An alternative way is to
    - remove a car from the left end 2 times. Time taken is 2 * 1 = 2.
    - remove a car from the right end 3 times. Time taken is 3 * 1 = 3.
    This also obtains a total time of 2 + 3 = 5.

    5 is the minimum time taken to remove all the cars containing illegal goods.
    There are no other ways to remove them with less time.
     */
    // Author: Anand
    // TC = O(n)
    public int minimumTime(String s) {

        int n = s.length();
        if (n == 1) return s.charAt(0) - '0';

        int[] dp1 = new int[n]; // the minimum steps of deletion required from left or middle
        int[] dp2 = new int[n]; // the minimum steps of deletion required from right or middle

        // Traverse from left to right and find the optimal number of deletion if 1
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) - '0' == 0) {
                dp1[i] = i > 0 ? dp1[i - 1] : 0;
            } else {
                dp1[i] = i > 0 ? Math.min(2 + dp1[i - 1], i + 1) : i + 1;
            }
        }


        // Traverse from right to left and find the optimal number of deletion if 1
        for (int i = n - 1; i >= 0; i--) {
            if (s.charAt(i) - '0' == 0) {
                dp2[i] = i < n - 1 ? dp2[i + 1] : 0;
            } else {
                dp2[i] = i < n - 1 ? Math.min(2 + dp2[i + 1], n - i) : n - i;
            }
        }

        // Evaluation  min possible answer for a valid index
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < n - 1; i++) {
            res = Math.min(res, dp1[i] + dp2[i + 1]);
        }
        return res;
    }

    public int findNumberOfLIS(int[] nums) {
        int ans = 0;
        return ans;
    }

    // Author: Anand
    // TC = O(mn)
    public int maxValueOfCoins(List<List<Integer>> piles, int k) {
        int[][] dp = new int[piles.size()][k + 1];
        for (int[] e : dp) Arrays.fill(e, -1);

        return (int) f(piles, 0, k, dp);
    }

    private long f(List<List<Integer>> piles, int idx, int k, int[][] dp) {
        // base case
        if (idx >= piles.size() || k <= 0) return 0;

        if (dp[idx][k] != -1) return dp[idx][k];

        // not take
        long best = f(piles, idx + 1, k, dp);
        long sum = 0;
        // take
        for (int i = 1; i <= Math.min(k, piles.get(idx).size()); i++) {
            sum += piles.get(idx).get(i - 1);
            best = Math.max(best, sum + f(piles, idx + 1, (k - i), dp));
        }
        return dp[idx][k] = (int) best;
    }

    // Author: Anand
    public long numberOfWaysDp(String s) {
        long[][][] dp = new long[100003][3][4];
        for (long[][] r : dp) {
            for (long[] c : r) Arrays.fill(c, -1);
        }

        return cntWays(s, 0, 0, 9, dp);
    }

    private long cntWays(String s, int i, int cnt, int prev, long[][][] dp) {
        // base case
        if (cnt == 3) return 1;
        if (i >= s.length()) return 0;

        if (dp[i][prev][cnt] != -1) return dp[i][prev][cnt];

        long take = 0;
        if (prev != s.charAt(i)) {
            take = cntWays(s, i + 1, cnt + 1, (int) s.charAt(i) - '0', dp);
        }
        long ntake = cntWays(s, i + 1, cnt, (int) prev - '0', dp);

        long ans = take + ntake;

        dp[i][prev][cnt] = ans;
        return ans;
    }

    private long maxf(int[] nums, int idx, int k, long prod) {
        // base case
        if (idx >= nums.length || k <= 0) return 0;

        // not take
        long best = maxf(nums, idx + 1, k, prod);
        // take
        for (int i = 0; i < Math.min(k, nums.length); i++) {
            prod *= (nums[i] + 1);
            best = Math.max(best, prod * maxf(nums, idx + 1, (k - i), prod));
        }
        return best;
    }

    //Author : Anand
    // TODO : Complete this
    public long maximumBeauty(int[] flowers, long newFlowers, int target, int full, int partial) {
        Arrays.sort(flowers);
        return mb(flowers, newFlowers, full, partial, target, 0, 0);
    }

    private long mb(int[] flowers, long newFlowers, int full, int partial, int target, int idx, long sum) {
        // base case

        if (idx == flowers.length) return 0;
        long max = Integer.MIN_VALUE;
        // take
        for (int i = 0; i < target; i++) {
            newFlowers -= i;
            if (newFlowers >= 0 && flowers[idx] + i >= target) sum += full;
            max = Math.max(max, sum + mb(flowers, newFlowers, full, partial, target, idx + 1, sum));
        }

        // Let's not add new flowers to garden
        int mini = Integer.MAX_VALUE;
        for (int flower : flowers) mini = Math.min(mini, flower);

        return Math.max(max, sum + (long) mini * partial);
    }

}
