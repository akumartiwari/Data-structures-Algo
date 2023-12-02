package com.company;

import java.util.HashMap;
import java.util.*;

public class DP {

    /*
    Input: s = "abcabcdabc"
    Output: 2
    Explanation:
    - Delete the first 3 letters ("abc") since the next 3 letters are equal. Now, s = "abcdabc".
    - Delete all the letters.
    We used 2 operations so return 2. It can be proven that 2 is the maximum number of operations needed.
    Note that in the second operation we cannot delete "abc" again because the next occurrence of "abc" does not happen in the next 3 letters.
     */

    // TC = O(n*nCk), SC = O(n*nCk)
    static long maxSum;
    // TODO : Complete it
    static int ans = 0;
    final int mod = 1_000_000_007;
    Integer[] dp;
    int MOD = 1_000_000_000 + 7;
    int sp;
    int op;
    //Author : Anand
    int m, n;
    Map<Integer, List<int[]>> map;
    Integer[][] cpDP;
    List<Integer> robot;
    int[][] factory;

    // DP to tabulation:-= Bottom-up approach
    // TC = O(n), SC = O(n)
    public static int frogJump(int n, int[] heights) {
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

    // TOP-DOWN DP
    // TC = O(n2*k)
    // SC = O(n*k)
    private static int ls(int ind, int[] arr, int num, int k, int len, int[] dp) {
        if (ind < 0) {
            if (num <= k) {
                return len;
            }
            return 0;
        }


        if (num > k) return -1;
        if (dp[ind] != -1) return dp[ind];

        int nt = Integer.MIN_VALUE, t = Integer.MIN_VALUE;

        if (!(arr[ind] == 1 && num >= k)) {
            // take
            num += arr[ind] == 1 ? Math.pow(2, len) : 0;
            t = ls(ind - 1, arr, num, k, len + 1, dp);
            num -= arr[ind] == 1 ? Math.pow(2, len) : 0; // backtrack
        }

        // not-take
        nt = ls(ind - 1, arr, num, k, len, dp);
        return dp[ind] = Math.max(t, nt);
    }

    public int deleteString(String s) {
        dp = new Integer[s.length()];
        return helper(s.toCharArray(), 0);
    }

    private int helper(char[] s, int index) {
        if (index >= s.length) return 0;

        if (dp[index] != null) return dp[index];

        int res = 0;
        boolean found = false;
        for (int i = index; i < index + (s.length - index) / 2; i++) {
            if (dp[i] != null) continue;
            if (isEqual(s, index, i + 1, i + 1, i + 1 + (i - index + 1))) {
                found = true;
                res = Math.max(res, 1 + helper(s, i + 1));
            }
        }
        if (!found) return dp[index] = 1;
        return dp[index] = res;
    }

    private boolean isEqual(char[] s, int st1, int en1, int st2, int en2) {
        boolean ans = true;
        for (int i = st1, j = st2; i < en1 && j < en2; i++, j++) {
            if (s[i] != s[j]) {
                ans = false;
                break;
            }
        }
        return ans;
    }

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


    //TODO:TLE
    class Solution {
        public int distributeCandies(int n, int limit) {
            int[][] dp = new int[n + 1][4];
            for (int[] d : dp) Arrays.fill(d, -1);
            return helper(n, limit, 3, dp);
        }

        private int helper(int n, int limit, int child, int[][] dp) {
            // base case
            if (child == 0) {
                if (n == 0) return 1;
                return 0;
            }

            if (n < 0) return 0;

            if (dp[n][child] != -1) return dp[n][child];

            int cnt = 0;
            for (int i = 0; i <= n; i++) {
                if (i <= limit && (child == 1 || (child - 1) * limit >= (n - i))) {
                    cnt += helper(n - i, limit, child - 1, dp);
                }
            }

            return dp[n][child] = cnt;
        }
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
        long ntake = cntWays(s, i + 1, cnt, prev - '0', dp);

        long ans = take + ntake;

        dp[i][prev][cnt] = ans;
        return ans;
    }

    /*
    Input: nums1 = [4,0,1,3,2], nums2 = [4,1,0,2,3]
    Output: 4
   */
    // Author: Anand
    // TC = O(2n)
    public long goodTriplets(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();// To look-up elements in nums2
        int idx = 0;
        for (int e : nums2) map.put(e, idx++);
        Map<String, Long> dp = new HashMap<>();// To look-up elements in nums2

        return recurse(nums1, 0, 3, true, map, new ArrayList<>(), dp);
    }

    private long recurse(int[] nums1, int idx, int t, boolean valid, Map<Integer, Integer> map, List<Integer> choices, Map<String, Long> dp) {
        // base case
        if (t == 0) {
            if (valid) return 1;
            return 0;
        }

        if (idx >= nums1.length) return 0;

        if (!valid) return 0;
        String key = idx + "-" + t;
        if (dp.containsKey(key)) dp.get(key);

        long take = 0, nt = 0;
        // t
        if (choices.size() == 0 || map.get(nums1[idx]) > choices.get(choices.size() - 1)) {
            choices.add(map.get(nums1[idx]));
            take += recurse(nums1, idx + 1, t - 1, valid, map, choices, dp);
            choices.remove(choices.size() - 1); // remove at last index in O(1)
        }

        // nt
        nt += recurse(nums1, idx + 1, t, valid, map, choices, dp);
        long ans = take + nt;
        dp.put(key, ans);
        return ans;
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

    /*
    Input: grid = [[5,3],[4,0],[2,1]], moveCost = [[9,8],[1,5],[10,12],[18,6],[2,4],[14,3]]
    `    Output: 17
    Explanation: The path with the minimum possible cost is the path 5 -> 0 -> 1.
    - The sum of the values of cells visited is 5 + 0 + 1 = 6.
    - The cost of moving from 5 to 0 is 3.
    - The cost of moving from 0 to 1 is 8.
    So the total cost of the path is 6 + 3 + 8 = 17.
`     */
    //Author: Anand
    public int minPathCost(int[][] grid, int[][] moveCost) {
        int row = grid.length;
        int col = grid[0].length;
        int[][] dp = new int[row][col]; // minimum cost to reach at r,c cell

        // For every cell in 1st row fill the cost
        System.arraycopy(grid[0], 0, dp[0], 0, col);
        // For remaining rows fill this
        for (int r = 1; r < row; r++) {
            for (int c = 0; c < col; c++) {
                dp[r][c] = getMin(grid, moveCost, dp, r, c);
            }
        }

        // For the last row just check the minimum to reach at any cell
        int result = Integer.MAX_VALUE;
        for (int r : dp[row - 1]) {
            result = Math.min(result, r);
        }
        return result;
    }

    private int getMin(int[][] grid, int[][] moveCost, int[][] dp, int row, int col) {
        // base case
        int min = Integer.MAX_VALUE, prevRow = row - 1;
        // traverse through prev row and evaluate from which cell you are getting min cost
        // jump from prev row to current cell
        for (int c = 0; c < grid[0].length; c++) {
            min = Math.min(min, dp[prevRow][c] + moveCost[grid[prevRow][c]][col] + grid[row][col]);
        }
        return min;
    }

    /*
    Input: cookies = [8,15,10,20,8], k = 2
    Output: 31
    Explanation: One optimal distribution is [8,15,8] and [10,20]
    - The 1st child receives [8,15,8] which has a total of 8 + 15 + 8 = 31 cookies.
    - The 2nd child receives [10,20] which has a total of 10 + 20 = 30 cookies.
    The unfairness of the distribution is max(31,30) = 31.
    It can be shown that there is no distribution with an unfairness less than 31.

    TC = O(n2)
    */
    public int distributeCookies(int[] cookies, int k) {
        int[] cc = new int[k];
        return helper(cookies, k, cc, 0);
    }

    private int helper(int[] cookies, int k, int[] cc, int ind) {
        // base case
        if (ind == cookies.length) // all cookies are distributed
        {
            // which child is having max cookie
            int max = Integer.MIN_VALUE;
            for (int c : cc) {
                max = Math.max(max, c);
            }
            return max;
        }

        int ans = Integer.MAX_VALUE;
        for (int child = 0; child < k; child++) {
            cc[child] += cookies[ind];
            ans = Math.min(ans, helper(cookies, k, cc, ind + 1));
            cc[child] -= cookies[ind];// backtrack
        }

        return ans;
    }

    /*
    Input: s = "1001010", k = 5
    Output: 5
    Explanation: The longest subsequence of s that makes up a binary number less than or equal to 5 is "00010", as this number is equal to 2 in decimal.
    Note that "00100" and "00101" are also possible, which are equal to 4 and 5 in decimal, respectively.
    The length of this subsequence is 5, so 5 is returned.
    */
    //Author: Anand
    public int longestSubsequence(String s, int k) {
        int[] dp = new int[s.length()];
        Arrays.fill(dp, -1);
        return ls(s.length() - 1, s.chars().map(x -> x - '0').toArray(), 0, k, 0, dp);
    }


    private int gcd(int a, int b) {
        if (b == 0)
            return a;
        return gcd(b, a % b);
    }

    /*
    The idea behind solving this DP problem was to analyse whether backtracking is needed OR not.
    Whenever we iterate from all possible states (For eg. dice number = {1,2,3,4,5,6}) we will get
    all possible paths from  recursion tree starting from root (n-1) under given constraint. Hence
    backtracking is not needed to count total number of ways.
    TC = O(n*6*6*6); n*6*6 based on states of  recursive call and one more 6 is for loop inside recursive function
    */
    //
    public int distinctSequences(int n) {
        int[][][] dp = new int[10007][7][7];
        for (int[][] d : dp) {
            for (int[] x : d) {
                Arrays.fill(x, -1);
            }
        }
        return ds(0, 0, 0, n, dp);
    }

    private int ds(int ind, int l1, int l2, int n, int[][][] dp) {
        // base case
        if (ind >= n) {
            return 1;
        }

        if (dp[ind][l1][l2] != -1) return dp[ind][l1][l2];

        int take = 0;

        for (int i = 1; i <= 6; i++) {
            // take
            if ((l1 != i && l2 != i) && (l1 == 0 || (gcd(l1, i) == 1))) {
                take = (take + ds(ind + 1, i, l1, n, dp)) % MOD;
            }
        }

        return dp[ind][l1][l2] = take;
    }

    /*
    Input: arr = [10,13,12,14,15]
    Output: 2
    Explanation:
    From starting index i = 0, we can make our 1st jump to i = 2 (since arr[2] is the smallest among arr[1], arr[2], arr[3], arr[4] that is greater or equal to arr[0]), then we cannot jump any more.
    From starting index i = 1 and i = 2, we can make our 1st jump to i = 3, then we cannot jump any more.
    From starting index i = 3, we can make our 1st jump to i = 4, so we have reached the end.
    From starting index i = 4, we have reached the end already.
    In total, there are 2 different starting indices i = 3 and i = 4, where we can reach the end with some number of
    jumps.
     */

    //Author: Anand
    public int countHousePlacements(int n) {
        Map<String, Integer> dp = new HashMap<>();
        long ways = (count_ways_on_one_side(n, false, dp) + count_ways_on_one_side(n, true, dp)) % MOD;
        return (int) ((ways * ways) % MOD);
    }

    private int count_ways_on_one_side(int ind, boolean filled, Map<String, Integer> dp) {
        // base case
        if (ind <= 1) return 1;
        String key = ind + "-" + filled;
        if (dp.containsKey(key)) return dp.get(key);

        int ans = -1;
        if (filled) ans = count_ways_on_one_side(ind - 1, false, dp) % MOD;
        else ans = (count_ways_on_one_side(ind - 1, true, dp) + count_ways_on_one_side(ind - 1, filled, dp)) % MOD;

        dp.put(key, ans);
        return ans;
    }

    /*
    The idea is to use DP :-
    Consider both arrays 1 by 1 that will yield max result
    I have used Map<String, Integer> in place of 3d DP array but it was giving TLE
    as sometimes key check in map is more than O(1) (Be careful while using Map)
    Return Maximum of ans1, ans2
    */
    //Author: Anand
    public int maximumsSplicedArray(int[] nums1, int[] nums2) {
        int[][][] dp = new int[nums1.length][2][2];
        for (int[][] d : dp) {
            for (int[] e : d) {
                Arrays.fill(e, -1);
            }
        }


        int[] suff_sum1 = new int[nums1.length], suff_sum2 = new int[nums2.length];

        for (int i = nums1.length - 1; i >= 0; i--)
            suff_sum1[i] = (i == nums1.length - 1 ? nums1[i] : (suff_sum1[i + 1] + nums1[i]));
        for (int i = nums1.length - 1; i >= 0; i--)
            suff_sum2[i] = (i == nums2.length - 1 ? nums2[i] : (suff_sum2[i + 1] + nums2[i]));


        int ans1 = DP(nums1, nums2, 0, 0, false, false, suff_sum1, suff_sum2, dp);

        dp = new int[nums1.length][2][2];
        for (int[][] d : dp) {
            for (int[] e : d) {
                Arrays.fill(e, -1);
            }
        }

        int ans2 = DP(nums1, nums2, 0, 1, false, false, suff_sum1, suff_sum2, dp);
        return Math.max(ans1, ans2);
    }

    private int DP(int[] nums1, int[] nums2, int pos, int state, boolean prev_swap, boolean swapped, int[] suff_sum1, int[] suff_sum2, int[][][] dp) {
        // base case
        if (pos == nums1.length) return 0;

        if (dp[pos][prev_swap ? 1 : 0][swapped ? 1 : 0] != -1) return dp[pos][prev_swap ? 1 : 0][swapped ? 1 : 0];

        int ans = (prev_swap ? (state == 0 ? Math.max(nums2[pos] + DP(nums1, nums2, pos + 1, state, prev_swap, true, suff_sum1, suff_sum2, dp),
                nums1[pos] + DP(nums1, nums2, pos + 1, state, false, true, suff_sum1, suff_sum2, dp))
                : Math.max(nums1[pos] + DP(nums1, nums2, pos + 1, state, prev_swap, true, suff_sum1, suff_sum2, dp),
                nums2[pos] + DP(nums1, nums2, pos + 1, state, false, true, suff_sum1, suff_sum2, dp)))

                : (swapped ? ((state == 0 ? nums1[pos] + (pos == (nums1.length - 1) ? 0 : suff_sum1[pos + 1]) : nums2[pos] + (pos == (nums2.length - 1) ? 0 : suff_sum2[pos + 1])))
                : Math.max(
                (state == 0 ? nums1[pos] : nums2[pos]) + DP(nums1, nums2, pos + 1, state, prev_swap, swapped, suff_sum1, suff_sum2, dp),
                (state == 0 ? nums2[pos] : nums1[pos]) + DP(nums1, nums2, pos + 1, state, true, true, suff_sum1, suff_sum2, dp)
        )
        ));

        return dp[pos][prev_swap ? 1 : 0][swapped ? 1 : 0] = ans;
    }

    /*
     Input: s = "acfgbd", k = 2
     Output: 4
     Explanation: The longest ideal string is "acbd". The length of this string is 4, so 4 is returned.
     Note that "acfgbd" is not ideal because 'c' and 'f' have a difference of 3 in alphabet order.
    */
    //Author: Anand
    public int longestIdealString(String s, int k) {

        int[][] dp = new int[s.length()][64];
        for (int[] d : dp) Arrays.fill(d, -1);
        return Math.max(1 + ls(s, k, 1, s.charAt(0), dp), ls(s, k, 1, '#', dp));
    }

    private int ls(String s, int k, int ind, char prev, int[][] dp) {
        // base case
        if (ind >= s.length()) return 0;

        if (prev != '#' && dp[ind][(int) prev - 'a'] != -1) return dp[ind][(int) prev - 'a'];
        int take = 0, nt = 0;
        //take
        if (prev == '#' || Math.abs(prev - s.charAt(ind)) <= k) {
            take = 1 + ls(s, k, ind + 1, s.charAt(ind), dp);
        }
        // not take
        nt = ls(s, k, ind + 1, prev, dp);
        return dp[ind][Math.abs((int) prev - 'a')] = Math.max(take, nt);
    }

    /*
    1. int[][] dp[n][2], to save if from "i" can jump to final (1 for true, 0 for false). [2] dimension for "odd" or "even" jumps.
    2. TreeMap, is to find the location to jump, because TreeMap provides ceilingKey() and floorKey() to find the index to jump
    3. The recursion is: dp[n-1][:] = 1 (last element must be true).
       The position "i" will jump to is TreeMap.ceilingKey(A[i]) if odd jump, or TreeMap.floorKey(A[i]) if even jump.
       So dp[i][0] = dp[TreeMap.ceilingKey(A[i])][1], and dp[i][1] = dp[TreeMap.floorKey(A[i])][0].
       (That is, if i will oddly jump to k, and if we know k can evenly jump to final, then i can oddly jump to final. Similarly, if i will evenly jump to k, and if k can evenly jump to final, then i can evenly jump to final)
   4. Remember to add (key = A[i], value=i) in TreeMap,
       so that we can get A[i]'s position (smallest index if duplicated values)
     */
    public int oddEvenJumps(int[] arr) {
        int n = arr.length;
        int cnt = 0;
        TreeMap<Integer, Integer> tm = new TreeMap<>(); // jump from arr[ind]->ind
        tm.put(arr[n - 1], n - 1);
        int[][] dp = new int[n][2]; // 0 -> odd jump, 1-> even jump
        dp[n - 1][0] = 1;
        dp[n - 1][1] = 1;
        cnt++; //n-1

        for (int i = n - 2; i >= 0; i--) {
            // odd jump starting from index i
            Integer next = tm.ceilingKey(arr[i]);
            if (next == null) dp[i][0] = 0;
            else if (dp[tm.get(next)][1] > 0) dp[i][0] = 1;

            // even jump-starting from index i
            next = tm.floorKey(arr[i]);
            if (next == null) dp[i][1] = 0;
            else if (dp[tm.get(next)][0] > 0) dp[i][1] = 1;


            if (dp[i][0] == 1) cnt++;
            tm.put(arr[i], i);
        }

        return cnt;

    }

    public int numberOfWays(int startPos, int endPos, int k) {
        sp = startPos;
        op = k;
        int[][] dp = new int[(startPos + 2 * k) + 1][k + 1];
        for (int[] d : dp) Arrays.fill(d, -1);
        return helper(startPos, endPos, k, dp);
    }


    private int helper(int currPos, int endPos, int k, int[][] dp) {
        // base case
        if (currPos == endPos && k == 0) return 1;

        if (k <= 0) return 0;

        int newPos;
        if (currPos < 0) newPos = (sp + 2 * op) - Math.abs(currPos);
        else newPos = currPos + k;

        // System.out.println(currPos + ":" + newPos);
        if (dp[newPos][k] != -1) return dp[newPos][k];
        int left = 0, right = 0;
        // left
        left += helper(currPos - 1, endPos, k - 1, dp);

        // right
        right += helper(currPos + 1, endPos, k - 1, dp);

        return dp[newPos][k] = (left + right) % mod;
    }

    public int numberOfPaths(int[][] grid, int k) {
        m = grid.length;
        n = grid[0].length;

        int[][][] dp = new int[m][n][k];

        for (int[][] f : dp) {
            for (int[] d : f) Arrays.fill(d, -1);
        }

        return helper(grid, k, m - 1, n - 1, 0, dp);
    }

    private int helper(int[][] grid, int k, int m, int n, int sum, int[][][] dp) {
        // base case
        if (n == 0 && m == 0) {
            if ((sum + grid[0][0]) % k == 0) return dp[m][n][sum] = 1;
            return dp[m][n][sum] = 0;
        }

        if (n < 0 || m < 0) {
            return 0;
        }

        if (dp[m][n][sum] != -1) return dp[m][n][sum] % MOD;

        int up = helper(grid, k, m - 1, n, (sum + grid[m][n]) % k, dp) % MOD;
        int down = helper(grid, k, m, n - 1, (sum + grid[m][n]) % k, dp) % MOD;

        return dp[m][n][sum] = (up + down) % MOD;
    }

    // Intial thoughts:-
    // BFS algorithm


    //  DP
    //  - create a map to adjancent nodes alogn with distance
    //  - iterate through all nodes and calculate cost of each path recursively
    //  - update minCost path if currNode reaches to destination
    //  - returm minCost;
    // TC  = (2^n)
    // SC = O(n)

    private boolean isSafe(int[][] grid, int i, int j) {
        int m = grid.length;
        int n = grid[0].length;
        return i >= 0 && i < m && j >= 0 && j < n;
    }

    // Thoughts:
	/*

	   TC = O(2^n), Sc = O(n)

	   Algorithm:-
	  - The idea is to split array in two parts such that
	     avg(A) = avg(B)
	  - Iterate through array elements and for each elem
	     check if we can split it in two parts with equals avg

	  -  We have choice of take or dont take in first part
	     ie. if arr(i) is taken in part1 sumA+arr(i)
	     else sumB + arr(i)

	  - Do above step recursilvely and backtrack
	  - check if sumA == sumB && (index == n-1) { that means all elements have been segregated into two parts successfuly
	  }
	     - if true return true
	      else return false and recurse further
	  - Add Memoization to improve exponential time complexity

	  total  = sumA + sumB
	  sumB = total - sumA

	  A+B=n
	  B=n-A

	  sumA/A = sumB/B

	  sumA/A = total-sumA/B
	  sumA/A = total-sumA/n-A
	  n*sumA/A  = total
	  sumA = total * lenA / n

	  problem boils down to finding a subsequence of length len1
	  with sum equals sumA
	*/
    public boolean splitArraySameAverage(int[] nums) {
        int n = nums.length, total = 0;
        for (int i = 0; i < n; i++) total += nums[i];

        HashMap<String, Boolean> map = new HashMap<>();
        for (int cnt = 1; cnt < n; cnt++) {
            if ((total * cnt) % n == 0) {
                if (isPossible(nums, 0, cnt, (total * cnt) / n, map)) return true;
            }
        }
        return false;
    }

    private boolean isPossible(int[] nums, int ind, int len, int sum, HashMap<String, Boolean> map) {
        int n = nums.length;
        // base case
        if (sum == 0 && len == 0) {
            return true;
        }

        if (ind >= n || len == 0) return false;
        String key = len + "-" + sum + "-" + ind;
        if (map.containsKey(key)) return map.get(key);
        // if number can be taken
        if (sum - nums[ind] >= 0) {
            // taken
            boolean case1 = isPossible(nums, ind + 1, len - 1, sum - nums[ind], map);

            // not taken
            boolean case2 = isPossible(nums, ind + 1, len, sum, map);

            map.put(key, (case1 || case2));
            return case1 || case2;
        }

        // Can't be taken
        boolean case2 = isPossible(nums, ind + 1, len, sum, map);

        map.put(key, case2);
        return case2;
    }

    // Min. no. of steps to  make both strings equal
    // Input: word1 = "sea", word2 = "eat"
    // Output: 2
    // TC = O(n1*n2), SC =  O(n1*n2)
    public int minDistance(String word1, String word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        int[][] dp = new int[n1 + 1][n2 + 1]; // To get no of steps after removing a character from either strings

        for (int i = 0; i <= n1; i++) {
            for (int j = 0; j <= n2; j++) {
                // if no character is removed from word1 then steps = no. of characters of word2
                if (i == 0) dp[i][j] = j;
                    // if no character is removed from word2 then steps = no. of character of word1
                else if (j == 0) dp[i][j] = i;
                else {
                    // if prev chartacters are same then steps = prev steps
                    if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                        dp[i][j] = dp[i - 1][j - 1];
                    }

                    // Else take min of both cases
                    else {
                        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1]);
                    }
                }
            }
        }
        return dp[n1][n2];
    }

    int find(int src, int dest, int k) {
        if (k < 0) return 1000_000_00;
        if (src == dest) return 0;
        if (cpDP[src][k] != null) return cpDP[src][k];
        int max = 1000_000_00;
        for (int[] arr : map.getOrDefault(src, new ArrayList<>())) {
            max = Math.min(max, arr[1] + find(arr[0], dest, k - 1));
        }
        return cpDP[src][k] = max;
    }

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int K) {
        map = new HashMap<>();
        cpDP = new Integer[n + 1][K + 2];
        for (int[] a : flights) {
            map.computeIfAbsent(a[0], k -> new ArrayList<>());
            map.get(a[0]).add(new int[]{a[1], a[2]});
        }
        int temp = find(src, dst, K + 1);
        return temp >= 1000_000_00 ? -1 : temp;
    }

    public int componentValue(int[] nums, int[][] edges) {

        Map<Integer, List<Integer>> graph = new HashMap<>();

        for (int[] edge : edges) {
            if (!graph.containsKey(edge[0])) graph.put(edge[0], new ArrayList<>());
            if (!graph.containsKey(edge[1])) graph.put(edge[1], new ArrayList<>());
            graph.get(edge[0]).add(edge[1]);
            graph.get(edge[1]).add(edge[0]);
        }
        int max = -1;
        for (int i = 0; i < nums.length; i++) {
            max = Math.max(max, helper(i, nums, graph, 0, 0));
        }

        return max;
    }

    //Knapsack based problem

    private int helper(int start, int[] nums, Map<Integer, List<Integer>> graph, int currentSum, int expectedSum) {
        // base case
        if (currentSum == expectedSum) return 1;
        if (start >= nums.length) return 0;

        int sum = nums[start];
        int cw = 0, ncw = 0;
        for (int node : graph.get(start)) {
            // make a cut
            cw += 1 + helper(node, nums, graph, 0, nums[start]);
            // Don't make a cut
            ncw += helper(node, nums, graph, sum + nums[node], expectedSum);
        }

        return Math.max(cw, ncw);
    }

    /*
   Input: n = 7
   Output: 9
   Explanation: We can at most get 9 A's on screen by pressing following key sequence:
   A, A, A, Ctrl A, Ctrl C, Ctrl V, Ctrl V
 */
    public int maxA(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, -1);
        return rec(n, dp);
    }

    private int rec(int n, int[] dp) {
        // base case
        if (n < 3) return Math.max(0, n);

        if (dp[n] != -1) return dp[n];
        int max = 0;
        max = Math.max(max, 1 + rec(n - 1, dp)); // 'A'
        max = Math.max(max, 2 * rec(n - 3, dp)); // ctrl+a -> ctrl+c -> ctrl+v
        max = Math.max(max, 3 * rec(n - 4, dp)); // ctrl+a -> ctrl+c -> ctrl+v -> ctrl+v
        max = Math.max(max, 4 * rec(n - 5, dp)); // ctrl+a -> ctrl+c -> ctrl+v -> ctrl+v -> ctrl+v

        return dp[n] = max;
    }


    /*
    Input: robot = [0,4,6], factory = [[2,2],[6,2]]
    Output: 4
    Explanation: As shown in the figure:
    - The first robot at position 0 moves in the positive direction. It will be repaired at the first factory.
    - The second robot at position 4 moves in the negative direction. It will be repaired at the first factory.
    - The third robot at position 6 will be repaired at the second factory. It does not need to move.
    The limit of the first factory is 2, and it fixed 2 robots.
    The limit of the second factory is 2, and it fixed 1 robot.
    The total distance is |2 - 0| + |2 - 4| + |6 - 6| = 4. It can be shown that we cannot achieve a better total distance than 4.

    ALGO- (Knapsack Type)
        Sort the robots postions.
        Sort the factory positions.
        Iterate from left robot to right. For each robot, you have 2 options:
        Choose to fix it in current factory if possible (If the factory hasn't reached its limit) (like take item from the current sack)
        or Try to fix the robot in the next factory (or don't take from the current sack)
        Answer = max(option1, option2)
        Time Complexity: O(n * n * n)
        Space Complexity: O(n * n * n)
     */
    public long minimumTotalDistance(List<Integer> robot, int[][] factory) {
        this.robot = robot;
        this.factory = factory;
        n = factory.length;
        Collections.sort(robot);
        Arrays.sort(factory, Comparator.comparingInt(a -> a[0]));
        long[][][] dp = new long[111][111][111];

        for (long[][] d1 : dp) for (long[] d2 : d1) Arrays.fill(d2, -1L);

        return helper(0, 0, factory[0][1], dp);
    }

    private long helper(int robot_index, int factory_index, int capacity, long[][][] dp) {
        // base case
        if (robot_index == this.robot.size()) return 0L;
        // Still there are robots to repair but no factories remained
        if (factory_index == this.factory.length) return Long.MAX_VALUE;

        if (dp[robot_index][factory_index][capacity] != -1L) return dp[robot_index][factory_index][capacity];

        // Means robot cant be fixed at current factory, let's repair it at next
        if (capacity == 0)
            return dp[robot_index][factory_index][capacity] = helper(robot_index, factory_index + 1, (factory_index + 1) == n ? 0 : factory[factory_index + 1][1], dp);

        // If robot can be fixed at current factory
        // then 2 cases ->
        // will fix at current OR will fix at next

        long subAns1 = helper(robot_index + 1, factory_index, capacity - 1, dp);
        if (subAns1 != Long.MAX_VALUE) subAns1 += Math.abs(this.robot.get(robot_index) - factory[factory_index][0]);
        long subAns2 = helper(robot_index, factory_index + 1, (factory_index + 1) == n ? 0 : factory[factory_index + 1][1], dp);
        return dp[robot_index][factory_index][capacity] = Math.min(subAns2, subAns1);
    }


    // TC = O(10*10*N)
    /*
       To create a 5 digit palindrome we do not need to care about the middle element.
       We just need to find subsequence of pattern XY_YX.
       Calculate number of subsequences of type XY and subsequences of type YX around any given point i and multiply them to find number of subsequences of type XY_YX.
       Since string only has digits, the time complexity will be 100*n.

    Approach -
    We will be maintaing the counts of digit in the list cnts
    Keep 2 arrays pre and suf to store the number of prefixes of type XY and suffixes of type YX. pre[i-1][1][2] means prefixes of type 12 before index i.
    Similarly suf[i+1][1][2] means suffixes of type 21 after index i
    Remember given string is made of digits that is 0123456789.
    That's a total of 10 unique characters
    Once we have calculated the prefix and suffix lists we just need to multiply pre[i - 1][j][k] with suf[i + 1][j][k] to find number of palindromic subsequences
     */
    public int countPalindromes(String s) {
        int n = s.length(), ans = 0;
        int[][][] prefix = new int[n][10][10], suffix = new int[n][10][10];

        int[] cnts = new int[10];
        for (int i = 0; i < n; i++) {
            int c = s.charAt(i) - '0';
            if (i != 0) {
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < 10; k++) {
                        prefix[i][j][k] = prefix[i - 1][j][k];
                        if (k == c) prefix[i][j][k] += cnts[j];
                    }
                }
            }
            cnts[c]++;
        }


        Arrays.fill(cnts, 0);
        for (int i = n - 1; i >= 0; i--) {
            int c = s.charAt(i) - '0';
            if (i != n - 1) {
                for (int j = 0; j < 10; j++) {
                    for (int k = 0; k < 10; k++) {
                        suffix[i][j][k] = suffix[i + 1][j][k];
                        if (k == c) suffix[i][j][k] += cnts[j];
                    }
                }
            }
            cnts[c]++;
        }


        for (int i = 1; i < n - 1; i++) {
            for (int j = 0; j < 10; j++) {
                for (int k = 0; k < 10; k++) {
                    ans = (int) ((ans + (long) prefix[i - 1][j][k] * suffix[i + 1][j][k]) % mod);
                }
            }
        }

        return ans;
    }

    public int countGoodStrings(int low, int high, int zero, int one) {
        String z = str('0', zero);
        String o = str('1', one);
        Map<Integer, Integer> dp = new HashMap<>();
        return rec(low, high, z, o, new StringBuilder(), dp);
    }

    private int rec(int low, int high, String zero, String one, StringBuilder sb, Map<Integer, Integer> dp) {
        // base case
        if (sb.length() > high) return 0;
        if (dp.containsKey(sb.length())) return dp.get(sb.length());
        int tz = 0, to = 0;
        // take zero
        sb.append(zero);
        tz += ((valid(sb, low, high) ? 1 : 0) + rec(low, high, zero, one, sb, dp)) % mod;

        // backtrack;
        sb.delete(sb.length() - zero.length(), sb.length());

        // take one
        sb.append(one);
        to += ((valid(sb, low, high) ? 1 : 0) + rec(low, high, zero, one, sb, dp)) % mod;
        // backtrack
        sb.delete(sb.length() - one.length(), sb.length());

        int ans = (tz + to) % mod;
        dp.put(sb.length(), ans);
        return ans;

    }

    private String str(char c, int times) {
        char[] repeat = new char[times];
        Arrays.fill(repeat, c);
        return new String(repeat);
    }

    private boolean valid(StringBuilder sb, int low, int high) {
        return sb.length() >= low && sb.length() <= high;
    }


    //Optimal solution
    public int maxPalindromesOptimal(String s, int k) {
        int ans = 0, n = s.length();
        int[] dp = new int[n + 1];
        for (int i = k - 1; i < n; i++) {
            dp[i + 1] = dp[i];
            if (helper(s, i - k + 1, i)) dp[i + 1] = Math.max(dp[i + 1], 1 + dp[i - k + 1]);
            if (i - k >= 0 && helper(s, i - k, i)) dp[i + 1] = Math.max(dp[i + 1], 1 + dp[i - k]);
        }
        return dp[n];
    }

    boolean helper(String s, int l, int r) {
        while (l < r) {
            if (s.charAt(l) != s.charAt(r)) return false;
            l++;
            r--;
        }
        return true;
    }


    // My Solution
    public int maxPalindromes(String s, int k) {
        int n = s.length();
        int[][] dp = new int[n][n + 1];
        for (int[] d : dp) Arrays.fill(d, -1);


        boolean[][] palind = new boolean[n][n + 1];
        boolean[][] memob = new boolean[n][n + 1];

        // precompute palindrome from indexes {i, j}
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j <= n; j++) {
                palind[i][j - 1] = isPalindrome(s, i, j - 1, memob);
            }
        }

        return rec(s, k, 0, 1, palind, dp);
    }


    private int rec(String s, int k, int i, int j, boolean[][] palind, int[][] dp) {
        int n = s.length();
        int ind = i;
        int left, right;
        if (i > j || i >= n || j >= n + 1) return 0;
        if (dp[i][j] != -1) return dp[i][j];


        while (ind < n && i < j) {
            //To check the palindrome of odd length palindromic sub-string
            if (palind[ind][ind] && (m - ind) >= k) {
                // take the current one
                left = 1 + rec(s, k, m, m + 1, palind, dp);
                // skip the current one
                right = rec(s, k, ind, m + 1, palind, dp);
                return dp[i][j] = Math.max(left, right);
            }

            //To check the palindrome of even length palindromic sub-string
            if (palind[ind][ind + 1] && (m - ind + 1) >= k) {
                // take the current one
                left = 1 + rec(s, k, m, m + 1, palind, dp);
                // skip the current one
                right = rec(s, k, ind, m + 1, palind, dp);
                return dp[i][j] = Math.max(left, right);
            }
            ind += 1;
        }

        return 0;
    }


    private boolean isPalindrome(String s, int i, int j, boolean[][] memob) {
        if (i == j || i > j) return true;
        if (memob[i][j]) return memob[i][j];

        char ch1 = s.charAt(i);
        char ch2 = s.charAt(j);

        if (ch1 == ch2) {
            memob[i][j] = isPalindrome(s, i + 1, j - 1, memob);
        } else return false;
        return memob[i][j];
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

    public int waysToReachTarget(int target, int[][] types) {
        int[][] dp = new int[target + 1][types.length + 1];
        for (int[] d : dp) Arrays.fill(d, -1);
        return helper(target, types, 0, dp);
    }

    private int helper(int target, int[][] types, int ind, int[][] dp) {
        // base case
        if (target == 0) return 1;
        if (ind >= types.length) return 0;

        int ans = 0;
        int cnt = types[ind][0];
        int value = types[ind][1];

        if (dp[target][ind] != -1) return dp[target][ind];
        for (int i = 0; i <= cnt; i++) {
            if (target - value * i >= 0) {
                // take
                ans = (ans + helper(target - value * i, types, ind + 1, dp)) % mod;
            }
        }
        return dp[target][ind] = ans % mod;
    }

    //DP
    public long countQuadruplets(int[] nums) {
        Map<String, Long> dp = new HashMap<>();
        return helper(nums, 0, new ArrayList<>(), dp);
    }

    private long helper(int[] nums, int ind, List<Integer> selected, Map<String, Long> dp) {

        String key = String.valueOf(ind);
        for (int s : selected) key += "-" + s;

        if (selected.size() == 4) {
            return 1;
        }

        if (ind >= nums.length || selected.size() > 4) return 0;

        if (dp.containsKey(key)) return dp.get(key);

        long cnt = 0L;
        // take
        if (selected.size() == 0
                || (selected.size() == 1 && selected.get(0) + 1 < nums[ind])
                || (selected.size() == 2 && selected.get(0) < nums[ind] && selected.get(1) > nums[ind])
                || (selected.size() == 3
                && selected.get(0) < nums[ind]
                && selected.get(1) < nums[ind]
                && selected.get(2) < nums[ind])
        ) {
            selected.add(nums[ind]);
            cnt += helper(nums, ind + 1, selected, dp);
            // bactrack
            selected.remove(selected.size() - 1); // deleted last added element
        }

        //not-take
        cnt += helper(nums, ind + 1, selected, dp);
        dp.put(key, cnt);
        return cnt;
    }

    /*
    You are given a 0-indexed string s and a dictionary of words dictionary.
    You have to break s into one or more non-overlapping substrings such that each substring is present in dictionary.
    There may be some extra characters in s which are not present in any of the substrings.

    Return the minimum number of extra characters left over if you break up s optimally.

    Input: s = "leetscode", dictionary = ["leet","code","leetcode"]
    Output: 1
    Explanation: We can break s in two substrings: "leet" from index 0 to 3 and "code" from index 5 to 8.
     There is only 1 unused character (at index 4), so we return 1.
     */
    public int func(int idx, String s, Set<String> st, int[] dp) {
        if (idx == s.length())
            return 0;
        if (dp[idx] != -1)
            return dp[idx];
        int res = Integer.MAX_VALUE;
        for (int j = idx; j < s.length(); ++j) {
            String str = s.substring(idx, j + 1);
            if (st.contains(str))
                res = Math.min(res, func(j + 1, s, st, dp));
            else
                res = Math.min(res, j - idx + 1 + func(j + 1, s, st, dp));
        }
        return dp[idx] = res;
    }

    public int minExtraChar(String s, String[] dictionary) {
        int[] dp = new int[s.length() + 1];
        Arrays.fill(dp, -1);
        Set<String> st = new HashSet<>(Arrays.asList(dictionary));
        return func(0, s, st, dp);
    }

    public int maximumJumps(int[] nums, int target) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, Integer.MIN_VALUE);
        return mj(nums, target, 0, nums[0], dp);
    }

    /*
    Input: nums = [1,3,6,4,1,2], target = 2
    Output: 3
    Explanation: To go from index 0 to index n - 1 with the maximum number of jumps, you can perform the following jumping sequence:
    - Jump from index 0 to index 1.
    - Jump from index 1 to index 3.
    - Jump from index 3 to index 5.
    It can be proven that there is no other jumping sequence that goes from 0 to n - 1 with more than 3 jumps. Hence, the answer is 3.
     */

    private int mj(int[] nums, int target, int ind, int last, int[] dp) {

        // base case
        if (ind == nums.length - 1) {
            return 0;
        }

        if (dp[ind] != Integer.MIN_VALUE) return dp[ind];

        int res = -1;

        for (int i = ind + 1; i < nums.length; i++) {

            if (Math.abs(nums[i] - last) <= target) {

                int solve = mj(nums, target, i, nums[i], dp);
                if (solve != -1) {
                    res = Math.max(res, 1 + solve);
                }
            }
        }

        return dp[ind] = res;
    }

    /*
    Input: nums1 = [2,3,1], nums2 = [1,2,1]
    Output: 2
    Explanation: One way to construct nums3 is:
    nums3 = [nums1[0], nums2[1], nums2[2]] => [2,2,1].
    The subarray starting from index 0 and ending at index 1, [2,2], forms a non-decreasing subarray of length 2.
    We can show that 2 is the maximum achievable length.

     */
    //TC = O(3N), SC = O(3n)
    public int maxNonDecreasingLength(int[] nums1, int[] nums2) {

        int[][] dp = new int[nums1.length][3];
        for (int[] d : dp) Arrays.fill(d, -1);
        return helper(0, nums1, nums2, 0, dp);
    }

    private int helper(int ind, int[] nums1, int[] nums2, int choice, int[][] dp) {
        // base case
        if (ind >= nums1.length) return 0;

        int maxLen = 0;

        if (dp[ind][choice] != -1) return dp[ind][choice];


        //take
        if (choice == 0) {
            // not-take current guy
            maxLen = Math.max(maxLen, helper(ind + 1, nums1, nums2, 0, dp));
        }

        int prev = choice == 0 ? -1 : choice == 1 ? nums1[ind - 1] : nums2[ind - 1];

        if (nums1[ind] >= prev) {
            maxLen = Math.max(maxLen, 1 + helper(ind + 1, nums1, nums2, 1, dp));
        }
        if (nums2[ind] >= prev) {
            maxLen = Math.max(maxLen, 1 + helper(ind + 1, nums1, nums2, 2, dp));
        }


        return dp[ind][choice] = maxLen;
    }

    public long maxScore(int[] nums, int x) {
        long[][] dp = new long[nums.length][2];
        for (long[] d : dp) Arrays.fill(d, -1L);

        return nums[0] + helper(nums, x, 0, nums[0] % 2, dp);
    }

    private long helper(int[] nums, int x, int ind, int parity, long[][] dp) {

        // base case
        if (ind >= nums.length - 1) return 0;
        if (dp[ind][parity] != -1) return dp[ind][parity];
        long max = 0;
        int i = ind + 1;
        //take
        if (parity == nums[i] % 2) {
            max = Math.max(max, nums[i] + helper(nums, x, i, parity, dp));
        } else {
            max = Math.max(max, nums[i] + helper(nums, x, i, nums[i] % 2, dp) - x);
        }

        //not-take
        max = Math.max(max, helper(nums, x, i, parity, dp));
        return dp[ind][parity] = max;
    }

    public int rob(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, -1);
        return Math.max(nums[0] + helper(nums, 0, dp),
                nums.length > 1 ? nums[1] + helper(nums, 1, dp) : 0);
    }

    private int helper(int[] nums, int ind, int[] dp) {

        // base case
        if (ind >= nums.length - 2) return 0;
        if (dp[ind] != -1) return dp[ind];
        int max = 0;
        int i = ind + 2;
        //take
        max = Math.max(max, nums[i] + helper(nums, i, dp));

        //not-take
        max = Math.max(max, helper(nums, i - 1, dp));
        return dp[ind] = max;
    }

    /*
    Input: nums = [2, 2, 1], m = 4
    Output: true
    Explanation: We can split the array into [2, 2] and [1] in the first step. Then, in the second step, we can split [2, 2] into [2] and [2]. As a result, the answer is true.
     */
    public boolean canSplitArray(List<Integer> nums, int m) {
        int n = nums.size();
        int[] ps = new int[n];
        if (n == 1 || n == 2) return true;
        for (int i = 0; i < n; i++) ps[i] = i > 0 ? (ps[i - 1] + nums.get(i)) : nums.get(i);
        int[][] dp = new int[n][n];
        for (int[] d : dp) Arrays.fill(d, -1);
        return f(0, n - 1, m, ps, dp);
    }

    private boolean f(int i, int j, int m, int[] ps, int[][] dp) {
        // base case
        if (i == j) return true;

        if (dp[i][j] != -1) return dp[i][j] == 1;
        boolean left = false, right = false;
        for (int ind = i; ind < j; ind++) {
            if ((ind == i || (ps[ind] - (i - 1 >= 0 ? ps[i - 1] : 0) >= m)) && (ind == j - 1 || (ps[j] - ps[ind] >= m))) {
                // System.out.println(i + ":" + j + ":" + ind);
                left = f(i, ind, m, ps, dp);
                right = f(ind + 1, j, m, ps, dp);
                boolean res = left && right;
                dp[i][j] = res ? 1 : 0;
                if (res) return true;
            }
        }

        dp[i][j] = 0;
        return false;
    }

    public int numberWaysTopDown(List<List<Integer>> hats) {
        int n = hats.size();
        int[] dp = new int[1 << n]; // {mask of selected person, ways}
        dp[0] = 1; // 1 way to not select any person for hat

        //create adj matrix for cap to people distribution
        List<Integer>[] hattToP = new List[41];
        for (int i = 1; i <= 40; i++) hattToP[i] = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            for (int h : hats.get(i)) {
                hattToP[h].add(i);
            }
        }
        int allMask = (1 << n) - 1;

        for (int hat = 40; hat > 0; hat--) {
            for (int mask = (1 << n) - 1; mask >= 0; mask--) {
                int ways = dp[mask]; // skip the hat
                for (Integer p : hattToP[hat]) {
                    // if person already assigned a hat the skip
                    if ((mask & (1 << p)) == 0) {
                        ways += dp[mask | (1 << p)];
                        ways %= mod;
                        dp[mask | (1 << p)] = ways;
                    }
                }
            }
        }

        return dp[allMask];
    }


    // take, nottake classical problem
    // The problem asks us to find largest subsequence that have sum equal to target
    // Algo:-
    // We need to iterate over list and use take, not-take based recursive approach and then memomise
    public int lengthOfLongestSubsequence(List<Integer> nums, int target) {
        Collections.sort(nums);
        int[][] dp = new int[target + 1][nums.size() + 1];
        for (int[] d : dp) Arrays.fill(d, -1);
        int ans = helper(nums, target, 0, dp);
        return Math.max(ans, -1);
    }

    private int helper(List<Integer> nums, int target, int ind, int[][] dp) {
        // base case
        if (target == 0) return 0;
        if (target < 0 || ind >= nums.size()) return -1001;

        if (dp[target][ind] != -1) return dp[target][ind];
        int take = 1 + helper(nums, target - nums.get(ind), ind + 1, dp);
        int nottake = helper(nums, target, ind + 1, dp);
        return dp[target][ind] = Math.max(take, nottake);
    }


    public int numberWaysRecursive(List<List<Integer>> hats) {
        int n = hats.size();
        Integer[][] dp = new Integer[41][1 << 10]; // {Pair(cap, mask of selected person), ways}
        //create adj matrix for cap to people distribution
        List<Integer>[] hattToP = new List[41];
        for (int i = 1; i <= 40; i++) hattToP[i] = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            for (int h : hats.get(i)) {
                hattToP[h].add(i);
            }
        }
        return dfs((1 << n) - 1, 0, 1, hattToP, dp); // start with 1st hat and traverse through all
    }

    private int dfs(int allMask,
                    int assignedPeople,
                    int hat,
                    List<Integer>[] hattToP,
                    Integer[][] dp) {

        if (assignedPeople == allMask) return 1;
        if (hat > 40) return 0; // person can't wear hat > 40 number

        if (dp[hat][assignedPeople] != null) return dp[hat][assignedPeople];
        int ways = dfs(allMask, assignedPeople, hat + 1, hattToP, dp); // skip the hat

        for (Integer p : hattToP[hat]) {
            // if person already assigned a hat the skip
            if ((assignedPeople & (1 << p)) == 0) {
                ways += dfs(allMask, assignedPeople | (1 << p), hat + 1, hattToP, dp);
                ways %= mod;
            }
        }
        return dp[hat][assignedPeople] = ways;

    }


    /*
    Input: nums = [2, 2, 1], m = 4
    Output: true
    Explanation: We can split the array into [2, 2] and [1] in the first step. Then, in the second step, we can split [2, 2] into [2] and [2]. As a result, the answer is true.
     */
    public boolean canSplitArrayTopDownDP(List<Integer> nums, int m) {
        int n = nums.size();
        int[] ps = new int[n];
        if (n == 1 || n == 2) return true;
        for (int i = 0; i < n; i++) ps[i] = i > 0 ? (ps[i - 1] + nums.get(i)) : nums.get(i);
        boolean[][] dp = new boolean[n][n];
        return f(m, ps, dp);
    }

    private boolean f(int m, int[] ps, boolean[][] dp) {
        int n = ps.length;
        boolean left = false, right = false;

        for (int i = 0; i < n; i++) {
            for (int j = n - 1; j >= 0; j--) {
                for (int ind = i; ind < j; ind++) {
                    if ((ind == i || (ps[ind] - (i - 1 >= 0 ? ps[i - 1] : 0) >= m)) && (ind == j - 1 || (ps[j] - ps[ind] >= m))) {
                        left = dp[i][ind];
                        right = dp[ind + 1][j];
                        boolean res = left && right;
                        dp[i][j] = res;
                        if (res) return true;
                    }
                }
            }
        }
        return false;
    }

    /*
    Traverse through list and set any values {1,2,3} whenever condition breaks.
    Condition :- nums.get(i) > nums.get(i+1) || nums.get(i) < nums.get(i-1);
    TC = O(N2*3), SC = O(N2)
    As N ~=100 that should be fine
    */
    public int minimumOperations(List<Integer> nums) {
        int op = 0;
        List<Integer> clone = new ArrayList<>();
        clone.addAll(nums);
        int benchmark = -1;

        if (nums.size() == 1) return op;
        for (int i = 0; i < nums.size(); i++) {

            if (i == 0) {
                if (clone.get(i + 1) < clone.get(i)) {
                    clone.set(i, clone.get(i + 1));
                    benchmark = clone.get(i + 1);
                    op++;
                }
            } else {
                if (clone.get(i) < benchmark) {
                    clone.set(i, benchmark);
                    op++;
                }
            }

            benchmark = clone.get(i);
        }

        return op;
    }

    // DP + BS
    // Upsolve
    public int maximizeTheProfit(int n, List<List<Integer>> offers) {
        Collections.sort(offers, new Comparator<List<Integer>>() {
            @Override
            public int compare(List<Integer> o1, List<Integer> o2) {
                return o1.get(0).compareTo(o2.get(0));
            }
        });
        System.out.println(Arrays.deepToString(offers.toArray()));
        return helper(0, -1, n, offers);
    }

    private int helper(int ind, int li, int n, List<List<Integer>> offers) {
        if (ind >= offers.size()) return 0;

        int max = 0;
        for (int i = ind; i < offers.size(); i++) {

            int o = 0, no = 0;
            List<Integer> offer = offers.get(i);
            // No overlap
            if (li == -1 || li < offer.get(0)) {
                no += offer.get(2) + helper(ind + 1, offer.get(1), n, offers);
            }
            //  They overlap
            else {
                int c = 0, p = 0;
                // take the current one
                int pv = offers.get(i - 1).get(2);
                c += offer.get(2) + helper(ind + 1, offer.get(1), n, offers);

                // take the previous
                p += pv + helper(ind + 1, li, n, offers);
                o += Math.max(c, p);
            }

            max = Math.max(o, no);
        }

        System.out.println("li=" + li + ", max=" + max);
        return max;
    }


    //TODO: Cleanup solution & solve for correct answer
    public long minIncrementOperations(int[] nums, int k) {

        if (nums.length == 3) {

            int closest = Integer.MAX_VALUE;
            for (int num : nums) {
                if (num >= k) {
                    return 0;
                } else {
                    closest = Math.min(closest, k - num);
                }
            }

            return closest == Integer.MAX_VALUE ? 0 : closest;
        }


        long ans = Math.min(helper(nums, k, nums.length - 1, new HashSet<>())
                , Math.min(helper(nums, k, nums.length - 2, new HashSet<>()),
                        helper(nums, k, nums.length - 3, new HashSet<>())));


        return ans == Long.MAX_VALUE ? 0 : ans;

    }

    private long helper(int[] nums, int k, int ind, HashSet<Integer> changedIndex) {
        // base case
        if (ind < 0) return 0;

        long take = 0, notake = 0;

        if (nums[ind] < k) {
            //change at index
            take += k - nums[ind];
            nums[ind] = k;
            changedIndex.add(ind);
            take += helper(nums, k, ind - 3, changedIndex);
        }

        // not take
        if (changedIndex.contains(ind + 1) || changedIndex.contains(ind + 2)) {
            // can skip to take only till 2 times consecutive
            notake += helper(nums, k, ind - 1, changedIndex);
        }

        if (take == 0) take = Long.MAX_VALUE;
        if (notake == 0) notake = Long.MAX_VALUE;

        System.out.println(take + ":" + notake);
        return Math.min(Math.max(take, 0), Math.max(notake, 0));
    }

    //TODO:TBD
    public int minOperations(int[] nums1, int[] nums2) {
        if (Arrays.stream(nums1).max().getAsInt() == nums1[nums1.length - 1]
                && Arrays.stream(nums2).max().getAsInt() == nums2[nums2.length - 1]
        ) return 0;

        TreeMap<Integer, Integer> freq1 = new TreeMap<>(Collections.reverseOrder());
        TreeMap<Integer, Integer> freq2 = new TreeMap<>(Collections.reverseOrder());

        for (int num : nums1) freq1.put(num, freq1.getOrDefault(num, 0) + 1);
        for (int num : nums2) freq2.put(num, freq2.getOrDefault(num, 0) + 1);

        if (freq1.firstEntry().getValue() == 1 && freq2.firstEntry().getValue() == 1) {
            // check if max of both arrays are at same index then  return -1
            int ind = -1;
            for (int i = 0; i < nums1.length; i++) {
                if (nums1[i] == freq1.firstKey()) {
                    ind = i;
                    break;
                }
            }

            for (int i = 0; i < nums2.length; i++) {
                if (nums2[i] == freq2.firstKey() && ind == i) {
                    return -1;
                }
            }
        }


        return helper(nums1, nums2, 0);
    }

    private int helper(int[] nums1, int[] nums2, int ind) {
        // base case
        if (ind >= nums1.length) {
            if (Arrays.stream(nums1).max().getAsInt() == nums1[nums1.length - 1]
                    && Arrays.stream(nums2).max().getAsInt() == nums2[nums2.length - 1]
            ) return 0;
            return 999999;
        }

        int take = 0, nottake = 0;
        for (int i = ind; i < nums1.length; i++) {
            // take

            //swap
            int temp = nums1[i];
            nums1[i] = nums2[i];
            nums2[i] = temp;
            take = 1 + helper(nums1, nums2, ind + 1);

            //backtrack
            int backtemp = nums2[i];
            nums2[i] = nums1[i];
            nums1[i] = backtemp;

            // not take
            nottake = helper(nums1, nums2, ind + 1);
        }

        return Math.min(take, nottake);

    }

    //TLE
    // O(N3) valid when N ~ 100
    public int waysToSplit(int[] nums) {
        int[] prefix = new int[nums.length];
        for (int i = 0; i < nums.length; i++)
            prefix[i] += nums[i] + (i > 0 ? prefix[i - 1] : 0);
        Map<String, Integer> dp = new HashMap<>();
        return ways(nums, prefix, new ArrayList<>(), 0, dp);
    }

    private int ways(int[] nums, int[] prefix, List<Integer> indexes, int i, Map<String, Integer> dp) {
        // base case
        if (i >= nums.length) return 0;
        String key = Arrays.toString(indexes.toArray()) + i;
        if (dp.containsKey(key)) return dp.get(key);
        int ways = 0;
        // split
        if (indexes.isEmpty()) {
            indexes.add(i);
            ways = (ways + ways(nums, prefix, indexes, i + 1, dp)) % mod;
            indexes.remove(indexes.size() - 1); //backtrack
        }

        if (indexes.size() == 1 && (prefix[i] >= 2 * prefix[indexes.get(0)])) {
            indexes.add(i);
            ways = (ways + ways(nums, prefix, indexes, i + 1, dp)) % mod;
            indexes.remove(indexes.size() - 1); //backtrack
        }

        if (indexes.size() == 2 && i == nums.length - 1) {
            int li = indexes.get(indexes.size() - 1);
            int currSum = prefix[i] - prefix[li];
            int lastSum = prefix[li] - prefix[indexes.get(0)];
            System.out.println(currSum + ":" + lastSum);
            if (currSum >= lastSum) return 1;
            return 0;
        }
        // not split
        ways = (ways + ways(nums, prefix, indexes, i + 1, dp)) % mod;
        return ways;
    }

}
