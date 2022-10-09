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
        for (int c = 0; c < col; c++) {
            dp[0][c] = grid[0][c];
        }
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
    I has used Map<String, Integer> in place of 3d DP array but it was giving TLE
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
        if (prev == '#' || (int) Math.abs(prev - s.charAt(ind)) <= k) {
            take = 1 + ls(s, k, ind + 1, s.charAt(ind), dp);
        }
        // not take
        nt = ls(s, k, ind + 1, prev, dp);
        return dp[ind][Math.abs((int) prev - 'a')] = Math.max(take, nt);
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

            // even jump starting from index i
            next = tm.floorKey(arr[i]);
            if (next == null) dp[i][1] = 0;
            else if (dp[tm.get(next)][0] > 0) dp[i][1] = 1;


            if (dp[i][0] == 1) cnt++;
            tm.put(arr[i], i);
        }

        return cnt;

    }

    final int mod = 1_000_000_007;
    int sp;
    int op;

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


    //Author : Anand
    int m, n;
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
}
