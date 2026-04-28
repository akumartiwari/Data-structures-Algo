package com.company;

import java.util.HashMap;import java.util.*;
import java.util.stream.Collectors;

class House {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // This [paise class stores if value/node is taken or not as first and its actual value and second param
    // {0,23} -> not taken  with value=23
    static class Pair<I extends Number, I1 extends Number> implements Comparable<Pair<Number, Number>> {
        int first;
        int second;

        Pair(int key, int val) {
            this.first = key;
            this.second = val;
        }

        /*
         * PROBLEM: Compare Pair by first-second (Helper)
         * Compares this Pair to another by computing this.first - other.second.
         *
         * ALGORITHM: Direct arithmetic comparison
         * TC: O(1) | SC: O(1)
         */
        @Override
        public int compareTo(Pair<Number, Number> pair) {
            return this.first - pair.second;
        }
    }


    /*
     * PROBLEM: House Robber III - DFS helper (LeetCode 337)
     * DFS returning a {rob, skip} pair for each node to enable tree DP.
     *
     * ALGORITHM: Post-order DFS (Tree DP)
     * TC: O(n) | SC: O(h)
     */
    public Pair<Number, Number> dfs1(TreeNode root) {
        if (root == null) return new Pair<>(0, 0);
        Pair<Number, Number> left = dfs1(root.left);
        Pair<Number, Number> right = dfs1(root.right);

        return new Pair<Number, Number>((root.val + left.second + right.second), (Math.max(left.first, left.second) + Math.max(right.first, right.second)));
    }

    /*
     * PROBLEM: House Robber III (LeetCode 337)
     * Max money robbing from a binary tree without robbing adjacent (parent-child) nodes.
     *
     * ALGORITHM: Tree DP via post-order DFS
     * TC: O(n) | SC: O(h)
     */
    public int rob(TreeNode root) {
        Pair<Number, Number> ans = dfs1(root);
        return Math.max(ans.first, ans.second);
    }

    /*
     public static int longestCommonSubstrLengthRec(String s1, String s2, int i1, int i2, int count) {
    if (i1 == s1.length() || i2 == s2.length())
      return count;

    if (s1.charAt(i1) == s2.charAt(i2))
      count = longestCommonSubstrLengthRec(s1, s2, i1 + 1, i2 + 1, count + 1);

    int c1 = longestCommonSubstrLengthRec(s1, s2, i1, i2 + 1, 0);
    int c2 = longestCommonSubstrLengthRec(s1, s2, i1 + 1, i2, 0);

    return Math.max(count, Math.max(c1, c2));
  }

  public static int longestCommonSubstrLength(String s1, String s2)
  {
    return longestCommonSubstrLengthRec(s1, s2, 0, 0, 0);
  }

     */

    /*
     * PROBLEM: House Robber II (LeetCode 213)
     * Max money robbing from a circular array where first and last houses are adjacent.
     *
     * ALGORITHM: Two-pass linear DP (once skipping first, once skipping last)
     * TC: O(n) | SC: O(1)
     */
    public int rob(int[] nums) {
        if (nums.length == 1) return nums[0];
        if (nums.length == 0) return 0;
        return Math.max(robRec(nums, 0, nums.length - 2), robRec(nums, 1, nums.length - 1));
    }

    /*
     * PROBLEM: House Robber linear range helper (Helper)
     * Robs houses in a linear sub-range [l, r] of nums using two-variable DP.
     *
     * ALGORITHM: Linear DP with constant space (two variables)
     * TC: O(n) | SC: O(1)
     */
    public int robRec(int[] nums, int l, int r) {
        int prev = 0, prepre = 0, curr = 0;
        for (int i = l; i <= r; i++) {
            curr = Math.max(nums[i] + prepre, prev);
            prepre = prev;
            prev = curr;
        }
        return curr;
    }


    class Solution {
        // Recurison Implementation
       /*
        int[] nums;

        public int rob(int[] nums) {
            if (nums.length == 0) return 0;
            if (nums.length == 1) return nums[0];


            // setting global int[] with current nums
            this.nums = nums;
            return Math.max(recurse(0, nums.length - 1), recurse(1, nums.length));
        }

        public int recurse(int curr, int end) {
            if (curr >= end) return 0;
            // 2 cases":-
            //   - Excluding the current num value ==> curr +=1
            //   - Including the current num ==>curr +=2


            return Math.max(recurse(curr + 1, end), nums[curr] + recurse(curr + 2, end));



//   2. Top-down DP ----> O(n), O(n)

        int[] nums, memo;

        public int rob(int[] nums) {

            int n = nums.length;
            this.nums = nums;
            // base cases
            if (n == 0) return 0;
            if (n == 1) return nums[0];
            memo = new int[n];

            // Reset memo table to get exact values
            Arrays.fill(memo, -1);
            int left = recurse(0, n - 1);

            // Reset again for next step
            Arrays.fill(memo, -1);
            int right = recurse(1, n);

            return Math.max(left, right);
        }

        public int recurse(int curr, int end) {

            if (curr >= end) return 0;
            else if (memo[curr] != -1) return memo[curr];

            Arrays.fill(memo, -1);

            memo[curr] = Math.max(recurse(curr + 1, end), nums[curr] + recurse(curr + 2, end));
            return memo[curr];

        }


    }


        // 3. Bottom-up DP O(n), O(n)
        int[] nums;
        int n;

        public int rob(int[] nums) {
            this.nums = nums;
            n = nums.length;
            // base cases
            if (n == 0) return 0;
            else if (n == 1) return nums[0];
            return Math.max(recurse(0, n - 1), recurse(1, n));
        }

        public int recurse(int start, int end) {
            int[] dp = new int[end];
            dp[start] = nums[start];
            if (end > start + 1) dp[start] = Math.max(dp[start], dp[start + 1]);

            for (int i = start + 2; i < n; i++) {
                // by including current num and by excluding it
                dp[i] = Math.max(nums[i] + dp[i - 1], dp[i - 2]);
            }

            return dp[end - 1];
        }
        */


        // 4. Bottom-up DP with constant space

        int[] nums;
        int n;

        /*
         * PROBLEM: House Robber II (LeetCode 213)
         * Max money robbing from a circular array (bottom-up DP, constant space).
         *
         * ALGORITHM: Bottom-up DP with O(1) space, two passes (skip first or skip last)
         * TC: O(n) | SC: O(1)
         */
        public int rob(int[] nums) {
            this.nums = nums;
            n = nums.length;
            // base cases
            if (n == 0) return 0;
            else if (n == 1) return nums[0];
            return Math.max(recurse(0, n - 1), recurse(1, n));
        }

        /*
         * PROBLEM: House Robber linear range DP helper (Helper)
         * Bottom-up DP for robbing houses in range [start, end) with constant space.
         *
         * ALGORITHM: Bottom-up DP with two variables
         * TC: O(n) | SC: O(1)
         */
        public int recurse(int start, int end) {
            int prev = 0, prepre = 0;
            if (end > start + 1) {
                prev = Math.max(nums[start], nums[start + 1]);
                prepre = nums[start];
            } else prev = nums[start];


            for (int i = start + 2; i < end; i++) {
                // by including current num and by excluding it
                int curr = Math.max(nums[i] + prepre, prev);
                prepre = prev;
                prev = curr;
            }
            return prev;
        }
        /*
        / 0 < x < n and n % x == 0.
// n = n-x
Input: n = 3
Output: true
dpa[0]=0
dpb[0]=0
dpa[1]=1
dpv[1]=1

n == 2
         */

        /*
         * PROBLEM: Divisor Game (LeetCode 1025)
         * Determine if Alice wins; she always wins when n is even.
         *
         * ALGORITHM: DP simulation (alternating turns)
         * TC: O(n) | SC: O(n)
         */
        public boolean divisorGame(int n) {
            if (n == 0) return false;

            int[] dpa = new int[n];
            int[] dpb = new int[n];

            Arrays.fill(dpa, -1);
            Arrays.fill(dpb, -1);
            dpa[0] = 0;
            dpb[0] = 0;

            int k = 0;
            while (n > 1) {
                dpa[k] = 1;
                n -= 1;
                if (n > 1) {
                    dpb[k] = 1;
                    n -= 1;
                }
                k++;
            }

            int a = 0;
            int b = 0;
            for (int i = dpa.length - 1; i > 0; i--) {
                if (dpa[i] != -1) {
                    a = i;
                    break;
                }
            }
            for (int i = dpb.length - 1; i > 0; i--) {
                if (dpb[i] != -1) {
                    b = i;
                    break;
                }
            }

            return a > b;
        }


        /*
         * PROBLEM: Matrix Block Sum (LeetCode 1314)
         * Compute answer matrix where each cell equals the sum of all elements in its k-radius block.
         *
         * ALGORITHM: 2D prefix sum (stub)
         * TC: O(m*n) | SC: O(m*n)
         */
        public int[][] matrixBlockSum(int[][] mat, int k) {
            int rows = mat.length;
            int cols = mat[0].length;
            int[][] ans = new int[rows][cols];

            if (n == 0) return new int[rows][cols];

            //Logic /crux of code
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {


                }
            }


            return ans;
        }

        int[] arr;

        /*
         * PROBLEM: Max Subset Sum (HackerRank)
         * Find the maximum sum of a subset where no two selected elements are adjacent.
         *
         * ALGORITHM: DP on non-adjacent elements
         * TC: O(n) | SC: O(n)
         */
        int maxSubsetSum(int[] arr) {
            this.arr = arr;
            n = arr.length;
            // base cases
            if (n == 0) return 0;
            else if (n == 1) return nums[0];
            return Math.max(maxSum(0, n - 1), maxSum(1, n));

        }

        /*
         * PROBLEM: Max subset sum DP helper (Helper)
         * DP helper computing max non-adjacent sum for array range [start, end).
         *
         * ALGORITHM: Linear DP with dp array
         * TC: O(n) | SC: O(n)
         */
        public int maxSum(int start, int end) {
            int[] dp = new int[end];
            dp[start] = nums[start];
            if (end > start + 1) dp[start] = Math.max(dp[start], dp[start + 1]);

            for (int i = start + 2; i < n; i++) {
                // by including current num and by excluding it
                dp[i] = Math.max(nums[i] + dp[i - 1], dp[i - 2]);
            }

            return dp[end - 1];
        }


        /*
         * PROBLEM: Max Subset Sum space-optimized (Helper)
         * Space-optimized in-place DP for max non-adjacent element sum.
         *
         * ALGORITHM: In-place DP (modifies input array)
         * TC: O(n) | SC: O(1)
         */
        public int maxSubsetSum1(int[] arr) {
            if (arr.length == 0) return 0;
            arr[0] = Math.max(0, arr[0]);
            if (arr.length == 1) return arr[0];
            arr[1] = Math.max(arr[0], arr[1]);
            for (int i = 2; i < arr.length; i++)
                arr[i] = Math.max(arr[i - 1], arr[i] + arr[i - 2]);
            return arr[arr.length - 1];
        }


        /*
        8
2
4
3
5
2
6
4
5
Sample Output 2

12
[22344556] ---> [21231212]
10


10

2
4
2
6
1
7
8
9
2
1
[1212]
6
3 5 7 11 5 8
5 7 11 10 5 8


3 5 5 7 8 11
5 5 7 8 10 11

         */


        /*
         * PROBLEM: Beautiful Pairs (HackerRank)
         * Maximize matching pairs between lists A and B after at most one element swap in B.
         *
         * ALGORITHM: Sort + binary search with one allowed swap
         * TC: O(n log n) | SC: O(1)
         */
        public int beautifulPairs(List<Integer> A, List<Integer> B) {
            int ans = 0;
            boolean isSwitched = false;
            Collections.sort(A);
            Collections.sort(B);
            for (int i = 0; i < A.size(); i++) {
                int index = Arrays.binarySearch(B.toArray(), A.get(i));
                if (index == -1) {
                    if (!isSwitched) {
                        if (B.get(i) != null) {
                            B.remove(i);
                        }
                        B.add(A.get(i));
                        Collections.sort(B);
                        isSwitched = true;
                        ans++;
                    }
                    continue;
                }
                ans++;
            }
            return ans;
        }


        /*
         * PROBLEM: Candies (HackerRank)
         * Minimum candies to distribute to n children satisfying left-right ordering constraints.
         *
         * ALGORITHM: Greedy single-pass with ascending/descending sequence tracking
         * TC: O(n) | SC: O(1)
         */
        public long candies(int n, List<Integer> arr) {
            int descending_seq = 0;
            long sum = 0;
            int prev_c = 0;
            int prev_num_of_candies = 0;
            for (int c : arr) {
                if (c >= prev_c) {
                    if (descending_seq > 0) {
                        // agjust local max value if descending sequence
                        // was longer than ascending
                        if (descending_seq >= prev_num_of_candies) {
                            sum += 1 + descending_seq - prev_num_of_candies;
                        }
                        // last of descending = local minimum
                        prev_num_of_candies = 1;
                        descending_seq = 0;
                    }
                    if (c > prev_c) {
                        ++prev_num_of_candies;
                    } else {
                        // optimal if previous value is the same
                        prev_num_of_candies = 1;
                    }
                    sum += prev_num_of_candies;
                } else {
                    ++descending_seq;
                    // For 3 descending numbers in a row this summing strategy
                    // will increment like sum+=1+2+3 which is the same as
                    // more usual and expected sum+=3+2+1
                    sum += descending_seq;
                }
                prev_c = c;
            }
            // If we finished on descending order, update last local max
            if (descending_seq >= prev_num_of_candies) {
                sum += 1 + descending_seq - prev_num_of_candies;
            }
            return sum;
        }
    }


    /*
     * PROBLEM: Count Beautiful Substrings / Vowel Substrings (LeetCode 1641 / 2401)
     * Count substrings that contain all five vowels (a, e, i, o, u).
     *
     * ALGORITHM: Brute-force enumerate all substrings, check each for all vowels
     * TC: O(n^2) | SC: O(n^2)
     */
    public static int vowelsubstring(String s) {
        int n = s.length();
        if (n < 5) return 0;

        int ans = 0;
        String[] dp = new String[n * (n + 1) / 2]; // to store lengths of all valid substrings

        for (int i = 0; i < n; i++)
            for (int j = i + 1; j <= n; j++)
                dp[i] = (s.substring(i, j));


        for (int m = 0; m < dp.length; m++) {

            String str = dp[m];
            // Hash Array of size 5
            // such that the index 0, 1, 2, 3 and 4
            // represent the vowels a, e, i, o and u
            int[] hash = new int[5];

            // Loop the String to mark the vowels
            // which are present
            for (int i = 0; i < str.length(); i++) {

                if (str.charAt(i) == 'A' || str.charAt(i) == 'a')
                    hash[0] = 1;

                else if (str.charAt(i) == 'E' || str.charAt(i) == 'e')
                    hash[1] = 1;

                else if (str.charAt(i) == 'I' || str.charAt(i) == 'i')
                    hash[2] = 1;

                else if (str.charAt(i) == 'O' || str.charAt(i) == 'o')
                    hash[3] = 1;

                else if (str.charAt(i) == 'U' || str.charAt(i) == 'u')
                    hash[4] = 1;
            }

            // Loop to check if there is any vowel
            // which is not present in the String
            for (int i = 0; i < 5; i++) {
                if (hash[i] == 0) {
                    ans++;
                }
            }
        }
        return ans;
    }


    /*
     * PROBLEM: Select Stocks to Maximize Profit (Helper)
     * Select non-adjacent stocks to maximize profit given a savings budget constraint.
     *
     * ALGORITHM: DP on non-adjacent selection with budget constraint
     * TC: O(n^2) | SC: O(1)
     */
    public static int selectStock(int saving, List<Integer> currentValue, List<Integer> futureValue) {
        int n = currentValue.size();
        int maxProfit = 0;
        for (int k = 0; k < n; k++) {
            int prev = 0, prepre = 0;

            for (int i = k; i < n; i++) {
                if (currentValue.get(i) < saving) {
                    int profit = futureValue.get(i) - currentValue.get(i);
                    int curr = Math.max(prev, prepre + profit);
                    prepre = prev;
                    prev = curr;
                    saving -= currentValue.get(i);
                }
            }
            if (prev > maxProfit) maxProfit = prev;
        }
        return maxProfit;
    }


    /*
     * PROBLEM: Reach the End of Grid (Helper)
     * Determine if the bottom-right corner of a grid can be reached within maxTime steps.
     *
     * ALGORITHM: DFS exploring all paths
     * TC: O(2^(m*n)) | SC: O(m*n)
     */
    public static String reachTheEnd(List<String> grid, int maxTime) {
        // Write your code here

        int n = grid.size();
        if (n == 0) return "No";

        // matrix is ready
        int[][] matrix = new int[n][n];
        for (int k = 0; k < grid.size(); k++) {
            for (int i = 0; i < grid.get(k).length(); i++) {
                matrix[i][k] = grid.get(k).charAt(i);
            }
        }

        // get min time for btoph cases --> to the right and bottom
        int minTime = dfs(matrix, n, n, 0, 0, new int[n * n], 0, Integer.MAX_VALUE);

        return minTime > maxTime ? "No" : "Yes";
    }


    /*
     * PROBLEM: DFS min path in matrix (Helper)
     * DFS to find the minimum-length path from top-left to bottom-right in a matrix.
     *
     * ALGORITHM: DFS with path tracking
     * TC: O(2^(m*n)) | SC: O(m*n)
     */
    private static int dfs(int mat[][], int m, int n,
                           int i, int j, int path[], int idx, int min) {
        path[idx] = mat[i][j];

        // Reached the bottom of the matrix so we are left with
        // only option to move right
        if (i == m - 1) {
            for (int k = j + 1; k < n; k++) {
                if (mat[i][k] != '#') path[idx + k - j] = mat[i][k];
                else return 0;
            }
            int pathArrSize = Math.toIntExact(Arrays.stream(path).filter(x -> x != -1).count());

            if (min > pathArrSize) min = pathArrSize;
            return min;
        }

        // Reached the right corner of the matrix we are left with
        // only the downward movement.
        if (j == n - 1) {
            for (int k = i + 1; k < m; k++) {
                if (mat[i][k] != '#') path[idx + k - i] = mat[k][j];
                else return 0;
            }
            int pathArrSize = Math.toIntExact(Arrays.stream(path).filter(x -> x != -1).count());
            if (min > pathArrSize) min = pathArrSize;
            return min;
        }
        // Find all the paths that are possible after moving down
        dfs(mat, m, n, i + 1, j, path, idx + 1, min);

        // Find all the paths that are possible after moving right
        dfs(mat, m, n, i, j + 1, path, idx + 1, min);
        return min;
    }

    /*
     * PROBLEM: Matrix DFS stub (Helper)
     * Stub for DFS traversal of a 2D matrix (unimplemented).
     *
     * ALGORITHM: DFS (stub)
     * TC: O(1) | SC: O(1)
     */
    public static int dfs(int[][] matrix, int row, int col) {

//        int n = matrix.length;
//        // dp array to store minimum via paths
//        int[] dp = new int[n  *  n];
//        Arrays.fill(dp, -1);
//
//        dp[0] = 0;
//
//        if (col == 1) {
//          // matrix traversal
//            for (int r = 0; r < n; r++) {
//                for (int c = 1; c < n; c++) {
//                    // check for boundary conditions
//                     if (c < n-1  && r < n-1 ){
//                         // if all directions are blocked
//                         if (matrix[r+1][c] == '#' && matrix[r][c+1] == '#' && matrix[r-1][c] == '#' && matrix[r][c-1] == '#')
//                             break;
//                     }
//                    dp[r][c] = Math.min()
//                }
//            }
//
//        }
//
        return 0;
    }


    int[] nums;

    /*
     * PROBLEM: Minimum Absolute Difference Queries (LeetCode 1906)
     * For each range query [l, r], find the minimum absolute difference between any two distinct elements.
     *
     * ALGORITHM: Per-query subarray sort and linear scan
     * TC: O(q*n log n) | SC: O(n)
     */
    public int[] minDifference(int[] nums, int[][] queries) {
        this.nums = nums;
        int[] ans = new int[queries.length];
        if (nums.length == 0) return ans;
        if (queries.length == 0) return ans;

        int k = 0;
        for (int[] q : queries) {
            ans[k] = processQuery(q);
            k++;
        }
        return ans;
    }

    /*
     * PROBLEM: Process single range query for min absolute difference (Helper)
     * Extracts the subarray for a query range, sorts it, and scans for the minimum absolute difference.
     *
     * ALGORITHM: Subarray extraction + sort + linear scan
     * TC: O(n log n) | SC: O(n)
     */
    protected int processQuery(int[] query) {
        int min = Integer.MAX_VALUE;
        int start = query[0];
        int end = query[1];
        int[] array = new int[end - start + 1];

        int index = 0;
        for (int i = start; i <= end; i++) {
            array[index] = this.nums[i];
            index++;
        }
        Arrays.sort(array);
        for (int i = 0; i < array.length; i++) {
            System.out.print(array[i] + " ");
        }
        System.out.println();
        int k = 0;
        for (int i = 0; i < array.length - 1; i++) {
            if (Math.abs(array[k + 1] - array[k]) == 0) min = -1;
            else if ((min == -1 || min > Math.abs(array[k + 1] - array[k]) && Math.abs(array[k + 1] - array[k]) != 0)) {
                min = Math.abs(array[k + 1] - array[k]);
            }
            k++;
        }

        System.out.println(min);
        return min;
    }

    /*
     * PROBLEM: Count Sub Islands (LeetCode 1905)
     * Count islands in grid2 that are fully contained within islands of grid1.
     *
     * ALGORITHM: DFS (stub)
     * TC: O(m*n) | SC: O(m*n)
     */
    public int countSubIslands(int[][] grid1, int[][] grid2) {

        int[] dp1 = new int[grid1.length];
        int[] dp2 = new int[grid2.length];


        return 0;

    }

    /*
     * PROBLEM: Largest Odd Number in String (LeetCode 1903)
     * Return the longest prefix of the numeric string that represents an odd number.
     *
     * ALGORITHM: Linear scan, keep last odd-digit prefix
     * TC: O(n) | SC: O(1)
     */
    public String largestOddNumber(String num) {
        String largest = "";
        if (num.length() == 0) return "";

        for (int i = 0; i < num.length(); i++) {
            if (Integer.parseInt(String.valueOf(num.charAt(i))) % 2 != 0)
                largest = (num.substring(0, i + 1));
        }

        return largest;
    }

    /*
     * PROBLEM: Number of Full Rounds You Have Played (LeetCode 1904)
     * Count how many complete 15-minute rounds fit between a start and finish time.
     *
     * ALGORITHM: Time arithmetic with modular rounding
     * TC: O(1) | SC: O(1)
     */
    public int numberOfRounds(String startTime, String finishTime) {
        int before = 0, after = 0;
        if (Integer.parseInt(startTime.split(":")[1]) % 15 != 0) { // Time can't be taken

            if (Integer.parseInt(startTime.split(":")[1]) <= 45) {
                startTime = startTime.split(":")[0] + ":" + String.valueOf((Integer.parseInt(startTime.split(":")[1]) / 15 + 1) * 15);
            } else {
                startTime = Integer.parseInt(startTime.split(":")[0] + 1) + ":00";
                System.out.println(startTime);
            }
        }
        if (Integer.parseInt(startTime.split(":")[0]) > Integer.parseInt(finishTime.split(":")[0])) {
            before = (24 - Integer.parseInt(startTime.split(":")[0])) * 60 + Integer.parseInt(startTime.split(":")[1]);
            after = Integer.parseInt(finishTime.split(":")[0]) * 60 + Integer.parseInt(finishTime.split(":")[1]);
        } else if (Integer.parseInt(startTime.split(":")[0]) == 0 && Integer.parseInt(finishTime.split(":")[0]) == 0) {
            after = 24 * 60 - (Integer.parseInt(finishTime.split(":")[1]) - Integer.parseInt(startTime.split(":")[1]));

        } else {
            after = (Integer.parseInt(finishTime.split(":")[0]) - Integer.parseInt(startTime.split(":")[0])) * 60 +
                    (Integer.parseInt(finishTime.split(":")[1]) - Integer.parseInt(startTime.split(":")[1]));
        }

        return (before + after) / 15;


    }

    PriorityQueue<Integer> pq = new PriorityQueue<Integer>();

    /*
     * PROBLEM: Kth Smallest Element in a BST (LeetCode 230)
     * Find the k-th smallest value in a BST using DFS and a min-heap.
     *
     * ALGORITHM: DFS + min-heap (priority queue)
     * TC: O(n) | SC: O(n)
     */
    public int kthSmallest(TreeNode root, int k) {
        dfs(root);
        int prev = pq.peek() != null ? pq.peek() : 0;

        while (pq.peek() != null && k > 0) {
            prev = pq.poll();
            if (pq.peek() != null && pq.peek() == prev) continue;
            else k--;
        }

        return prev;
    }

    /*
     * PROBLEM: BST DFS traversal into priority queue (Helper)
     * Recursively traverses all tree nodes, adding each node's value to the min-heap.
     *
     * ALGORITHM: Pre-order DFS
     * TC: O(n) | SC: O(n)
     */
    public void dfs(TreeNode root) {

        if (root == null)
            return;

        /* first print data of root */
        System.out.print(root.val + " ");

        pq.add(root.val);
        /* then recur on left sutree */
        dfs(root.left);


        pq.add(root.val);
        /* now recur on right subtree */
        dfs(root.right);

    }


    // left root  right
    List<Integer> ans = new ArrayList<>();

    /*
     * PROBLEM: Binary Tree Inorder Traversal (LeetCode 94)
     * Return the inorder (left-root-right) traversal of a binary tree as a list.
     *
     * ALGORITHM: Recursive inorder DFS
     * TC: O(n) | SC: O(n)
     */
    public List<Integer> inorderTraversal(TreeNode root) {

        inorder(root);
        return this.ans;
    }


    /*
     * PROBLEM: Recursive inorder traversal helper (Helper)
     * Recursively visits left subtree, root, then right subtree, appending values to the result list.
     *
     * ALGORITHM: Recursive inorder DFS
     * TC: O(n) | SC: O(h)
     */
    public void inorder(TreeNode root) {

        if (root == null)
            return;

        /* then recur on left subtree */
        dfs(root.left);

        ans.add(root.val);
        /* first print data of root */
        System.out.print(root.val + " ");

        /* now recur on right subtree */
        dfs(root.right);

    }

//    List<List<Integer>> ans = new ArrayList<>();

    /*
    public List<List<Integer>> levelOrder(Node root) {

        if (root == null)
            return new ArrayList<>();

        List<Integer> result = new ArrayList<>();
        // Standard level order traversal code
        // using queue
        Queue<Node> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root
        while (!q.isEmpty()) {
            int n = q.size();

            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                Node p = q.peek();
                q.remove();
                result.add(p.val);
                // Enqueue all children of
                // the dequeued item
                for (int i = 0; i < p.children.size(); i++)
                    q.add(p.children.get(i));
                n--;
            }
            ans.add(result);
            result = new ArrayList<>();
        }
        Collections.reverse(ans);
        return ans;

        return ans;
    }

     */

    List<Double> answer = new ArrayList<>();

    /*
     * PROBLEM: Average of Levels in Binary Tree (LeetCode 637)
     * Return the average value of nodes at each level of a binary tree.
     *
     * ALGORITHM: BFS level-order traversal
     * TC: O(n) | SC: O(n)
     */
    public List<Double> averageOfLevels(TreeNode root) {

        if (root == null)
            return new ArrayList<>();

        List<Integer> result = new ArrayList<>();
        // Standard level order traversal code
        // using queue
        Queue<TreeNode> q = new LinkedList<>(); // Create a queue
        q.add(root); // Enqueue root
        while (!q.isEmpty()) {
            int n = q.size();

            // If this node has children
            while (n > 0) {
                // Dequeue an item from queue
                TreeNode p = q.peek();
                q.remove();
                result.add(p.val);
                // Enqueue all children of
                // the dequeued item
                if (p.left != null) q.add(p.left);
                if (p.right != null) q.add(p.right);
                n--;
            }

            double value = result.stream().mapToDouble(a -> a).sum() / result.size();
            answer.add(value);
            result = new ArrayList<>();
        }
        return answer;
    }

    /*
     * PROBLEM: Convert BST to Greater Tree (LeetCode 538 / 1038)
     * Replace each node's value with the sum of all values greater than or equal to it in the BST.
     *
     * ALGORITHM: Reverse inorder traversal (right-root-left) with running sum
     * TC: O(n) | SC: O(h)
     */
    public TreeNode convertBST(TreeNode root) {
        convertBSTRec(root, 0);
        return root;
    }

    /*
     * PROBLEM: BST to Greater Tree recursive helper (Helper)
     * Recursively accumulates the sum from the right subtree and updates each node's value.
     *
     * ALGORITHM: Reverse inorder DFS with accumulated parent value
     * TC: O(n) | SC: O(h)
     */
    private int convertBSTRec(TreeNode root, int parentVal) {
        if (root == null) return 0;
        int rightVal = convertBSTRec(root.right, parentVal);
        root.val = rightVal + parentVal + root.val;
        int leftVal = convertBSTRec(root.left, parentVal);
        return root.val - parentVal + leftVal;
    }


}
