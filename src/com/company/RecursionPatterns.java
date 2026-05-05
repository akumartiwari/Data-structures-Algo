package com.company;

import common.Pair;

import java.util.HashMap;
import java.util.*;
import java.util.stream.Collectors;

public class RecursionPatterns {
    private static Set<List<Integer>> ans;

    /*
     * PROBLEM: Entry point (Main)
     * Program entry point for RecursionPatterns demonstrations.
     *
     * ALGORITHM: N/A
     * TC: O(1) | SC: O(1)
     */
    public static void main(String[] args) {
        int[] arr = {2, 3, 0, 1, 4};
//        int n = 4;
//        int targetSum = 7;
//        List<Integer> ds = new ArrayList<>();
//        printAllSubsequences(arr, targetSum, 0, 0, ds);
//        printAnyoneSubsequence(arr, targetSum, 0, 0, ds);
//        System.out.println();
//        System.out.println("Count: " + printCountSubsequences(arr, targetSum, 0, 0));
//        combinationSum(arr, targetSum);
//        System.out.println((int) ans.stream().filter(x -> x.size() > 0).count());
//        for (List<Integer> list : ans) {
//            System.out.println(list.stream().map(Object::toString).collect(Collectors.joining(", ")));
//        }
//        String str1 = "iloveicecreamandmango"; // for first test case
//        String str2 ="ilovesamsungmobile";     // for second test case
//        int n1 = str1.length();                 // length of first string
//        int n2 = str2.length();                 // length of second string
//
//        // List of strings in dictionary
//        List <String> dict= Arrays.asList("mobile","samsung","sam","sung",
//                "man","mango", "icecream","and",
//                "go","i","love","ice","cream");
//        System.out.println("First Test:");
//
//        // call to the method
//        wordBreak(n1,dict,str1);
//        System.out.println("\nSecond Test:");
//
//        // call to the method
//        wordBreak(n2,dict,str2);

        System.out.println("jumps=" + minJumps(arr));
    }

    /*
     * PROBLEM: Print All Subsequences With Given Sum (Helper)
     * Print all subsequences of the array that sum to targetSum.
     *
     * ALGORITHM: Recursion (take/not-take)
     * TC: O(2^N) | SC: O(N)
     */
    private static void printAllSubsequences(int[] arr, int targetSum, int idx, int sum, List<Integer> ds) {
        // base case
        if (idx == arr.length) {
            if (sum == targetSum) {
                ds.forEach(x -> System.out.print(x + " "));
                System.out.println();
            }
            return;
        }


        // take
        sum += arr[idx];
        ds.add(arr[idx]);
        printAllSubsequences(arr, targetSum, idx + 1, sum, ds);
        sum -= arr[idx];
        ds.remove(ds.size() - 1); // remove last element
        // not-take
        printAllSubsequences(arr, targetSum, idx + 1, sum, ds);
    }

    /*
     * PROBLEM: Print Any One Subsequence With Given Sum (Helper)
     * Print a single subsequence that sums to targetSum, short-circuiting on first find.
     *
     * ALGORITHM: Recursion (take/not-take, short-circuit)
     * TC: O(2^N) | SC: O(N)
     */
    private static boolean printAnyoneSubsequence(int[] arr, int targetSum, int idx, int sum, List<Integer> ds) {
        // base case
        if (idx == arr.length) {
            if (sum == targetSum) {
                ds.forEach(x -> System.out.print(x + " "));
                return true;
            }
            return false;
        }


        // take
        sum += arr[idx];
        ds.add(arr[idx]);
        if (printAnyoneSubsequence(arr, targetSum, idx + 1, sum, ds)) return true;
        sum -= arr[idx];
        ds.remove(ds.size() - 1); // remove last element
        // not-take
        return printAnyoneSubsequence(arr, targetSum, idx + 1, sum, ds);
    }

    /*
     * PROBLEM: Count Subsequences With Given Sum (Helper)
     * Count the number of subsequences of the array that sum to targetSum.
     *
     * ALGORITHM: Recursion (take/not-take)
     * TC: O(2^N) | SC: O(N)
     */
    private static int printCountSubsequences(int[] arr, int targetSum, int idx, int sum) {
        // base case
        // Only can be done if array contains positive elements
        if (sum > targetSum) return 0;
        if (idx == arr.length) {
            if (sum == targetSum) {
                return 1;
            }
            return 0;
        }


        // take
        sum += arr[idx];
        int take = printCountSubsequences(arr, targetSum, idx + 1, sum);
        sum -= arr[idx];
        // not-take
        int notTake = printCountSubsequences(arr, targetSum, idx + 1, sum);
        return take + notTake;
    }

    /*
     * PROBLEM: Combination Sum (LeetCode 39)
     * Find all combinations of candidates that sum to target; each candidate may be reused.
     *
     * ALGORITHM: Backtracking
     * TC: O(2^N * N) | SC: O(N)
     */
//    Input: candidates = [2,3,6,7], target = 7
//    Output: [[2,2,3],[7]]
//    Input: candidates = [2,3,5], target = 8
//    Output: [[2,2,2,2],[2,3,3],[3,5]]
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {

        ans = new HashSet<>();
        List<Integer> ds = new ArrayList<>();
        combinations(candidates, target, 0, 0, ds);
        return new ArrayList<>(ans);
    }

    /*
     * PROBLEM: Combination Sum Backtrack (Helper)
     * Backtracking helper that builds candidate combinations accumulating into ds.
     *
     * ALGORITHM: Backtracking
     * TC: O(2^N) | SC: O(N)
     */
    private static void combinations(int[] arr, int targetSum, int idx, int sum, List<Integer> ds) {

        // base case
        if (sum == targetSum) {
            ans.add(new ArrayList<>(ds));
        } else if (sum > targetSum) return;
        else {
            for (int i = idx; i < arr.length; i++) {
                // take
                sum += arr[i];
                ds.add(arr[i]);
                combinations(arr, targetSum, i, sum, ds);
                sum -= arr[i];
                ds.remove(new Integer(arr[i])); // remove last element
            }
        }
    }

    /*
     * PROBLEM: Factor Combinations (LeetCode 254)
     * Return all ways to express n as a product of factors (excluding 1 and n itself).
     *
     * ALGORITHM: Recursion (DFS)
     * TC: O(N) | SC: O(N)
     * Important: always seek factors greater than the current one to avoid duplicates.
     */
//    Input: n = 12
//    Output: [[2,6],[3,4],[2,2,3]]
    public List<List<Integer>> getFactors(int n) {
        if (n <= 2) return new ArrayList<>();

        return factors(n, 2);
    }

    /*
     * PROBLEM: Factor Combinations Helper (Helper)
     * Recursively enumerate factor pairs of num starting from factor fact.
     *
     * ALGORITHM: Recursive factorization
     * TC: O(sqrt(N)) | SC: O(log N)
     */
    private List<List<Integer>> factors(int num, int fact) {

        List<List<Integer>> ans = new ArrayList<>();

        // Iterate from 2 till sqrt of num as factors can only exist upto sqrt of num

        int end = (int) Math.sqrt(num);
        for (int idx = fact; idx <= end; idx++) {
            int y = num / idx;
            int z = num % idx;

            // Always look for greater number while finding index
            if (y < idx) break;
            else if (z == 0) {
                List<Integer> parts = new ArrayList<>();
                parts.add(idx);
                parts.add(y);
                ans.add(parts);
                List<List<Integer>> list = factors(y, idx);
                for (List<Integer> factorsRelativeToPart : list) {
                    factorsRelativeToPart.add(idx);
                    ans.add(factorsRelativeToPart);
                }
            }
        }
        return ans;
    }

    /*
     * PROBLEM: Check if all a's precede all b's (LeetCode 2124)
     * Return true if all 'a' characters appear before any 'b' character in s.
     *
     * ALGORITHM: Linear scan
     * TC: O(N) | SC: O(1)
     */
    public boolean checkString(String s) {
        int n = s.length();
        boolean aFlag = true;
        for (int i = 0; i < n; i++) {
            if (aFlag && s.charAt(i) == 'b') {
                aFlag = false;
                continue;
            }
            if (!aFlag && s.charAt(i) == 'a') return false;
        }
        return true;
    }

    /*
     * PROBLEM: Number of Laser Beams in a Bank (LeetCode 2125)
     * Count laser beams between consecutive non-empty rows of a bank's security device grid.
     *
     * ALGORITHM: Linear scan (row-by-row product)
     * TC: O(N * M) | SC: O(N)
     */
    public int numberOfBeams(String[] bank) {
        int n = bank.length;
        int ans = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int one = countOne(bank[i]);
            map.put(i, one);
        }

        int prev = map.get(0);
        for (int i = 1; i < n; i++) {
            if (map.get(i) > 0) {
                ans += prev * map.get(i);
                prev = map.get(i);
            }
        }

        return ans;
    }


    /*
     * PROBLEM: Count 1s in String (Helper)
     * Count the number of '1' characters in a binary string.
     *
     * ALGORITHM: Linear scan
     * TC: O(N) | SC: O(1)
     */
    private int countOne(String str) {
        int cnt = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '1') cnt++;
        }
        return cnt;
    }

    /*
     * PROBLEM: Destroying Asteroids (LeetCode 2126)
     * Determine if a planet can absorb all asteroids by always absorbing the smallest reachable one.
     *
     * ALGORITHM: Greedy + Binary Search
     * TC: O(N log N) | SC: O(N)
     */
    public boolean asteroidsDestroyed(int mass, int[] asteroids) {
        int n = asteroids.length;
        List<Integer> coll = Arrays.stream(asteroids).boxed().sorted().collect(Collectors.toList());

        long nMass = mass;
        while (n-- > 0) {
            int idx = closestMass(coll, nMass);
            if ((long) coll.get(idx) > nMass) {
                return false;
            }
            nMass += (long) coll.get(idx);
            coll.remove(idx); // O(1)
        }

        return n <= 0;
    }

    /*
     * PROBLEM: Binary Search Closest Mass (Helper)
     * Binary search for the index of the largest asteroid not exceeding mass.
     *
     * ALGORITHM: Binary Search
     * TC: O(log N) | SC: O(1)
     */
    private int closestMass(List<Integer> asteroids, long mass) {
        int l = 0, r = asteroids.size() - 1;
        while (l <= r) {
            int mid = (int) Math.abs(l + (r - l) / 2);
            if ((long) asteroids.get(mid) > mass) {
                r = mid - 1;
            } else if ((long) asteroids.get(mid) < mass) {
                if (l != mid) l = mid;
                else return l;
            } else {
                return mid;
            }
        }

        return l;
    }

    /*
     * PROBLEM: Maximum Employees to Be Invited to a Meeting (LeetCode 2127)
     * Find the maximum number of employees that can sit at a round table given mutual-favorite constraints.
     *
     * ALGORITHM: Cycle detection + DFS (directed graph)
     * TC: O(N) | SC: O(N)
     */
    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        Set<Integer> visited = new HashSet<>();
        // initiate dependencies
        Map<Integer, Set<Integer>> childToParents = new HashMap<>();
        for (int i = 0; i < favorite.length; i++) {
            childToParents.computeIfAbsent(favorite[i], k -> new HashSet<>());
            childToParents.get(favorite[i]).add(i);
        }

        int max = 0;
        // all the cycles with size 2, along with its connected chain, can fit into a table
        int size2Together = 0;
        for (int i = 0; i < n; i++) {
            if (visited.contains(i))
                continue;

            // cycleSize & cycleEntryPoint
            int[] tableMeta = findCycle(i, favorite, visited);
            if (tableMeta[0] == 2) {
                childToParents.get(tableMeta[1]).remove(favorite[tableMeta[1]]);
                childToParents.get(favorite[tableMeta[1]]).remove(tableMeta[1]);

                tableMeta[0] = dfs(tableMeta[1], childToParents, visited) + dfs(favorite[tableMeta[1]], childToParents, visited);
                size2Together += tableMeta[0];
            }
            max = Math.max(max, tableMeta[0]);
        }

        return Math.max(max, size2Together);

    }

    /*
     * PROBLEM: Find Cycle in Favorite Array (Helper)
     * Detect the cycle reachable from startPoint in the favorite[] functional graph.
     *
     * ALGORITHM: Floyd-style pointer walk
     * TC: O(N) | SC: O(N)
     * Returns: new int[] {cycleSize, entryPoint}
     */
    private int[] findCycle(int startPoint, int[] favorite, Set<Integer> visited) {
        int next = startPoint;
        int entryPoint = -1;
        int cycleSize = 0;

        // find entry point of cycle
        while (entryPoint == -1) {
            visited.add(next);
            cycleSize++;
            next = favorite[next];
            if (visited.contains(next))
                entryPoint = next;
        }

        // remove the segment from startPoint to entryPoint
        next = startPoint;
        while (next != entryPoint) {
            cycleSize--;
            next = favorite[next];
        }

        return new int[]{cycleSize, entryPoint};
    }

    /*
     * PROBLEM: DFS Chain Length (Helper)
     * Compute the longest chain ending at child by walking parent pointers in the DAG.
     *
     * ALGORITHM: DFS (tree chain)
     * TC: O(N) | SC: O(N)
     */
    private int dfs(int child, Map<Integer, Set<Integer>> childToParents, Set<Integer> visited) {
        visited.add(child);

        if (!childToParents.containsKey(child) || childToParents.get(child).isEmpty())
            return 1;

        int max = 0;
        for (int parent : childToParents.get(child)) {
            max = Math.max(max, dfs(parent, childToParents, visited) + 1);
        }

        return max;
    }

    /*
     * PROBLEM: Longest Increasing Subsequence (LeetCode 300)
     * Find the length of the longest strictly increasing subsequence using patience sorting.
     *
     * ALGORITHM: Patience sorting (binary search)
     * TC: O(N log N) | SC: O(N)
     * NOTE: missing return statement (intentional bug preserved).
     */
    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) return 0;
        List<Integer> lis = new ArrayList<>();
        for (int num : nums) {
            int idx = Collections.binarySearch(lis, num);
            if (idx < 0) idx = -(idx + 1);
            if (idx == lis.size()) lis.add(num);
            else lis.set(idx, num);
        }
        return lis.size();
    }

    private int LIS(List<Integer> nums) {
        List<Integer> tails = new ArrayList<>();
        for (int num : nums) {
            int idx = Collections.binarySearch(tails, num);
            if (idx < 0) idx = -(idx + 1);
            if (idx == tails.size()) tails.add(num);
            else tails.set(idx, num);
        }
        return tails.size();
    }

    /*
     * PROBLEM: Minimum Operations to Make Array K-Increasing (LeetCode 2111)
     * Find the minimum replacements so every k-th subsequence is non-decreasing.
     *
     * ALGORITHM: LIS on each k-indexed subsequence
     * TC: O(N log N) | SC: O(N)
     */
    public int kIncreasing(int[] arr, int k) {
        int n = arr.length;
        int total = 0;
        List<List<Integer>> subsequences = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            List<Integer> part = new ArrayList<>();
            for (int j = i; j < n; j += k) {
                part.add(arr[j]);
            }
            subsequences.add(part);
        }

        for (List<Integer> part : subsequences) {
            total += part.size() - LIS(part);
        }

        return total;
    }


    //TODO: We need to acumulate all possible longest subsequences with non-zero AND
    class Solution {
        /*
         * PROBLEM: Longest Subsequence With Positive AND (LeetCode Helper)
         * Find the longest subsequence whose bitwise AND is non-zero.
         *
         * ALGORITHM: LIS variant with AND constraint
         * TC: O(N * 32) | SC: O(N)
         */
        public int longestSubsequence(int[] nums) {
            Set<Integer> ans = LIS(Arrays.stream(nums).boxed().collect(Collectors.toList()));
            List<Integer> result = new ArrayList<>();
            for (int bit = 0; bit < 32; bit++) {
                List<Integer> temp = new ArrayList<>();
                for (int num : ans) {
                    if (((num >> bit) & 1) == 1) {
                        temp.add(num);
                    }
                }
                if (temp.size() > result.size()) {
                    result = temp;
                }
            }

            System.out.println(result);
            return result.size();
        }

        /*
         * PROBLEM: LIS with Non-Zero AND (Helper)
         * Build a non-decreasing subsequence set using patience-sort with AND tracking.
         *
         * ALGORITHM: Patience sorting (binary search)
         * TC: O(N log N) | SC: O(N)
         */
        private Set<Integer> LIS(List<Integer> part) {
            List<Integer> ans = new ArrayList<>();
            int lastItem = part.get(0);
            for (Integer integer : part) {
                if (integer >= lastItem) {
                    ans.add(integer);
                } else {
                    // next greater element than current one in the ans list
                    int idx = nextGreaterElementNonZeroAND(ans, integer);

                    if (idx < 0) idx = -(idx + 1);
                    if (idx == ans.size()) ans.add(integer);
                    else ans.set(idx, integer);

                }
                lastItem = ans.get(ans.size() - 1);
            }

            return new HashSet<>(ans);
        }

        /*
         * PROBLEM: Next Greater Element Search (Helper)
         * Binary search for the leftmost index in ans whose value exceeds item.
         *
         * ALGORITHM: Binary Search
         * TC: O(log N) | SC: O(1)
         */
        private int nextGreaterElementNonZeroAND(List<Integer> ans, Integer item) {

            int l = 0, r = ans.size() - 1;
            while (l < r) {
                int mid = Math.abs(l + (r - l) / 2);
                if (ans.get(mid) <= item) {
                    l = mid + 1;
                } else {
                    r = mid;
                }
            }

            return l;
        }
    }

    /*
     * PROBLEM: Longest Non-decreasing Subsequence With Non-Zero AND (Helper)
     * Find the longest non-decreasing subsequence where the running AND stays positive.
     *
     * ALGORITHM: Greedy + Binary Search
     * TC: O(N log N) | SC: O(N)
     */
    public int longestSubsequence(int[] nums) {
        List<Integer> ans = new ArrayList<>();
        List<Integer> part = Arrays.stream(nums).boxed().collect(Collectors.toList());
        int lastItem = part.get(0);
        int res = -1;
        for (Integer integer : part) {
            if (res == -1) {
                ans.add(integer);
                res = integer;
            } else if (integer >= lastItem && ((res & integer) > 0)) {
                ans.add(integer);
                res &= integer;
            } else if ((res & integer) > 0) {
                // next greater element than current one in the ans list
                int idx = nextGreaterElement(ans, integer);
                if (idx < 0) idx = -(idx + 1);

                if (ans.size() == idx) ans.add(integer);
                else ans.set(idx, integer);
                res &= integer;
            }
            lastItem = ans.get(ans.size() - 1);
        }

        return ans.size();
    }

    /*
     * PROBLEM: Binary Search for Next Greater Element (Helper)
     * Binary search for the leftmost index in ans whose value exceeds item.
     *
     * ALGORITHM: Binary Search
     * TC: O(log N) | SC: O(1)
     */
    private int nextGreaterElement(List<Integer> ans, Integer item) {

        int l = 0, r = ans.size() - 1;
        while (l < r) {
            int mid = (int) Math.abs(l + (r - l) / 2);
            if (ans.get(mid) <= item) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        return l;
    }


    /*
     * PROBLEM: Frog Jump (GFG)
     * Find the minimum energy cost for a frog to jump from step 0 to step n-1.
     *
     * ALGORITHM: Recursion (memoized)
     * TC: O(2^N) | SC: O(N)
     */
    public static int frogJumpRecursion(int n, int heights[]) {
        return helper(0, heights);
    }

    /*
     * PROBLEM: Frog Jump Recursion (Helper)
     * Recursively compute the minimum cost to reach the last step from the given index.
     *
     * ALGORITHM: Recursion (brute force)
     * TC: O(2^N) | SC: O(N)
     */
    private static int helper(int index, int[] heights) {
        if (index == (heights.length - 1)) {
            // Reached the last stair.
            return 0;
        }

        // Two available choices.
        int oneJump = Integer.MAX_VALUE;
        int twoJump = Integer.MAX_VALUE;

        if ((index + 1) < heights.length) {
            oneJump = Math.abs(heights[index] - heights[index + 1]) + helper(index + 1, heights);
        }

        if ((index + 2) < heights.length) {
            twoJump = Math.abs(heights[index] - heights[index + 2]) + helper(index + 2, heights);
        }

        return Math.min(oneJump, twoJump);
    }

    /*
     * PROBLEM: Solving Questions With Brainpower (LeetCode 2140)
     * Maximize exam points by choosing to solve or skip each question with brainpower cooldowns.
     *
     * ALGORITHM: DP (memoized recursion)
     * TC: O(N) | SC: O(N)
     */
    public long mostPoints(int[][] questions) {
        if (questions.length == 0) return 0L;
        long[] dp = new long[questions.length];
        Arrays.fill(dp, -1);
        return recursive(questions, 0, dp);
    }

    /*
     * PROBLEM: Most Points Memoized Recursion (Helper)
     * Memoized recursion helper for mostPoints; returns max points achievable from index onward.
     *
     * ALGORITHM: DP (memoized recursion)
     * TC: O(N) | SC: O(N)
     */
    private long recursive(int[][] questions, int index, long[] dp) {
        // base case
        if (index >= questions.length) return 0;

        if (dp[index] != -1) return dp[index];
        // take
        long left = questions[index][0] + recursive(questions, index + 1 + questions[index][1], dp);

        //skip
        long right = recursive(questions, index + 1, dp);

        return dp[index] = Math.max(left, right);
    }

    /*
     * PROBLEM: Jump Game II (Min Jumps) (LeetCode 45 variant)
     * Find the minimum number of jumps to reach the last index of the array.
     *
     * ALGORITHM: DFS + memoization
     * TC: O(N^2) | SC: O(N)
     */
    public static int minJumps(int[] arr) {
        if (arr.length == 0) return 0;
        int[] dp = new int[arr.length];
        Arrays.fill(dp, -1);
        return dfs(arr, 0, dp);
    }


    /*
     * PROBLEM: Min Jumps DFS (Helper)
     * DFS with memoization helper returning minimum jumps from index to end.
     *
     * ALGORITHM: DFS + memoization
     * TC: O(N^2) | SC: O(N)
     */
    private static int dfs(int[] arr, int index, int[] dp) {
        // base-case
        // If you move out of array that jump is invalid and hence count=0
        if (index >= arr.length - 1) return 0;

        // If precomputed use it
        if (dp[index] != -1) return dp[index];

        int min = 1_0000_000;
        int ca = 0;
        // For all possible jumps we need to consider all possible choices to be made
        for (int i = 1; i <= arr[index]; i++) {

            int nextPos = index + i;

            if (nextPos < arr.length) {
                // Take
                ca = 1 + dfs(arr, nextPos, dp);
            }

            min = Math.min(ca, min);
            dp[index] = min;
        }

        return min;
    }

    /*
     * PROBLEM: Find the Winner of the Circular Game (LeetCode 1823)
     * Simulate Josephus-style elimination to find the last remaining player.
     *
     * ALGORITHM: Recursion (DFS simulation)
     * TC: O(N) | SC: O(N)
     */
    public int findTheWinner(int n, int k) {
        List<Integer> friends = new ArrayList<>();
        // Store all person in an ArrayList
        for (int i = 1; i <= n; i++) friends.add(i);

        return dfs(n, k, friends, 0);
    }

    /*
     * PROBLEM: Josephus DFS (Helper)
     * Recursively eliminate every k-th friend until one remains.
     *
     * ALGORITHM: Recursion (list simulation)
     * TC: O(N) | SC: O(N)
     */
    private int dfs(int n, int k, List<Integer> friends, int ind) {
        if (friends.size() == 1) return friends.get(0);
        int ni = (ind + k - 1) % friends.size();
        friends.remove(ni);
        return dfs(n, k, friends, ni);
    }

    List<Long> res = new ArrayList<>();
    boolean solFound = false;

    /*
     * PROBLEM: Maximum Split of Positive Even Integers (LeetCode 2178)
     * Split finalSum into the maximum number of distinct positive even integers.
     *
     * ALGORITHM: Greedy + DFS
     * TC: O(sqrt(N)) | SC: O(sqrt(N))
     */
    public List<Long> maximumEvenSplit(long finalSum) {
        if (finalSum % 2 != 0) return res;
        if (finalSum == 2) {
            res.add(2L);
            return res;
        }

        mes(finalSum, 2, 0, new ArrayList<>());
        return res;
    }

    /*
     * PROBLEM: Maximum Even Split DFS (Helper)
     * DFS helper that greedily adds even numbers to the partition list.
     *
     * ALGORITHM: DFS (backtracking)
     * TC: O(sqrt(N)) | SC: O(sqrt(N))
     */
    private void mes(long finalSum, long num, long sum, List<Long> ds) {
        // base case
        if (solFound) {
            return;
        }

        if (sum == finalSum) {
            res = new ArrayList<>(ds);
            solFound = true;
            return;
        }


        if (sum + num > finalSum) return;

        // not-take
        if (sum + num <= finalSum) {
            //take
            ds.add(num);
            mes(finalSum, num + 2, sum + num, ds);
            // backtrack
            ds.remove(ds.size() - 1);
        }

        mes(finalSum, num + 2, sum, ds);
    }

    /*
     * PROBLEM: Longest Increasing Subsequence (DP) (LeetCode 300)
     * Find the length of the longest strictly increasing subsequence using memoized recursion.
     *
     * ALGORITHM: DP (memoized recursion)
     * TC: O(N^2) | SC: O(N^2)
     */
    public int lengthOfLISDP(int[] nums) {
        if (nums.length == 0) return 0;
        int[][] dp = new int[nums.length][nums.length + 1];
        for (int[] d : dp) Arrays.fill(d, -1);
        return recurse(nums, 0, -1, dp);
    }

    /*
     * PROBLEM: LIS DP Recursion (Helper)
     * Memoized recursion helper for lengthOfLISDP computing LIS from idx with previous index prev_idx.
     *
     * ALGORITHM: DP (memoized recursion)
     * TC: O(N^2) | SC: O(N^2)
     */
    private int recurse(int[] nums, int idx, int prev_idx, int[][] dp) {

        // base case
        if (idx == nums.length) return 0;

        if (dp[idx][prev_idx + 1] != -1) return dp[idx][prev_idx + 1];
        // not-take
        int len = recurse(nums, idx + 1, prev_idx, dp);
        if (prev_idx == -1 || nums[idx] > nums[prev_idx]) {
            len = Math.max(len, 1 + recurse(nums, idx + 1, idx, dp));
        }
        return dp[idx][prev_idx + 1] = len;
    }
    
    /*
     * PROBLEM: Word Break (GFG)
     * Print all possible ways to segment a string into valid dictionary words.
     *
     * ALGORITHM: Recursion
     * TC: O(2^N) | SC: O(N)
     */
    static void wordBreak(int n, List<String> dict, String s) {
        String ans = "";
        wordBreakUtil(n, s, dict, ans);
    }

    /*
     * PROBLEM: Word Break Utility (Helper)
     * Recursive helper that tries each dictionary prefix of s and recurses on the remainder.
     *
     * ALGORITHM: Recursion (prefix matching)
     * TC: O(2^N) | SC: O(N)
     */
    private static void wordBreakUtil(int n, String s, List<String> dict, String ans) {

        for (int i = 1; i <= n; i++) {
            String prefix = s.substring(0, i);
            if (dict.contains(prefix)) {
                if (i == n) {
                    ans += prefix;
                    System.out.println(ans);
                    return;
                }
                wordBreakUtil(n - i, s.substring(i, n), dict, ans + prefix + " ");
            }
        }
    }

    /*
     * PROBLEM: Decode String (LeetCode 394)
     * Decode an encoded string where k[encoded_string] means encoded_string repeated k times.
     *
     * ALGORITHM: Recursion (bracket matching)
     * TC: O(N) | SC: O(N)
     */
    public String decodeString(String s) {
        return build(s, 0, s.length());
    }

    /*
     * PROBLEM: Decode String Build (Helper)
     * Recursively build the decoded string for substring s[i..j).
     *
     * ALGORITHM: Recursion (bracket matching)
     * TC: O(N) | SC: O(N)
     */
    private String build(String s, int i, int j) {
        StringBuilder sb = new StringBuilder();
        while (i < j) {
            if (Character.isDigit(s.charAt(i))) {
                int nr = 0;
                while (Character.isDigit(s.charAt(i))) {
                    nr *= 10;
                    nr += s.charAt(i) - '0'; // digit value
                    i++;
                }

                int endBkt = i + 1, open = 1;

                while (open != 0) {
                    if (s.charAt(endBkt) == '[') open++;
                    else if (s.charAt(endBkt) == ']') open--;
                    endBkt++;
                }


                String encode = build(s, i + 1, endBkt - 1);
                // Repeat the square bracket string k times
                for (int m = 0; m < nr; m++) sb.append(encode);
                i = endBkt;
            } else {
                sb.append(s.charAt(i));
                i++;
            }
        }

        return sb.toString();
    }


    long result = Long.MAX_VALUE;

    /*
     * PROBLEM: Construct Smallest Number From DI String (LeetCode 2375)
     * Build the lexicographically smallest digit string satisfying the I/D pattern.
     *
     * ALGORITHM: DFS (backtracking)
     * TC: O(9!) | SC: O(N)
     */
    public String smallestNumber(String pattern) {

        Map<Character, Integer> map = new HashMap<>();

        Set<Integer> set = new HashSet<>();
        for (int i = 1; i <= 9; i++) set.add(i);

        for (int i = 1; i <= 9; i++) {
            set.remove(i);
            StringBuilder take = new StringBuilder();
            take.append(i);

            Set<Integer> taken = new HashSet<>();
            String sb = pattern + pattern.charAt(pattern.length() - 1);
            taken.add(i);
            dfs(1, sb, set, take, taken);
            set.clear();

            for (int k = 1; k <= 9; k++) set.add(k);
        }

        return String.valueOf(result);
    }

    /*
     * PROBLEM: Smallest Number DFS (Helper)
     * DFS backtracking helper that tries each unused digit to build a valid number matching the pattern.
     *
     * ALGORITHM: DFS (backtracking)
     * TC: O(9!) | SC: O(N)
     */
    private void dfs(int ind, String pattern, Set<Integer> set, StringBuilder take, Set<Integer> taken) {
        if (ind >= pattern.length()) {
            result = Math.min(!take.toString().equals("") ? Long.parseLong(take.toString()) : 0, result);
            return;
        }

        if (String.valueOf(take).length() >= 9) return;

        for (int s : set) {
            if (taken.contains(s)) continue;
            taken.add(s);
            if (take.toString().equals("")
                    || (pattern.charAt(ind - 1) == 'I' && s > Integer.parseInt(String.valueOf(take.charAt(ind - 1))))
                    || (pattern.charAt(ind - 1) == 'D' && s < Integer.parseInt(String.valueOf(take.charAt(ind - 1))))

            ) {
                take.append(s);
                dfs(ind + 1, pattern, set, take, taken);
                take.deleteCharAt(take.length() - 1);
            }
            taken.remove(s);
        }
    }
}