package com.company;

import java.lang.reflect.Array;
import java.util.*;
import java.util.HashMap;
import java.util.stream.Collectors;

public class PractiseProblems {

    /*
     * PROBLEM: Make String a Subsequence Using Cyclic Increments (LeetCode 2825)
     * Check if str2 can become a subsequence of str1 by cyclically incrementing characters of str1 at most once each.
     *
     * ALGORITHM: Two-pointer greedy scan
     * TC: O(n) | SC: O(1)
     *
     * Example: str1 = "abc", str2 = "ad" → increment str1[2] 'c'→'d', str1 becomes "abd", str2 is subsequence → true
     * Example: str1 = "zc",  str2 = "ad" → increment 'z'→'a' and 'c'→'d', str1 becomes "ad" → true
     */
    public boolean canMakeSubsequence(String str1, String str2) {
        int i, j;
        for (i = 0, j = 0; i < str1.length() && j < str2.length(); i++) {
            char curr = str1.charAt(i), other = str2.charAt(j);
            char next = (char) ('a' + (((curr - 'a') + 1) % 26));
            if (curr == other || next == other) j++;
        }
        return j == str2.length();
    }

    /*
     * PROBLEM: Intersection of Multiple Arrays (LeetCode 2248)
     * Find all integers present in every array of the 2D input, returned sorted in ascending order.
     *
     * ALGORITHM: Iterative set intersection across all rows
     * TC: O(n*m) | SC: O(n)
     */
    public List<Integer> intersection(int[][] nums) {
        Set<Integer> set1 = new HashSet<>();
        for (int[] num : nums) {
            Set<Integer> set2 = Arrays.stream(num).boxed().collect(Collectors.toCollection(HashSet::new));
            if (set1.isEmpty()) set1.addAll(set2);
            set1.retainAll(set2); // set1 now contains only common elements
        }
        return new ArrayList<>(set1).stream().sorted().collect(Collectors.toList());
    }

    /*
     * PROBLEM: Partition Array Such That Maximum Difference Is K (LeetCode 2294)
     * Determine whether the array can be partitioned into groups of size k such that
     * the difference between the max and min of each group does not exceed k.
     *
     * ALGORITHM: Frequency map + greedy validation
     * TC: O(n) | SC: O(n)
     */
    public boolean partitionArray(int[] nums, int k) {
        int n = nums.length;
        if (k == 1) return true;
        int v = n / k;
        if (n % k != 0) return false;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) map.put(num, map.getOrDefault(num, 0) + 1);
        if (map.keySet().size() < k) return false;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) if (entry.getValue() > v) return false;
        return true;
    }

    /*
     * PROBLEM: Find Least Frequent Digit (Helper)
     * Return the digit that appears least frequently in the decimal representation of n; ties broken by smallest digit.
     *
     * ALGORITHM: Digit frequency map + linear scan
     * TC: O(d) | SC: O(d)  where d = number of digits in n
     */
    public int getLeastFrequentDigit(int n) {
        Map<Integer, Integer> tm = new HashMap<>();
        for (char c : String.valueOf(n).toCharArray()) {
            int d = c - '0';
            tm.put(d, tm.getOrDefault(d, 0) + 1);
        }

        int min = Integer.MAX_VALUE, freq = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            if (freq > entry.getValue()) {
                freq = entry.getValue();
                min = entry.getKey();
            }
            if (freq == entry.getValue()) {
                min = Math.min(min, entry.getKey());
            }
        }

        return min;
    }

    /*
     * PROBLEM: Score Cards Matching Character (Helper)
     * Count the score of cards whose left or right face matches character x using set-based deduplication.
     *
     * ALGORITHM: HashSet categorisation + combinatorial counting
     * TC: O(n) | SC: O(n)
     */
    //Not a good quality problem, Can skip it
    public int score(String[] cards, char x) {
        Set<String> bothSameSet = new HashSet<>(), leftSameSet = new HashSet<>(), rightSameSet = new HashSet<>();
        int bothSame = 0, leftSame = 0, rightSame = 0;
        for (String card : cards) {
            char l = card.charAt(0), r = card.charAt(1);
            if (l == x && r == x) {
                bothSame++;
                bothSameSet.add(card);
            } else if (l == x) {
                leftSame++;
                leftSameSet.add(card);
            } else if (r == x) {
                rightSame++;
                rightSameSet.add(card);
            }
        }

        if (bothSame == 0 && leftSameSet.size() <= 1 && rightSameSet.size() <= 1) return 0;
        if (bothSame == 0 && (leftSame == 0 || rightSame == 0)) {
            if (leftSameSet.size() == 1 || rightSameSet.size() == 1) return 0;
            return Math.max(leftSame, rightSame) / 2;
        }

        if (bothSame == leftSame + rightSame) return bothSame;
        return Math.min(bothSameSet.size(), Math.abs(rightSameSet.size() - leftSameSet.size())) + Math.min(rightSameSet.size(), leftSameSet.size());
    }


    /*
     * PROBLEM: Recover Relative Order of Friends (Helper)
     * Return the subset of friends that appear in the given order array, preserving the order's sequence.
     *
     * ALGORITHM: HashSet membership check + order traversal
     * TC: O(n) | SC: O(n)
     */
    public int[] recoverOrder(int[] order, int[] friends) {
        int[] ans = new int[friends.length];
        Arrays.fill(ans, 0);
        int idx = 0;
        Set<Integer> set = Arrays.stream(friends).boxed().collect(Collectors.toSet());
        for (int o : order) {
            if (set.contains(o)) ans[idx++] = o;
        }
        return ans;
    }

    class Solution {
        /*
         * PROBLEM: Subset Sum Feasibility (Helper)
         * Check whether any subset of arr sums exactly to k using bottom-up DP.
         *
         * ALGORITHM: 0/1 Knapsack (1-D DP boolean table)
         * TC: O(n*k) | SC: O(k)
         */
        private boolean possible(int[] arr, int k) {
                boolean[] dp = new boolean[k + 1];
                int[] prev = new int[k + 1];
                dp[0] = true;

                for (int i = 0; i < arr.length; i++) {
                    for (int j = k; j >= arr[i]; j--) {
                        if (dp[j - arr[i]]) {
                            dp[j] = true;
                            prev[j] = i; // Track the index of the element contributing to the sum
                        }
                    }
                }

                if (!dp[k]) return false; // If sum k is not achievable
                return true;
            }
    }


    /*
     * PROBLEM: Minimum Operations to Make Array Non-Decreasing (LeetCode 2033)
     * Find the minimum number of increment operations to make each element no smaller than the previous.
     *
     * ALGORITHM: Recursive enumeration (TODO: add memoisation to fix stack overflow)
     * TC: O(n) | SC: O(n)
     */
    //TODO: Fix stack overflow
    // add DP to optimise solution
        public int minimumOperations(List<Integer> nums) {
            int[] dp = new int[nums.size()];
            Arrays.fill(dp, -1);
            return mo(nums, 0, -1, 0);
        }

        /*
         * PROBLEM: Recursive Helper for minimumOperations (Helper)
         * Recursively counts operations needed starting from index ind with the given previous element and running max.
         *
         * ALGORITHM: Recursion (needs memoisation for correctness/performance)
         * TC: O(n) | SC: O(n)
         */
        private int mo(List<Integer> nums, int ind, int prev, int max) {
            // base case
            if (ind >= nums.size()) return 0;

            int cnt = 0 ;
            prev = nums.get(ind);
            //take
            if (nums.get(ind) > prev) cnt ++;
            else max = Math.max(cnt, max);
            mo(nums, ind++, prev, max);
            return max;
        }


}
