package com.company;

import java.util.*;
import java.util.stream.Collectors;

class CombinationSum {
    /*
     * PROBLEM: Combination Sum (LeetCode 39)
     * Find all unique combinations of candidates that sum to target (reuse allowed).
     *
     * ALGORITHM: Backtracking (DFS)
     * TC: O(2^n) | SC: O(2^n)
     */
    // TC = O(2^n), SC = O(2^n)
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        bfs(new ArrayList<>(), 0, 0, ans, target, candidates);
        return ans;
    }

    /*
     * PROBLEM: Combination Sum Backtrack (Helper)
     * Recursive backtracking helper building combinations that sum to target.
     *
     * ALGORITHM: Backtracking
     * TC: O(2^n) | SC: O(2^n)
     */
    private void bfs(List<Integer> num, int sum, int index, List<List<Integer>> ans, int k, int[] candidates) {
        // base case
        if (sum == k) {
            ans.add(new ArrayList<>(num));
        } else if (sum > k) {
            return;
        } else {
            // For all position place numbers
            for (int p = index; p < candidates.length; p++) {
                num.add(candidates[p]);
                sum += candidates[p];
                bfs(num, sum, p, ans, k, candidates);
                num.remove(new Integer(candidates[p]));
                sum -= candidates[p];
            }
        }
    }

    /*
     * PROBLEM: Combination Sum II (LeetCode 40)
     * Find all unique combinations where each number is used at most once.
     *
     * ALGORITHM: Backtracking + HashSet dedup
     * TC: O(2^n) | SC: O(2^n)
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        // Arrays.sort(candidates);
        Set<List<Integer>> ans = new HashSet<>();
        bfs(new ArrayList<>(), 0, 0, ans, target, candidates);
        return new ArrayList<>(ans);
    }

    /*
     * PROBLEM: Combination Sum II Backtrack (Helper)
     * Recursive backtracking helper with set-based deduplication.
     *
     * ALGORITHM: Backtracking + HashSet
     * TC: O(2^n) | SC: O(2^n)
     */
    private void bfs(List<Integer> num, int sum, int index, Set<List<Integer>> ans, int k, int[] candidates) {
        // base case
        if (sum == k) {
            Collections.sort(num);
            ans.add(new ArrayList<>(num));
        } else if (sum > k) {
            return;
        } else {
            // For all position place numbers
            for (int p = index; p < candidates.length; p++) {
                num.add(candidates[p]);
                sum += candidates[p];
                bfs(num, sum, p + 1, ans, k, candidates);
                num.remove(new Integer(candidates[p]));
                sum -= candidates[p];
            }
        }

    }
}
