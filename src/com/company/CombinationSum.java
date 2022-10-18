package com.company;

import java.util.ArrayList;
import java.util.List;

class CombinationSum {
    // TC = O(2^n), SC = O(2^n)
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        // Arrays.sort(candidates);
        List<List<Integer>> ans = new ArrayList<>();
        bfs(new ArrayList<>(), 0, 0, ans, target, candidates);
        return ans;
    }

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
}
