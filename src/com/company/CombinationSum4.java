package com.company;

import java.util.Arrays;

// TOP_DOWN
// Backtracking
class CombinationSum4 {
    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        int[] memo = new int[target + 1];
        Arrays.fill(memo, -1);
        return backtrack(nums, target, memo);
    }

    private int backtrack(int[] nums, int target, int[] memo) {
        // base cases
        if (target == 0) return 1;
        if (target < 0) return 0;

        if (memo[target] != -1) return memo[target];

        int total = 0;
        for (int i = 0; i < nums.length; i++) {
            total += backtrack(nums, target - nums[i], memo);
        }
        memo[target] = total;
        return total;
    }
}
