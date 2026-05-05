package com.company;

import java.util.*;

public class MonotonicStack {

    /*
     * PROBLEM: Subarray With Elements Greater Than Varying Threshold (LeetCode 2334)
     * Find the length of the smallest subarray where every element > threshold/subarray_length; return -1 if none.
     *
     * ALGORITHM: Monotonic Decreasing Stack
     * TC: O(N) | SC: O(N)
     *
     * Each pop identifies the largest window for which that element is the minimum.
     * Window width = i (empty stack) or i - stack.peek() - 1.
     */
    public int validSubarraySize(int[] nums, int threshold) {
        Stack<Integer> stack = new Stack<>();
        int subArraySize = -1;
        for (int i = 0; i <= nums.length; i++) {
            while (!stack.isEmpty() && (i == nums.length || nums[stack.peek()] > nums[i])) {
                int height = nums[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                if (height * width > threshold) return width;
            }
            stack.push(i);
        }
        return subArraySize;
    }


    /*
     * PROBLEM: Steps to Make Array Non-decreasing (LeetCode 2289)
     * Repeatedly remove elements smaller than their left neighbour; return the number of steps until non-decreasing.
     *
     * ALGORITHM: Monotonic Stack (right to left), tracking [value, steps] pairs
     * TC: O(N) | SC: O(N)
     *
     * For each element, count how many right-side elements will be removed before it.
     * Answer = max steps value ever pushed onto the stack.
     */
    public int totalSteps(int[] nums) {
        int ans = 0;
        Stack<int[]> stk = new Stack<>();

        for (int i = nums.length - 1; i >= 0; i--) {
            if (stk.isEmpty() || stk.peek()[0] >= nums[i]) {
                stk.push(new int[]{nums[i], 0});
            } else {
                int count = 0;
                while (!stk.isEmpty() && stk.peek()[0] < nums[i]) {
                    count++;
                    int[] item = stk.pop();
                    count = Math.max(count, item[1]);
                }
                stk.push(new int[]{nums[i], count});
                ans = Math.max(ans, count);
            }
        }
        return ans;
    }

    /*
     * PROBLEM: Count Valid Subarrays (GFG)
     * Count subarrays where the leftmost element is the minimum of the subarray.
     *
     * ALGORITHM: Monotonic Increasing Stack (contribution counting)
     * TC: O(N) | SC: O(N)
     *
     * When nums[top] < nums[i], pop and add (i - popped_index) subarrays.
     * Remaining stack elements each contribute (n - index) subarrays.
     */
    public int validSubarraysStack(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
                count += (i - stack.pop());
            }
            stack.push(i);
        }
        while (!stack.isEmpty()) {
            count += nums.length - stack.pop();
        }
        return count;
    }

    private static final int mod = 1_000_000_000 + 7;

    /*
     * PROBLEM: Sum of Subarray Minimums (LeetCode 907)
     * Find the sum of min(subarray) for every contiguous subarray, modulo 1e9+7.
     *
     * ALGORITHM: Contribution Technique + two-pass Monotonic Stack
     * TC: O(N) | SC: O(N)
     *
     * For each a[i]: left[i] = span to the left where a[i] is minimum (non-strict),
     * right[i] = span to the right where a[i] is minimum (strict).
     * Contribution = a[i] * (left[i]+1) * (right[i]+1).
     */
    // TODO :- Need to solved again
    public int sumSubarrayMins(int[] a) {
        int sum = 0, n = a.length;
        int[] left = new int[n], right = new int[n];
        Stack<Integer> s = new Stack<>();
        for (int i = 0, j = 0; i < n; i++) {
            while (!s.isEmpty() && a[s.peek()] > a[i]) {
                j = s.pop();
                right[j] = i - 1 - (j + 1) + 1;
            }
            s.push(i);
        }
        while (!s.isEmpty()) {
            int j = s.pop();
            right[j] = n - 1 - (j + 1) + 1;
        }
        for (int i = n - 1, j = 0; i >= 0; i--) {
            while (!s.isEmpty() && a[s.peek()] >= a[i]) {
                j = s.pop();
                left[j] = j - 1 - (i + 1) + 1;
            }
            s.push(i);
        }
        while (!s.isEmpty()) {
            int j = s.pop();
            left[j] = (j - 1) + 1;
        }
        for (int i = 0; i < n; i++)
            sum = sum % mod + ((a[i] * (left[i] + 1)) % mod) * ((right[i] + 1) % mod) % mod;
        return sum % mod;
    }


    /*
     * PROBLEM: Maximum Subarray Min-Product (LeetCode 1856)
     * Find the maximum value of min(subarray) * sum(subarray) over all non-empty subarrays; return mod 1e9+7.
     *
     * ALGORITHM: Prefix Sum + Monotonic Stack (histogram-style)
     * TC: O(N) | SC: O(N)
     *
     * For each popped element (minimum of its window): window_sum = pref_sum[i] - pref_sum[peek].
     * Maximise min * window_sum across all pops.
     */
    //TODO: Complete this
    class Solution {
        private static final int mod = 1000000007;

        /*
         * PROBLEM: Maximum Subarray Min-Product (LeetCode 1856)
         * Find the maximum value of min(subarray) * sum(subarray) over all non-empty subarrays; return mod 1e9+7.
         *
         * ALGORITHM: Prefix Sum + Monotonic Stack (histogram-style)
         * TC: O(N) | SC: O(N)
         *
         * For each popped element (minimum of its window): window_sum = pref_sum[i] - pref_sum[peek].
         * Maximise min * window_sum across all pops.
         */
        public int maxSumMinProduct(int[] nums) {
            int ind = 0;
            int[] pref_sum = new int[nums.length];
            for (int num : nums) {
                pref_sum[ind] = ind == 0 ? num : pref_sum[ind - 1] + num;
                ind++;
            }
            Stack<Integer> stack = new Stack<>();
            int subArraySize = Integer.MIN_VALUE;
            for (int i = 0; i <= nums.length; i++) {
                while (!stack.isEmpty() && (i == nums.length || nums[stack.peek()] < nums[i])) {
                    int height = nums[stack.pop()];
                    int width = stack.isEmpty() ? i : pref_sum[i] - pref_sum[stack.peek()];
                    subArraySize = Math.max(subArraySize, (height * width) % mod);
                }
                stack.push(i);
            }
            return subArraySize;
        }
    }

    /*
     * PROBLEM: Maximum Sum of Heights (LeetCode 2865)
     * Choose one peak index; build a mountain array ≤ maxHeights that maximises the total sum.
     *
     * ALGORITHM: Two-pass Monotonic Stack (left pass + right pass)
     * TC: O(N) | SC: O(N)
     *
     * left[i] = max sum contribution from left side when i is peak.
     * right[i] = max sum contribution from right side when i is peak.
     * Answer = max over all i of (left[i] + right[i] - maxHeights[i]).
     */
    //Use monotonic stack
    public long maximumSumOfHeights(List<Integer> maxHeights) {
        return 0L;
    }

    /*
     * PROBLEM: Max Value After Jumping Left or Right (Helper)
     * For each index, compute the maximum reachable value using prefix-max chaining to the right.
     *
     * ALGORITHM: Prefix Max + Suffix Min arrays, right-to-left result propagation
     * TC: O(N) | SC: O(N)
     *
     * If pref[i] > suff[i+1], the prefix max at i can chain to the next result rightward.
     */
    public int[] maxValue(int[] nums) {
        int n = nums.length;
        int[] pref = new int[n], suff = new int[n], res = new int[n];
        int max = -1, min = Integer.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, nums[i]);
            pref[i] = max;
            min = Math.min(min, nums[n - 1 - i]);
            suff[i] = min;
        }

        res[n - 1] = pref[n - 1];
        for (int i = n - 1; i >= 0; i--) {
            res[i] = pref[i];
            if (pref[i] > suff[i + 1]) res[i] = res[i + 1];
        }
        return res;
    }

}
