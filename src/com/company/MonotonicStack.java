package com.company;

import java.util.List;
import java.util.Stack;

public class MonotonicStack {


    /*
    Input: nums = [1,3,4,3,1], threshold = 6
    Output: 3
    Explanation: The subarray [3,4,3] has a size of 3, and every element is greater than 6 / 3 = 2.
    Note that this is the only valid subarray.

    Algorithm :-
    - ArrayDeque is used a stack.
    - Iterate through all elements of array
    - check if stack top consist of greater element OR Last element it means all element to left  are also greater
     It means it can be a valid subArray
    - To check if its valid we need get smallest element *  size of subArray > threshold --> return Valid Subarray
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


    //Author: Anand
    /*
     Input: nums = [5,3,4,4,7,3,6,11,8,5,11]
    Output: 3
    Explanation: The following are the steps performed:
    - Step 1: [5,3,4,4,7,3,6,11,8,5,11] becomes [5,4,4,7,6,11,11]
    - Step 2: [5,4,4,7,6,11,11] becomes [5,4,7,11,11]
    - Step 3: [5,4,7,11,11] becomes [5,7,11,11]
    [5,7,11,11] is a non-decreasing array. Therefore, we return 3.

    Algorithm :-
    The Idea is to calculate distance between the closest left strictly greater element for every index i
    Return maximum of distance (As distance refers to numbers of steps needed to remove that element )
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

    // monotonic increasing stack
    // runtime o(n)
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

    // TODO :- Need to solved again
    public int sumSubarrayMins(int[] a) {
        int sum = 0, n = a.length;
        int[] left = new int[n], right = new int[n];
        Stack<Integer> s = new Stack<>();
        //for each i, first right side number less than me
        //monotonically increasing stack
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
        //for each i, first left side number less than me
        //monotonically increasing stack from back
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


    //TODO: Complete this
    class Solution {
        private static final int mod = 1000000007;

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
    //Use monotonic stack
    public long maximumSumOfHeights(List<Integer> maxHeights) {
        return 0L;
    }
}
