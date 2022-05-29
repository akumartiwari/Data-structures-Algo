package com.company;

import java.util.Stack;

public class MonotonicStack {
    //Author: Anand
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
                    if (item[1] > count) count = item[1];
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

    private static final int mod = 1000000007;

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
            left[j] = j - 1 - (-1 + 1) + 1;
        }
        for (int i = 0; i < n; i++)
            sum = sum % mod + (a[i] * (left[i] + 1) * (right[i] + 1)) % mod;
        return sum % mod;
    }
}
