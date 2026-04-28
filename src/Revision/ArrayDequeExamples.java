package com.company;

import java.util.ArrayDeque;
import java.util.Deque;

public class ArrayDequeExamples {

    /*
     * PROBLEM: Find Subarray with Elements Greater Than Varying Threshold (LeetCode 2334)
     * Find the minimum size of a subarray where every element is greater than threshold/size.
     *
     * ALGORITHM: Monotonic Stack
     * TC: O(n) | SC: O(n)
     */
    public int validSubarraySize(int[] nums, int threshold) {
        Deque<Integer> stack = new ArrayDeque<>(); // ArrayDeque is used a stack
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
}
