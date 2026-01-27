package com.company;

import java.util.ArrayDeque;
import java.util.Deque;

public class ArrayDequeExamples {

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
