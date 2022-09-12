package com.company;

import java.util.*;

public class StackExamples {

    //Author: Anand
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {

        Map<Integer, Integer> map = new HashMap<>();
        Stack<Integer> stk = new Stack<>();

        for (int i = nums2.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums2[stk.peek()] <= nums2[i]) stk.pop();

            if (!stk.isEmpty()) map.put(nums2[i], nums2[stk.peek()]);
            else map.put(nums2[i], -1);

            stk.push(i);
        }


        int[] ans = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) ans[i] = map.getOrDefault(nums1[i], -1);

        return ans;
    }

    //Author: Anand
    public int[] nextGreaterElements(int[] nums) {

        Map<Integer, List<Integer>> map = new HashMap<>();

        java.util.Stack<Integer> stk = new java.util.Stack<>();

        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[stk.peek()] <= nums[i]) stk.pop();

            if (!map.containsKey(nums[i])) map.put(nums[i], new ArrayList<>());

            if (!stk.isEmpty()) map.get(nums[i]).add(nums[stk.peek()]);

            stk.push(i);
        }

        int[] ans = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i]) && map.get(nums[i]).size() > 0) {
                ans[i] = map.get(nums[i]).get(map.get(nums[i]).size() - 1);
                map.get(nums[i]).remove(map.get(nums[i]).size() - 1);
            } else ans[i] = Integer.MIN_VALUE;
        }


        Map<Integer, Boolean> cm = new HashMap<>();

        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[stk.peek()] <= nums[i]) stk.pop();

            if (!map.containsKey(nums[i])) map.put(nums[i], new ArrayList<>());

            if (!stk.isEmpty() && !cm.containsKey(nums[i])) {
                map.get(nums[i]).add(nums[stk.peek()]);
                cm.put(nums[i], true);
            }

            stk.push(i);
        }

        for (int i = 0; i < nums.length; i++) {
            if (ans[i] != Integer.MIN_VALUE) continue;

            if (map.containsKey(nums[i]) && map.get(nums[i]).size() > 0)
                ans[i] = map.get(nums[i]).get(map.get(nums[i]).size() - 1);
            else ans[i] = -1;
        }

        return ans;
    }
}

