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

    /*
    Input: s = "zza"
    Output: "azz"
    Explanation: Let p denote the written string.
    Initially p="", s="zza", t="".
    Perform first operation three times p="", s="", t="zza".
    Perform second operation three times p="azz", s="", t="".
    */
    /*
    The first loop is to record the frequence of every character, the array freq,
    the second loop is to add every character into a stack,
    when adding the character into the stack, decreate the frequency of the character by one in the array freq,
    then the array freq is the frequency of every character in the rest of string.
    When adding one character from the top of the stack to the result, we check if there is one smaller character in the rest of the string,
    if there is, keep pushing the character of the rest of the string into the stack, if there is not,
    then add the top character into the result.
     */
    public String robotWithString(String s) {
        int[] freq = new int[26];
        for (char c : s.toCharArray()) freq[c - 'a']++;

        StringBuilder sb = new StringBuilder();
        Stack<Character> stack = new Stack<>();

        for (char c : s.toCharArray()) {
            stack.add(c);
            freq[c - 'a']--;

            while (!stack.isEmpty()) {
                char curr = stack.peek();
                if (hasSmaller(curr, freq)) break;
                sb.append(stack.pop());
            }
        }
        return sb.toString();
    }

    private boolean hasSmaller(char c, int[] freq) {
        for (int i = 0; i < (c - 'a'); ++i) if (freq[i] > 0) return true;
        return false;
    }


}

