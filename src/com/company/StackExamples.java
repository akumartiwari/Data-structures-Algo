package com.company;

import java.util.HashMap;
import java.util.*;

public class StackExamples {

    /*
     * PROBLEM: Daily Temperatures (LeetCode 739)
     * Return answer[i] = number of days to wait after day i for a warmer temperature.
     *
     * ALGORITHM: Monotonic Stack (left to right, decreasing by temperature)
     * TC: O(N) | SC: O(N)
     *
     * Maintain a stack of INDICES whose "next warmer day" hasn't been found yet.
     * For each day i:
     *   1. While stack non-empty AND temperatures[stack.top] < temperatures[i]:
     *        pop idx → answer[idx] = i - idx
     *   2. Push i onto the stack.
     * Any index remaining in the stack at the end has no warmer future day → stays 0.
     *
     * Example: temperatures = [73,74,75,71,69,72,76,73] → [1,1,4,2,1,1,0,0]
     */
    public int[] dailyTemperatures(int[] temperatures) {
        // base cases
           /*
            int n = temperatures.length;

            int[] ans = new int[n];
            // iterate throughout all element
            for (int i = 0; i < n; i++) {
                int count = 0;
                for (int j = i + 1; j < n; j++) {
                    count++;
                    if (temperatures[j] > temperatures[i]) ans[i] = count;
                }
            }

            return ans;
        }
        */


        int n = temperatures.length;
        int[] nextWarmerday = new int[n];
        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack

        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday[idx] = idx - i;
            }
            stk.push(i);
        }
        return nextWarmerday;
    }

    /*
     * PROBLEM: Next Greater Element I (LeetCode 496)
     * For each element of nums1 (a subset of nums2), find the next greater element in nums2.
     *
     * ALGORITHM: Monotonic Stack + HashMap (right to left on nums2)
     * TC: O(N+M) | SC: O(N)
     *
     * Pre-process nums2 with a monotonic stack to build value→nextGreater map.
     * Then answer each nums1 query in O(1) via the map.
     *
     * Example: nums1=[4,1,2], nums2=[1,3,4,2] → [-1,3,-1]
     */
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



    /*
     * PROBLEM: Next Greater Element II - Alternate (LeetCode 503)
     * For a circular array with possible duplicates, find the next greater element for each position.
     *
     * ALGORITHM: Two-pass Monotonic Stack + HashMap<value, List<nextGreater>>
     * TC: O(N) | SC: O(N)
     *
     * Pass 1 (right to left): Build value→[nextGreater] map for non-circular part.
     * Pass 2 (right to left): Handle circular wrap-around using a "visited" guard map.
     * Slots still set to MIN_VALUE after pass 1 are filled from pass 2 results.
     *
     * Example: nums=[1,2,1] → [2,-1,2]
     */
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
     * PROBLEM: Next Greater Element II (LeetCode 503)
     * Given a circular integer array, find the next greater element for every position.
     *
     * ALGORITHM: Monotonic Stack (traverse array twice using i % n)
     * TC: O(N) | SC: O(N)
     *
     * Iterate i from 2N-1 down to 0; use (i % n) to simulate circular wrap-around.
     * Maintain a monotonic decreasing stack of indices.
     *
     * Example: nums=[1,2,1] → [2,-1,2]
     */
    public int[] nextGreaterElementsOptimised(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Stack<Integer> stk = new Stack<>(); // to store next greater elements in stack
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[i % n] >= nums[stk.peek()]) stk.pop();

            ans[i % n] = stk.isEmpty() ? -1 : nums[stk.peek()];
            stk.push(i % n);
        }
        return ans;
    }


    /*
     * PROBLEM: Find the Second Greater Element (LeetCode 2454)
     * Return answer[i] = second distinct greater integer to the right of nums[i], or -1.
     *
     * ALGORITHM: Monotonic Stack + Jump Pointer (right to left)
     * TC: O(N) | SC: O(N)
     *
     * Step 1: Build map[i] = index of first-greater-element to the right (monotonic stack).
     * Step 2: For each i, start at sgi = map[i]+1; jump via map[sgi] while nums[sgi] <= nums[i].
     *
     * Example: nums=[2,4,0,9,6] → [9,6,6,-1,-1]
     */
    public int[] secondGreaterElement(int[] nums) {

        Map<Integer, Integer> map = new HashMap<>(); // ind , NG ind
        Stack<Integer> stk = new Stack<>();

        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[stk.peek()] <= nums[i]) stk.pop();

            if (!stk.isEmpty()) map.put(i, stk.peek());
            else map.put(i, -1);

            stk.push(i);
        }

        int[] ans = new int[nums.length];
        Arrays.fill(ans, -1);

        for (int i = 0; i < nums.length - 2; i++) {
            if (map.get(i) == -1) continue;

            int fgi = map.get(i);

            int sgi = fgi + 1;

            // For eg. if nums[sgi] is <= current element then all the remaining smaller (smaller than sgi) elements will be smaller than current.
            // Hence, we can directly jump to next greater of sgi ie, map.get(sgi)

            while (sgi != -1 && sgi < nums.length && nums[sgi] <= nums[i]) sgi = map.get(sgi);

            if (sgi < nums.length && sgi != -1) ans[i] = nums[sgi];

        }
        return ans;
    }


    /*
     * PROBLEM: Using Robot to Make Lexicographically Smallest String (LeetCode 2434)
     * Use a robot with a string buffer and stack to produce the lexicographically smallest result.
     *
     * ALGORITHM: Greedy + Monotonic Stack + frequency array
     * TC: O(N) | SC: O(N)
     *
     * Push characters onto a stack while tracking remaining frequencies.
     * Pop and append to result whenever the stack top is <= smallest remaining character.
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

    /*
     * PROBLEM: Has Smaller Character in Remaining (Helper)
     * Check if any character smaller than c still has remaining occurrences.
     *
     * ALGORITHM: Linear scan of frequency array
     * TC: O(26) | SC: O(1)
     */
    private boolean hasSmaller(char c, int[] freq) {
        for (int i = 0; i < (c - 'a'); ++i) if (freq[i] > 0) return true;
        return false;
    }


    /*
     * PROBLEM: Count Warmer Days Variant (LeetCode 739 Variant)
     * Variant of Daily Temperatures: collect all wait-day counts into a list as warmer days are discovered.
     *
     * ALGORITHM: Monotonic Stack (left to right, collect differences to list)
     * TC: O(N) | SC: O(N)
     *
     * Same stack logic as dailyTemperatures; instead of storing per-index, append each gap to a list.
     * Indices with no warmer day remain produce no list entry.
     *
     * Example: temperatures=[73,74,75,71,72] → list=[1,1,1]
     */
    private static List<Integer> countcolderDays1(int[] temperatures) {
        int n = temperatures.length;
        List<Integer> nextWarmerday = new ArrayList<>();

        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack
        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday.add(i - idx);
            }
            stk.push(i);
        }
        return nextWarmerday;
    }

}

