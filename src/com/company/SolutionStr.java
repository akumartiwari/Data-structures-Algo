package com.company;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

class SolutionStr {
    /*
     * PROBLEM: Build Frequency Map (Helper)
     * Build a frequency count map from an integer array.
     *
     * ALGORITHM: HashMap population
     * TC: O(n) | SC: O(n)
     */
    // Function to return the updated frequency map
    // for the array passed as argument
    static HashMap<Integer, Integer> getFrequencyMap(int arr[]) {
        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            if (hashMap.containsKey(arr[i])) {
                hashMap.put(arr[i], hashMap.get(arr[i]) + 1);
            } else {
                hashMap.put(arr[i], 1);
            }
        }
        return hashMap;
    }

    /*
     * PROBLEM: Check and Consume Digit from Map (Helper)
     * Check if a digit exists in the frequency map; if so, decrement and return true.
     *
     * ALGORITHM: HashMap lookup and update
     * TC: O(1) | SC: O(1)
     */
    // Function that returns true if the passed digit is present
    // in the map after decrementing it's frequency by 1
    static boolean hasDigit(HashMap<Integer, Integer> hashMap, int digit) {

        // If map contains the digit
        if (hashMap.containsKey(digit) && hashMap.get(digit) > 0) {

            // Decrement the frequency of the digit by 1
            hashMap.put(digit, hashMap.get(digit) - 1);

            // True here indicates that the digit was found in the map
            return true;
        }

        // Digit not found
        return false;
    }

    /*
     * PROBLEM: Largest Time for Given Digits (LeetCode 949)
     * Given 4 digits, return the largest 24-hour time that can be made, or "" if impossible.
     *
     * ALGORITHM: Greedy with permutation fallback
     * TC: O(1) (at most 4! = 24 permutations) | SC: O(1)
     */
    public String largestTimeFromDigits(int[] input) {
        int iszero = 0;
        for (int a : input) {
            if (a == 0) iszero++;
        }

        if (iszero != 1) {

            HashMap<Integer, Integer> hashMap = getFrequencyMap(input);
            int i;
            boolean flag;
            String time = "";

            flag = false;

            // First digit of hours can be from the range [0, 2]
            for (i = 2; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }

            // If no valid digit found
            if (!flag) {
                return "";
            }

            flag = false;

            // If first digit of hours was chosen as 2 then
            // the second digit of hours can be
            // from the range [0, 3]
            if (time.charAt(0) == '2') {
                for (i = 3; i >= 0; i--) {
                    if (hasDigit(hashMap, i)) {
                        flag = true;
                        time += i;
                        break;
                    }
                }
            }

            // Else it can be from the range [0, 9]
            else {
                for (i = 9; i >= 0; i--) {
                    if (hasDigit(hashMap, i)) {
                        flag = true;
                        time += i;
                        if (hasDigit(hashMap, 0)) time = "0" + time;
                        break;
                    }
                }
            }
            if (!flag) {
                return "";
            }

            // Hours and minutes separator
            time += ":";

            flag = false;

            // First digit of minutes can be from the range [0, 5]
            for (i = 5; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }
            if (!flag) {
                return "";
            }

            flag = false;

            // Second digit of minutes can be from the range [0, 9]
            for (i = 9; i >= 0; i--) {
                if (hasDigit(hashMap, i)) {
                    flag = true;
                    time += i;
                    break;
                }
            }
            if (!flag) {
                return "";
            }

            // Return the maximum possible time
            return time;
        } else {
            String largestTime = "00:00";
            String str = input[0] + "" + input[1] + "" + input[2] + "" + input[3];
            List<String> times = new ArrayList<>();
            permutation(str, times);
            Collections.sort(times, Collections.reverseOrder());
            for (String t : times) {
                int hours = Integer.parseInt(t) / 100;
                int minutes = Integer.parseInt(t) % 100;
                if (hours < 24 && minutes < 60) {
                    if (hours < 10 && minutes < 10) {
                        largestTime = "0" + hours + ":0" + minutes;
                    } else if (hours < 10) {
                        largestTime = "0" + hours + ":" + minutes;
                    } else if (minutes < 10) {
                        largestTime = hours + ":0" + minutes;
                    } else {
                        largestTime = hours + ":" + minutes;
                    }
                }
            }
            return largestTime;
        }

    }

    /*
     * PROBLEM: Generate All Permutations (Helper)
     * Generate all permutations of string str and collect them in list.
     *
     * ALGORITHM: Recursive Backtracking
     * TC: O(n! * n) | SC: O(n! * n)
     */
    public static void permutation(String str, List<String> list) {
        permutation("", str, list);
    }

    /*
     * PROBLEM: Generate Permutations Recursive (Helper)
     * Recursively build permutations by choosing each character as prefix.
     *
     * ALGORITHM: Recursive Backtracking
     * TC: O(n! * n) | SC: O(n)
     */
    private static void permutation(String prefix, String str, List<String> list) {
        int n = str.length();
        if (n == 0) list.add(prefix);
        else {
            for (int i = 0; i < n; i++)
                permutation(prefix + str.charAt(i), str.substring(0, i) + str.substring(i + 1, n), list);
        }
    }

    int[] nums;

    int[] prefix = new int[nums.length];

    /*
     * PROBLEM: Range Sum Query (LeetCode 307) initialization
     * Initialize the prefix sum array for range sum queries.
     *
     * ALGORITHM: Prefix Sum initialization
     * TC: O(n) | SC: O(n)
     */
    public void NumArray(int[] nums) {
        this.nums = nums;
        int index = 0;
        for (int n : nums) {
            prefix[index] = n;
            index++;
        }

        return;
    }

    /*
     * PROBLEM: Range Sum Query Update (LeetCode 307)
     * Update element at index and adjust prefix sum accordingly.
     *
     * ALGORITHM: Prefix Sum point update
     * TC: O(1) | SC: O(1)
     */
    public void update(int index, int val) {
        this.nums[index] = val;
        prefix[index] -= this.nums[index];
        prefix[index] += val;
        return;
    }

    /*
     * PROBLEM: Range Sum Query (LeetCode 307)
     * Return the sum of elements between indices left and right (inclusive).
     *
     * ALGORITHM: Prefix Sum query
     * TC: O(1) | SC: O(1)
     */
    public int sumRange(int left, int right) {
        return prefix[right] - prefix[left];
    }
    /**
     * Your NumArray object will be instantiated and called as such:
     * NumArray obj = new NumArray(nums);
     * obj.update(index,val);
     * int param_2 = obj.sumRange(left,right);
     */
}
