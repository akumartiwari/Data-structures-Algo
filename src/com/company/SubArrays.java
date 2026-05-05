package com.company;

import java.util.HashMap;import java.util.*;
import java.util.stream.Collectors;

class SubArrays {
    /*
     * PROBLEM: Entry point (Main)
     * Demonstrates subarray counting utilities.
     *
     * ALGORITHM: N/A
     * TC: O(n) | SC: O(n)
     */
    public static void main(String[] args) {
        int[] input = {3, 5, 2, 7, 8, 9, 11, 2, 5, 8, 3};
        SubArrays subArrays = new SubArrays();
        System.out.println(subArrays.count(input, 9).stream().collect(Collectors.toSet()));

    }


    // To create tuples in  java we have top create their oops class
    static class pair {
        int index;
        int key;
        String value;

        pair(int index) {
            this.index = index;
            this.key = 0;
            this.value = null;
        }

    }

    static class sort implements Comparable<pair> {
        pair p;

        @Override
        public int compareTo(pair pair) {
            return p.key - pair.key;
        }
    }

    /*
     * PROBLEM: Build Map from Array (Helper)
     * Build a HashMap/PQ structure from input array elements.
     *
     * ALGORITHM: HashMap + Priority Queue
     * TC: O(n) | SC: O(n)
     */
    private Map<Integer, pair> map(int[] input, int k) {
        int n = input.length;
        if (n == 0) return new HashMap<>();
        Map<Integer, pair> map = new HashMap<>();
        pair p = new pair(n);
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>(Comparator.comparingInt(a -> a));
        pq.offer(0);
        for (int e : input) {
            map.put(e, map.getOrDefault(e, new pair(e % n)));
            n /= e;
        }
        return map;
    }

    /*
     * PROBLEM: Count Elements Less Than K (Helper)
     * Return list of all elements in input that are less than k.
     *
     * ALGORITHM: Linear Scan
     * TC: O(n) | SC: O(n)
     */
    private List<Integer> count(int[] input, int k) {
        int count = 0;
        int n = input.length;
        if (n == 0) return new ArrayList<>();
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (input[i] < k) {
                while (i < n && input[i] < k) {
                    ans.add(input[i]);
                    count++;
                    i++;
                }
            }
        }
        return ans;
    }

    /*
     * PROBLEM: Find Subarrays with Sum Less Than K (Helper)
     * Return elements of each maximal contiguous block where all elements < k.
     *
     * ALGORITHM: Linear Scan
     * TC: O(n) | SC: O(n)
     */
    //  Total sum of subarray must be less than k
    private List<Integer> subarrays(int[] input, int k) {
        int n = input.length;
        if (n == 0) return new ArrayList<>();
        int sum = 0;
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (input[i] < k) {
                while (i < n && input[i] < k) {
                    sum += input[i];
                    i++;
                    ans.add(input[i]);
                }
            }
            sum = 0;
        }
        return ans;
    }

    /*
     * PROBLEM: All Possible Subarrays Less Than K (Helper)
     * Return elements from all contiguous sub-blocks where all values < k.
     *
     * ALGORITHM: Linear Scan
     * TC: O(n) | SC: O(n)
     */
// all  possible total sum of subarray must be less than k
//    private

    private List<Integer> allPossibleSubarrays(int[] input, int k) {
        int n = input.length;
        if (n == 0) return new ArrayList<>();
        int sum = 0;
        List<Integer> ans = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (input[i] < k) {
                while (input[i] < k) {
                    sum += input[i];
                    i++;
                    ans.add(input[i]);
                }

            }
            sum = 0;
        }
        return ans;
    }


    /*
     * PROBLEM: Count Subarrays from Range (Helper)
     * Count total subarrays in a window of size (right - left) using arithmetic formula.
     *
     * ALGORITHM: Arithmetic formula n*(n+1)/2
     * TC: O(1) | SC: O(1)
     */
    int calc(int left, int right) {
        int n = right - left;
        return n * (n + 1) / 2;
    }

    /*
     * PROBLEM: Find Subarrays Less Than K (Helper)
     * Count all subarrays where every element is less than k using calc helper.
     *
     * ALGORITHM: Brute Force with arithmetic formula
     * TC: O(n) | SC: O(1)
     */
    int findSubArrays(int[] arr, int k) {
        int ans = 0, sum = 0, ptr = 0;
        while (ptr < arr.length) {
            if (arr[ptr] < k) {
                int ptrForward = ptr;
                while (arr[ptrForward] < k)
                    ptrForward++;
                sum += calc(ptr, ptrForward);
            } else {
                ptr++;
            }
        }
        return ans;
    }
}
