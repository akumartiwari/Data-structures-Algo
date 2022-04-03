package com.company;

public class BinarySearch {
    /*
    Input: candies = [5,8,6], k = 3
    Output: 5
   */
    // Author: Anand
    public static long binarySearch(int n, long k, long[] arr, long max) {
        long low = 1;
        long high = max;
        long mid;
        long res = 0;

        while (high >= low) {
            mid = low + (high - low) / 2;
            if (canDistribute(mid, n, k, arr)) {
                low = mid + 1;
                res = mid;
            } else {
                high = mid - 1;
            }
        }
        return res;
    }

    public static boolean canDistribute(long val, int n, long k, long[] arr) {
        if (k == 1) {
            return true;
        }
        long peopleServed = 0;
        for (int i = n - 1; i >= 0; i--) {
//this is the number of people who can get candy
            peopleServed += arr[i] / val;
        }
        if (peopleServed >= k) {
            return true;
        }
        return false;
    }

    public int maximumCandies(int[] candies, long k) {
        int n = candies.length;
        long max = Integer.MIN_VALUE;
        long[] arr = new long[n];

        for (int i = 0; i < n; i++) {
            arr[i] = (long) (candies[i]);
            //we need to find max for the upper bound in the binary search.
            max = Math.max(max, arr[i]);
        }

        return (int) binarySearch(n, k, arr, max);
    }
}
