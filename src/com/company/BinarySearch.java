package com.company;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    /*
       Return the length of longest increasing subsequence
       For eg.
       Input = [1 5 2 8 9]
       Output = 4 ({1 5 8 9} | {1 2 8 9})
       The idea is to use binarySearch based solution
       Steps:-
       - Iterate through array and check if current is larger than previous selected element
       - if yes then simply store it next to prev
       - Else BS for index at which it can  be inserted (lower_bound function), Replace the new element at that index
       - Update length if LIS
       - Return length

       TC = O(nlogn), SC = O(n)
     */
    private int lower(int[] arr, int target) {
        if (arr == null || arr.length == 0) {
            return 0;
        }
        int l = 0;
        int r = arr.length - 1;
        if (target <= arr[0]) {
            return 0;
        }
        if (target > arr[r]) {
            return -1;
        }
        while (l < r) {
            int m = l + (r - l) / 2;

            if (arr[m] >= target) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return r;
    }

    public int lengthOfLIS(int[] arr) {
        List<Integer> lis = new ArrayList<>();
        int len = 0;
        for (int a : arr) {

            if (lis.size() == 0) {
                lis.add(a);
                len++;
                continue;
            }
            if (a > lis.get(lis.size() - 1)) {
                lis.add(a);
                len++;
            } else {
                int idx = lower(lis.stream().mapToInt(x -> x).toArray(), a);
                lis.set(idx, a);
            }
        }
        return len;
    }

    /*
      Intution
    - Sort the array ,
    - choose every pivot from end to begin where element is less than target and make this = target
    - keep count of element >= target
    - Distribute remaining newFlowers from begining till index where it can be afforded
    - Check for max
     TC = O(nlogn), SC = O(n)
     TODO: Understand this
     */
    public long maximumBeauty(int[] flowers, long newFlowers, int target, int full, int partial) {
        int len = flowers.length;
        long cumulativeCostArray[] = new long[len];
        Arrays.sort(flowers);

        for (int i = 1; i < len; i++) {
            cumulativeCostArray[i] = cumulativeCostArray[i - 1] + (long) i * (flowers[i] - flowers[i - 1]);
        }

        long max = 0;
        int i;
        int countComplete = 0;
        for (i = len - 1; i >= 0; i--) {
            if (flowers[i] < target)
                break;
            countComplete++;
        }

        if (countComplete == len)
            return ((countComplete * (long) full));

        int id = binarySearchCumulativeCost(cumulativeCostArray, newFlowers, 0, i);
        max = currentPartitionCost(flowers, newFlowers, target, full, partial, cumulativeCostArray, max, countComplete, id);

        for (int j = i; j >= 0; j--) {
            newFlowers = newFlowers - (target - flowers[j]);
            if (newFlowers < 0)
                break;
            countComplete++;
            if (j == 0) {
                max = Math.max(max, countComplete * (long) full);
                break;
            }
            id = binarySearchCumulativeCost(cumulativeCostArray, newFlowers, 0, j - 1);
            max = Math.max(max, currentPartitionCost(flowers, newFlowers, target, full, partial, cumulativeCostArray, max,
                    countComplete, id));
        }

        return max;
    }

    private long currentPartitionCost(int[] flowers, long newFlowers, int target, int full, int partial,
                                      long[] costArray, long max, int countComplete, int id) {
        if (id >= 0) {
            long rem = (newFlowers - costArray[id]);
            long minToAddFromRem = rem / (id + 1);
            max = ((countComplete * (long) full) + ((Math.min(target - 1, minToAddFromRem + flowers[id])) * (long) partial));
        }
        return max;
    }

    public int binarySearchCumulativeCost(long cost[], long num, int s, int e) {
        int i = s, j = e;
        while (i < j) {
            int mid = (i + j) / 2;
            if (cost[mid] <= num) {
                i = mid + 1;
            } else {
                j = mid;
            }
        }
        return cost[i] <= num ? i : (i - 1);
    }
}
