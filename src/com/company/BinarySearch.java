package com.company;

import java.util.*;

public class BinarySearch {

    // Author: Anand
    // TC = O(nlogn)
    public long minimumTime(int[] time, int totalTrips) {
        long l = 0, h = 1_00_000_000_000_000L;
        while (l < h) {
            long mid = l + (h - l) / 2;
            long ans = 0;
            for (int t : time) {
                ans += mid / t;
            }
            if (ans < totalTrips) l = mid + 1;
            else h = mid;
        }
        return l;
    }

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
        return peopleServed >= k;
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
Input: spells = [5,1,3], potions = [1,2,3,4,5], success = 7
Output: [4,0,3]
Explanation:
- 0th spell: 5 * [1,2,3,4,5] = [5,10,15,20,25]. 4 pairs are successful.
- 1st spell: 1 * [1,2,3,4,5] = [1,2,3,4,5]. 0 pairs are successful.
- 2nd spell: 3 * [1,2,3,4,5] = [3,6,9,12,15]. 3 pairs are successful.
Thus, [4,0,3] is returned.
*/
    //Author: Anand
    Map<Integer, int[]> duplicates;

    public int[] successfulPairs(int[] spells, int[] potions, long success) {
        int[] pairs = new int[spells.length];
        Arrays.sort(potions);
        int ind = 0;
        duplicates = new HashMap<>();
        for (int i = 0; i < potions.length; i++) {
            if (duplicates.containsKey(potions[i])) {
                int[] idx = duplicates.get(potions[i]);
                duplicates.put(potions[i], new int[]{idx[0], i});
            } else duplicates.put(potions[i], new int[]{i});
        }

        for (int s : spells) {
            int idx = bs(potions, (long) Math.ceil((double) success / s));
            if (idx >= 0) {
                pairs[ind++] = potions.length - idx;
            } else {
                pairs[ind++] = 0;
            }
        }
        return pairs;
    }

    private int bs(int[] potions, long value) {
        int l = 0, h = potions.length - 1;
        while (l < h) {
            int mid = l + (h - l) / 2;
            if (value < potions[mid]) {
                h = mid;
            } else if (value > potions[mid]) {
                l = mid + 1;
            } else if (value == potions[mid]) {
                return duplicates.get(potions[mid])[0];
            }
        }

        return potions[l] < value ? -1 : l;
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
        long[] cumulativeCostArray = new long[len];
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

    /*
    Input: rectangles = [[1,1],[2,2],[3,3]], points = [[1,3],[1,1]]
    Output: [1,3]
    Explanation:
    The first rectangle contains only the point (1, 1).
    The second rectangle contains only the point (1, 1).
    The third rectangle contains the points (1, 3) and (1, 1).
    The number of rectangles that contain the point (1, 3) is 1.
    The number of rectangles that contain the point (1, 1) is 3.
    Therefore, we return [1, 3].
     */
    //Author: Anand
    // TC = O(nlogn)
    public int[] countRectangles(int[][] rectangles, int[][] points) {
        TreeMap<Integer, List<Integer>> map = new TreeMap<>(); // k=y, v=[x]

        int max = Integer.MIN_VALUE;
        int[] ans = new int[points.length];

        for (int[] r : rectangles) {
            int key = r[1];
            int value = r[0];
            List<Integer> list = new ArrayList<>();
            if (map.containsKey(key)) list = map.get(key);
            list.add(value);
            map.put(key, list);
            max = Math.max(max, key);
        }

        for (int key : map.keySet()) Collections.sort(map.get(key));

        for (int i = 0; i < points.length; i++) {
            int key = points[i][0];
            int value = points[i][1];

            if (value > max) continue;
            int count = 0;

            // search for all x <= key
            for (int entry : map.subMap(value, max + 1).keySet())
                count += bs(map.get(entry), key);

            ans[i] = count;
        }

        return ans;
    }

    private int bs(List<Integer> xc, int key) {

        int l = 0, h = xc.size() - 1;
        int idx = -1;
        while (l <= h) {
            int m = l + (h - l) / 2;
            if (xc.get(m) >= key) {
                idx = m;
                h = m - 1;
            } else l = m + 1;
        }

        return idx < 0 ? 0 : xc.size() - idx;
    }
}
