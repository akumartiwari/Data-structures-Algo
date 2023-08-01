package com.company;

import java.util.HashMap;
import java.util.HashMap;import java.util.*;

public class BinarySearch {

    class Solution {

        class Tuple2 {
            boolean result;
            boolean suffcient;

            Tuple2(boolean result, boolean suffcient) {
                this.result = result;
                this.suffcient = suffcient;
            }

            Tuple2() {
            }
        }

        class P {
            boolean res;
            int op;

            P(boolean res, int op) {
                this.res = res;
                this.op = op;
            }

            P() {
            }
        }

        public int minimumSize(int[] nums, int maxOperations) {
            int l = 1, h = Arrays.stream(nums).max().getAsInt();

            int ans = Integer.MAX_VALUE;
            while (l < h) {
                int mid = (l + h) / 2;
                Tuple2 tuple2 = minG(nums, maxOperations, mid);
                System.out.println(tuple2.result + ":" + tuple2.suffcient);
                if (tuple2.suffcient && mid > 2) {
                    h = mid - 1;
                    if (tuple2.result) ans = Math.min(ans, mid);
                } else if (tuple2.result) {
                    h = mid - 1;
                    ans = Math.min(ans, mid);
                } else if (!tuple2.suffcient && !tuple2.result) h = mid - 1;
                else l = mid + 1;
            }
            return ans;
        }

        private Tuple2 minG(int[] nums, int maxOperations, int target) {

            boolean success = true;
            for (int num : nums) {
                P pow = pow(num, target);
                maxOperations -= pow.op;
                if (target > num) return new Tuple2(false, false);
                if (!pow.res) success = false;
                if (maxOperations < 0) return new Tuple2(false, false);
            }

            return new Tuple2(success, maxOperations >= 0);
        }

        private P pow(int num, int target) {
            int cnt = 0;
            while (target < num) {
                target *= target;
                cnt++;
            }

            return new P(target == num, target == num ? cnt + 1 : cnt);
        }


    }


    // [4,2,7,6,9,14,12]
    /*
    Intuition
    BS:-
    The Aim is to find max index reachable. Consider any random index b/w { 0 to n-1} and check if it's reachable.
    if yes all index before that random-index must also be rechable and try the same in other half.

    To check reachability used greedy approach below-
    Greedy:-
    The idea is to use ladder whenever you have to make max height jump i.e. h[i+1]-h[i] is largest as possible untill all ladders exhausted.
    After that start using bricks and check if that index is reachable.

    Complexity:-
    Time complexity: O(KlogNlogK)
    Space complexity: O(N) -> max-heap
     */
    public int furthestBuilding(int[] heights, int bricks, int ladders) {
        int l = 0, h = heights.length - 1;
        int ans = 0;
        while (l <= h) {
            int mid = l + (h - l) / 2;
            if (canReach(heights, bricks, ladders, mid)) {
                l = mid + 1;
                ans = Math.max(ans, mid);
            } else h = mid - 1;
        }
        return ans;
    }

    private boolean canReach(int[] heights, int bricks, int ladders, int ind) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder()); // maxPQ
        for (int i = 0; i < ind; i++)
            if (heights[i + 1] - heights[i] > 0) pq.add(heights[i + 1] - heights[i]);

        while (!pq.isEmpty()) {
            int diff = pq.poll();
            if (ladders > 0) {
                ladders--;
                continue;
            }

            if (bricks > 0 && diff <= bricks) {
                bricks -= diff;
                continue;
            }
            return false;
        }

        return true;
    }

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
        if (k == 1) return true;
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
            arr[i] = candies[i];
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

    public int binarySearchCumulativeCost(long[] cost, long num, int s, int e) {
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

    /*
    Input: nums = [4,5,2,1], queries = [3,10,21]
    Output: [2,3,4]
    Explanation: We answer the queries as follows:
    - The subsequence [2,1] has a sum less than or equal to 3. It can be proven that 2 is the maximum size of such a subsequence, so answer[0] = 2.
    - The subsequence [4,5,1] has a sum less than or equal to 10. It can be proven that 3 is the maximum size of such a subsequence, so answer[1] = 3.
    - The subsequence [4,5,2,1] has a sum less than or equal to 21. It can be proven that 4 is the maximum size of such a subsequence, so answer[2] = 4.
     */
    //Author:Anand
    public int[] answerQueries(int[] nums, int[] queries) {
        Arrays.sort(nums);
        int[] ps = new int[nums.length];

        for (int i = 0; i < nums.length; i++) {
            if (i == 0) ps[i] = nums[i];
            else ps[i] = ps[i - 1] + nums[i];
        }

        int idx = 0;
        int[] ans = new int[queries.length];
        for (int query : queries)
            ans[idx++] = Math.max(bs(ps, query), 0);

        return ans;
    }

    private int bs(int[] ps, int bus) {
        int l = 0, h = ps.length - 1;
        while (l <= h) {
            int mid = l + (h - l) / 2;

            if (ps[mid] <= bus) {
                l = mid + 1;
            } else h = mid;

            if (l == h && l == mid) return l;
        }
        return l;
    }

    /*
      Input: buses = [10,20], passengers = [2,17,18,19], capacity = 2
      Output: 16
      Explanation:
      The 1st bus departs with the 1st passenger.
      The 2nd bus departs with you and the 2nd passenger.
      Note that you must not arrive at the same time as the passengers, which is why you must arrive before the 2nd passenger to catch the bus.


       buses = [10,20], pass = [2,17,18,19]
   */
    // TC = O(nlogn)
    public int latestTimeCatchTheBus(int[] buses, int[] passengers, int capacity) {
        Arrays.sort(buses);
        Arrays.sort(passengers);

        int prev = -1;
        int pcb = 0;
        for (int bus : buses) {
            int ind = bs(passengers, bus);
            if (prev == -1) {
                prev = Math.min(ind, capacity);
                pcb = prev;
            } else {
                pcb = (ind > (prev + capacity)) ? capacity : ind;
                prev += pcb;
            }
        }

        int value = 0;
        if (pcb >= capacity) {
            prev--;
            value = passengers[prev];
            for (int i = prev - 1; i >= 0; i--) {
                if ((value - passengers[i]) != 1) return value - 1;
                value = passengers[i];
            }
        } else {
            value = buses[buses.length - 1];
            for (int i = passengers.length - 1; i >= 0; i--) {
                if (value > passengers[i]) return value;
                value = passengers[i];
            }
        }

        return value - 1;
    }


    /*
    Input: divisor1 = 2, divisor2 = 7, uniqueCnt1 = 1, uniqueCnt2 = 3
    Output: 4
    Explanation:
    We can distribute the first 4 natural numbers into arr1 and arr2.
    arr1 = [1] and arr2 = [2,3,4].
    We can see that both arrays satisfy all the conditions.
    Since the maximum value is 4, we return it.

    Intution-
      Check if we can divide all numbers 1 to Integer.MAX_VALUE to arr1, arr2 such that conditions are met.
      We will continue to minimise h i.e. higher limit of BS
     */
    // TC = O(lognN) where n ~= 10^9
    public int minimizeSet(int d1, int d2, int c1, int c2) {
        long l = 0L, h = Integer.MAX_VALUE;
        long ans = Integer.MAX_VALUE;

        while (l <= h) {
            long mid = l + (h - l) / 2;

            if (safely(mid, d1, d2, c1, c2)) {
                ans = Math.min(ans, mid);
                h = mid - 1;
            } else l = mid + 1;
        }

        return (int) ans;
    }


    // This Fn will check if we can allocate elements within a range 0-mid safely as per above needs
    private boolean safely(long mid, long d1, long d2, long c1, long c2) {

        long notDivByD1 = mid - (mid / d1);
        long notDivByD2 = mid - (mid / d2);

        long notDivByBoth = mid - (mid / lcm(d1, d2));

        return notDivByD1 >= c1 && notDivByD2 >= c2 && notDivByBoth >= c1 + c2;
    }

    private long lcm(long n1, long n2) {
        return n1 * n2 / gcd(n1, n2);
    }

    private long gcd(long n1, long n2) {
        if (n2 == 0) return n1;
        return gcd(n2, n1 % n2);
    }


    /*
    Approach
    To find smaller numbers than query[i] we can sort the array and use binary search
    Binary search over sorted nums to find index of query[i]
    Then use prefix sums to find sum of number in smaller and larger segments
    prefix[n] - prefix[i] is sum of numbers greater than or equal to query[i]
    prefix[i] is sum of numbers smaller than query[i]
    query[i] * i - prefix[i] is increments required
    prefix[n] - prefix[i] - query[i] * (n - i) is decrements required
    Total = query[i] * i - prefix[i] + prefix[n] - prefix[i] - query[i] * (n - i)
    Can be simplified to query[i] * (2 * i - n) + prefix[n] - 2 * prefix[i]

     */
    public List<Long> minOperations(int[] nums, int[] queries) {
        List<Long> ans = new ArrayList<>();
        int n = nums.length;
        Arrays.sort(nums);
        long[] prefix = new long[n + 1];
        for (int i = 1; i <= n; i++) prefix[i] = prefix[i - 1] + nums[i - 1];

        for (int query : queries) {
            int i = Arrays.binarySearch(nums, query);

            if (i < 0) i = -(i + 1);
            // insertion point in array, i.e. if target is not found in array
            // then it retruns negative of insertion point

            ans.add((long) query * (2 * i - n) + prefix[n] - 2 * prefix[i]);
        }

        return ans;
    }

}
