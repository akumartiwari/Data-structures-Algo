package com.company;

import java.util.HashMap;
import java.util.*;

public class BinarySearch {

    /*
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Longest Subsequence With Limited Sum (LeetCode 2389)
     * ──────────────────────────────────────────────────────────────────
     * Given an integer array nums and a queries array, for each query[i]
     * return the MAXIMUM number of elements you can pick from nums such
     * that their sum is ≤ query[i].
     *
     * Example: nums = [4,5,2,1], queries = [3,10,21]
     *   - query=3:  pick [2,1] → sum=3, size=2  → answer[0]=2
     *   - query=10: pick [4,5,1] → sum=10, size=3 → answer[1]=3
     *   - query=21: pick all [4,5,2,1] → sum=12, size=4 → answer[2]=4
     *   Output: [2, 3, 4]
     *
     * SOLUTION — Sort + Prefix Sum + Binary Search:
     *   1. Sort nums (to greedily pick the smallest elements first).
     *   2. Build a prefix sum array ps[] where ps[i] = sum of first i+1 elements.
     *   3. For each query, binary search in ps[] for the largest index where
     *      ps[index] <= query. That index + 1 is the maximum count.
     *
     * Time:  O((n + q) log n)  — sort + q binary searches
     * Space: O(n)              — prefix sum array
     * ──────────────────────────────────────────────────────────────────
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Furthest Building You Can Reach (LeetCode 1642)
     * ──────────────────────────────────────────────────────────────────
     * You are given an array of building heights and a fixed supply of
     * bricks and ladders. Moving from building i to i+1:
     *   - If heights[i+1] <= heights[i]: free move (going down or flat).
     *   - If heights[i+1] >  heights[i]: you must use bricks (exact diff)
     *     OR one ladder (handles any jump for free).
     * Return the index of the furthest building you can reach.
     *
     * Example: heights = [4,2,7,6,9,14,12], bricks = 5, ladders = 1
     *          → Output: 4
     *
     * SOLUTION — Binary Search on Answer + Greedy Check:
     *   Binary search on the destination index (0 … n-1).
     *   For a candidate index `mid`, check reachability greedily:
     *     1. Collect all positive height differences up to `mid` in a max-heap.
     *     2. Always use a ladder for the LARGEST jumps (greedy optimal).
     *     3. After ladders are exhausted, cover remaining jumps with bricks.
     *     4. If bricks suffice → reachable; binary search right; else left.
     *
     * Time:  O(K·logN·logK)  — BS × heap ops per check
     * Space: O(N)             — max-heap
     * ──────────────────────────────────────────────────────────────────
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

    /*
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Minimum Time to Complete Trips (LeetCode 2187)
     * ──────────────────────────────────────────────────────────────────
     * You have n buses. Bus i completes one trip every time[i] units.
     * Multiple buses can run simultaneously. Find the MINIMUM total time
     * needed for all buses together to complete at least `totalTrips`.
     *
     * Example: time = [1,2,3], totalTrips = 5  →  Output: 3
     *   At t=3: bus0 → 3 trips, bus1 → 1 trip, bus2 → 1 trip = 5 total ✓
     *
     * SOLUTION — Binary Search on Time:
     *   Key insight: if T time units pass, bus i completes ⌊T / time[i]⌋ trips.
     *   → Binary search on T in range [0, max_time * totalTrips].
     *   → For each candidate T (mid), sum up trips completed by all buses.
     *   → If sum >= totalTrips → T is feasible, try smaller (h = mid).
     *   → Else need more time (l = mid + 1).
     *   → Converges to the minimum feasible T.
     *
     * Time:  O(n · log(maxTime × totalTrips))
     * Space: O(1)
     * ──────────────────────────────────────────────────────────────────
     */
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Minimum Value of K Such That Partition Is Possible
     * ──────────────────────────────────────────────────────────────────
     * Given an array nums[], find the minimum integer k (1 ≤ k ≤ 9999)
     * such that when every element arr[i] is split into ceil(arr[i]/k)
     * parts, the total number of parts is ≤ k².
     * In other words, we want the smallest k where we can form exactly
     * k groups of k elements (a k×k grid partition).
     *
     * SOLUTION — Binary Search on k:
     *   1. Binary search k in range [1, 9999].
     *   2. For each candidate k, compute total parts = Σ ceil(arr[i] / k).
     *   3. If total parts ≤ k² → k is feasible; record result and go lower.
     *   4. Else → k is too small, go higher.
     *   A visited-set guards against infinite loops in edge cases.
     *
     * Time:  O(n · log(9999))
     * Space: O(log(9999)) — for the visited set
     * ──────────────────────────────────────────────────────────────────
     */
    public int minimumK(int[] nums) {
        int low = 1;
        int high = 9999;
        int mid = high;
        int res = 0;
        Set<Integer> mids = new HashSet<>();
        while (mid >= 1 && !mids.contains(mid)) {
            mids.add(mid);
            mid = low + (high - low) / 2;
            if (possible(mid, nums.length, nums)) {
                high = mid - 1;
                res = mid;
            } else {
                low = mid + 1;
            }
        }
        return res;
    }

    public boolean possible(int val, int n, int[] arr) {
        long op = 0;
        for (int i = n - 1; i >= 0; i--) {
            op += (arr[i] / val) + (arr[i] % val == 0 ? 0 : 1);
        }
        return op <= val * val;
    }

    /*
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Maximum Candies Allocated to K Children (LeetCode 2226)
     * ──────────────────────────────────────────────────────────────────
     * You have n piles of candies. You want to allocate some piles to k
     * children such that:
     *   - Each child receives the SAME number of candies.
     *   - Each child's candies come from exactly ONE pile (a pile can be
     *     split but not merged).
     *   - Some piles may go unused.
     * Return the MAXIMUM number of candies each child can get (0 if impossible).
     *
     * Example: candies = [5,8,6], k = 3  →  Output: 5
     *   Split pile 8 → [5,3], pile 6 → [5,1].
     *   Three children each get 5.
     *
     * Example: candies = [2,5], k = 11  →  Output: 0
     *   Only 7 total candies for 11 children — impossible.
     *
     * SOLUTION — Binary Search on Candy Count:
     *   1. Binary search on the amount `mid` each child receives [1, max pile].
     *   2. For each `mid`, count how many children can be served:
     *        served = Σ floor(pile[i] / mid)
     *   3. If served >= k → `mid` is feasible; record and go higher.
     *   4. Else → `mid` is too large; go lower.
     *
     * Time:  O(n · log(maxPile))
     * Space: O(1)
     * ──────────────────────────────────────────────────────────────────
     */
    // Author: Anand
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

    /*
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Longest Increasing Subsequence — O(n log n) (LeetCode 300)
     * ──────────────────────────────────────────────────────────────────
     * Given an integer array, return the length of the longest strictly
     * increasing subsequence (elements don't need to be contiguous).
     *
     * Example: Input = [1, 5, 2, 8, 9]  →  Output: 4
     *          Valid LIS: {1, 5, 8, 9} or {1, 2, 8, 9}
     *
     * SOLUTION — Patience Sorting with Binary Search (lower_bound):
     *   Maintain a `lis` list that always holds the smallest possible
     *   tail element for each LIS length found so far.
     *
     *   For each number `a` in the array:
     *     - If a > last element in lis → extend the LIS; append and increment length.
     *     - Else → find the first element in lis that is >= a using lower_bound
     *              (binary search), and REPLACE it with a.
     *              This keeps the tails as small as possible, maximising future growth.
     *
     *   The LENGTH of `lis` at the end is the answer.
     *   (Note: `lis` itself may NOT be a valid subsequence, only its length is correct.)
     *
     * Time:  O(n log n)  — one binary search per element
     * Space: O(n)        — the lis list
     * ──────────────────────────────────────────────────────────────────
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

    public int lengthOfLISLB(int[] arr) {
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Maximum Total Beauty of the Gardens (LeetCode 2234)
     * ──────────────────────────────────────────────────────────────────
     * You have n gardens. Garden i has flowers[i] flowers. You can plant
     * `newFlowers` more flowers (distributed freely). Scoring:
     *   - A garden is "complete" if it reaches the `target` flowers.
     *     Each complete garden earns `full` points.
     *   - Among all incomplete gardens, the MINIMUM flower count × `partial`
     *     earns additional points.
     * Maximise total beauty = (# complete × full) + (min incomplete × partial).
     *
     * Example: flowers=[1,3,1,1], newFlowers=7, target=6, full=12, partial=1
     *          → Output: 14
     *
     * SOLUTION — Sort + Prefix Sum + Binary Search:
     *   1. Sort gardens. Gardens already at/above target are always complete.
     *   2. Iterate from right to left, greedily making more gardens complete
     *      by spending (target - flowers[j]) flowers.
     *   3. For remaining incomplete gardens, use a cumulative cost array
     *      to figure out how high we can uniformly raise all incomplete gardens.
     *      Binary search on the cumulative cost array to find the best index
     *      where we can afford to raise all gardens up to that index.
     *   4. At each step compute beauty and track the maximum.
     *
     * Time:  O(n log n)  — sort + binary searches
     * Space: O(n)        — cumulative cost array
     * ──────────────────────────────────────────────────────────────────
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Successful Pairs of Spells and Potions (LeetCode 2300)
     * ──────────────────────────────────────────────────────────────────
     * You have n spells and m potions. A spell-potion pair is "successful"
     * if spell[i] * potion[j] >= success.
     * Return an array where result[i] = number of potions that form a
     * successful pair with spells[i].
     *
     * Example: spells = [5,1,3], potions = [1,2,3,4,5], success = 7
     *   - Spell 5: needs potion >= ceil(7/5)=2 → potions [2,3,4,5] → 4 pairs
     *   - Spell 1: needs potion >= ceil(7/1)=7 → none → 0 pairs
     *   - Spell 3: needs potion >= ceil(7/3)=3 → potions [3,4,5] → 3 pairs
     *   Output: [4, 0, 3]
     *
     * SOLUTION — Sort Potions + Binary Search per Spell:
     *   1. Sort potions array.
     *   2. For each spell s, the minimum required potion strength is
     *      ceil(success / s). Binary search for its first occurrence in
     *      the sorted potions array.
     *   3. All potions from that index onward form valid pairs.
     *      count = potions.length - firstValidIndex.
     *   4. A duplicate-index map is maintained to handle ties correctly
     *      (finds the leftmost index of a duplicate value).
     *
     * Time:  O((n + m) log m)  — sort + n binary searches
     * Space: O(m)              — duplicate index map
     * ──────────────────────────────────────────────────────────────────
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Count Number of Rectangles Containing Each Point (LeetCode 2250)
     * ──────────────────────────────────────────────────────────────────
     * You are given a list of axis-aligned rectangles where each rectangle
     * is defined by its top-right corner [li, hi] (bottom-left is always [0,0]).
     * A rectangle contains a point [px, py] if 0 <= px <= li AND 0 <= py <= hi.
     * For each query point, return how many rectangles contain it.
     *
     * Example: rectangles = [[1,1],[2,2],[3,3]], points = [[1,3],[1,1]]
     *   - Point (1,3): only rectangle [3,3] contains it → 1
     *   - Point (1,1): all three rectangles contain it  → 3
     *   Output: [1, 3]
     *
     * SOLUTION — Group by Height + Sorted X + Binary Search:
     *   1. Group all rectangle x-values by their y (height) in a TreeMap.
     *   2. Sort the x-values within each height group.
     *   3. For each query point (px, py):
     *      - Only consider rectangles with height >= py (TreeMap.subMap).
     *      - For each such height, binary search in the sorted x-list to find
     *        the first x >= px; all elements from that index onward are valid.
     *      - Sum up the counts.
     *
     * Time:  O((n + q) · log n)  — grouping + binary searches
     * Space: O(n)                — height-to-x mapping
     * ──────────────────────────────────────────────────────────────────
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
            if (map.containsKey(key)) list.addAll(map.get(key));
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: The Latest Time to Catch a Bus (LeetCode 2332)
     * ──────────────────────────────────────────────────────────────────
     * There are n buses and m passengers. Each bus departs at buses[i]
     * and can carry at most `capacity` passengers (taken in arrival order).
     * You want to arrive at the bus stop as LATE as possible and still
     * board a bus. You cannot arrive at the exact same time as any passenger.
     *
     * Example: buses=[10,20], passengers=[2,17,18,19], capacity=2 → Output: 16
     *   Bus 10 takes passenger 2. Bus 20 can take 2 passengers → takes 17 and you.
     *   You arrive at 16 (just before passenger 17 arrives at 17). ✓
     *
     * SOLUTION — Sort + Greedy Simulation:
     *   1. Sort both buses and passengers.
     *   2. Simulate passenger boarding across all buses to figure out the last
     *      bus's final state (how many boarded, last passenger index).
     *   3. Two cases for the latest valid arrival time:
     *      - If last bus was FULL: go backwards from the last boarded passenger
     *        index until you find a gap in consecutive arrival times.
     *      - If last bus had SPACE: the latest valid time is the bus departure
     *        itself, or just before any passenger occupying that slot.
     *
     * Time:  O(n log n + m log m)  — sorting buses and passengers
     * Space: O(1)
     * ──────────────────────────────────────────────────────────────────
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Minimize the Maximum of Two Arrays (LeetCode 2513)
     * ──────────────────────────────────────────────────────────────────
     * You want to build two arrays arr1 and arr2 using the first n positive
     * integers (each used exactly once), where:
     *   - arr1 has uniqueCnt1 distinct integers, NONE divisible by divisor1.
     *   - arr2 has uniqueCnt2 distinct integers, NONE divisible by divisor2.
     * Return the MINIMUM possible value of max(max(arr1), max(arr2)).
     *
     * Example: divisor1=2, divisor2=7, uniqueCnt1=1, uniqueCnt2=3 → Output: 4
     *   arr1=[1], arr2=[2,3,4]. No element of arr1 is div by 2 ✓
     *   No element of arr2 is div by 7 ✓
     *
     * SOLUTION — Binary Search on Answer + Math (Inclusion-Exclusion):
     *   Binary search on `mid` = the upper bound of integers we can use (1…mid).
     *   For a candidate `mid`, check if we can fill both arrays:
     *     - Numbers NOT divisible by d1 = mid - floor(mid/d1)  → available for arr1
     *     - Numbers NOT divisible by d2 = mid - floor(mid/d2)  → available for arr2
     *     - Numbers NOT divisible by EITHER = mid - floor(mid/lcm(d1,d2)) → usable by both
     *   Feasibility: notDivD1 >= c1 AND notDivD2 >= c2
     *                AND notDivByBoth >= c1 + c2   (shared pool must cover combined need)
     *   Use LCM (via GCD) to find numbers excluded from both arrays simultaneously.
     *
     * Time:  O(log(INT_MAX))  — binary search over ~2×10⁹
     * Space: O(1)
     * ──────────────────────────────────────────────────────────────────
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
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Minimum Operations to Make All Array Elements Equal (LeetCode 2602)
     * ──────────────────────────────────────────────────────────────────
     * You are given an integer array nums and a queries array.
     * For each query[i], find the minimum number of operations needed to
     * make every element in nums equal to query[i], where one operation
     * increments or decrements any element by 1.
     *
     * Example: nums = [3,1,6,8], queries = [1,5]
     *   - query=1: cost = |3-1|+|1-1|+|6-1|+|8-1| = 2+0+5+7 = 14
     *   - query=5: cost = |3-5|+|1-5|+|6-5|+|8-5| = 2+4+1+3 = 10
     *   Output: [14, 10]
     *
     * SOLUTION — Sort + Prefix Sum + Binary Search:
     *   Total cost to make everything equal to q =
     *       Σ|nums[i] - q|
     *     = (sum of elements > q) - q*(count of elements > q)
     *     + q*(count of elements < q) - (sum of elements < q)
     *
     *   Using a sorted array and prefix sums:
     *     - Binary search for the insertion index i of query q.
     *     - Elements left of i are smaller; elements right are >= q.
     *     - Formula: q*(2i - n) + prefix[n] - 2*prefix[i]
     *
     * Time:  O(n log n + q log n)  — sort + binary search per query
     * Space: O(n)                  — prefix sum array
     * ──────────────────────────────────────────────────────────────────
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

    /*
     * ──────────────────────────────────────────────────────────────────
     * PROBLEM: Longest Increasing Subsequence (LeetCode 300)
     * ──────────────────────────────────────────────────────────────────
     * Given an integer array, return the length of the longest strictly
     * increasing subsequence (elements need not be contiguous).
     *
     * Example: nums = [10,9,2,5,3,7,101,18]  →  Output: 4
     *          LIS: [2, 3, 7, 101]
     *
     * Example: nums = [0,1,0,3,2,3]  →  Output: 4
     *
     * SOLUTION — Patience Sorting (O(n log n)):
     *   Maintain a list `ans` of the smallest tail elements for each length.
     *   For each num in nums:
     *     - If num > last element → simply extend the list (new longer LIS found).
     *     - Else → use Collections.binarySearch to find the leftmost position
     *              where num can replace an existing element.
     *              Replacing keeps tails as small as possible for future growth.
     *   The final length of the list is the LIS length.
     *   (The list itself is NOT necessarily the actual LIS sequence.)
     *
     * Time:  O(n log n)  — binary search for each element
     * Space: O(n)        — the tails list
     * ──────────────────────────────────────────────────────────────────
     */
    public int lengthOfLIS(int[] nums) {

        List<Integer> ans = new ArrayList<>();
        int len = 0;
        for (int num : nums) {
            if (len == 0 || num > ans.get(len - 1)) {
                ans.add(num);
                len++;
            } else {
                // next greater element than current one in the ans list
                int idx = Collections.binarySearch(ans, num);
                if (idx < 0) idx = -(idx + 1);
                if (ans.size() == idx) ans.add(num);
                else ans.set(idx, num);
            }
        }

        return len;
    }
}
