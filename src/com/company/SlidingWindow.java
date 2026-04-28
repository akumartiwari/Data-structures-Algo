package com.company;

import java.util.HashMap;
import java.util.*;
import java.util.stream.Collectors;

public class SlidingWindow {
    /*
    Input: tiles = [[1,5],[10,11],[12,18],[20,25],[30,32]], carpetLen = 10
    Output: 9
    Explanation: Place the carpet starting on tile 10.
    It covers 9 white tiles, so we return 9.
    Note that there may be other places where the carpet covers 9 white tiles.
    It can be shown that the carpet cannot cover more than 9 white tiles.
     */
    // Author: Anand
    int end = 1;
    int start = 0;

    // Sliding window
    // TC = O(26N)
    //Author: Anand
    /*
     * PROBLEM: Total Appeal of All Substrings (LeetCode 2262)
     * Compute the sum of appeal (number of distinct chars) of all substrings of s.
     *
     * ALGORITHM: Sliding Window with per-character occurrence positions
     * TC: O(26*n) | SC: O(26*n)
     */
    public long appealSum(String s) {
        int n = s.length();
        List<Integer>[] oc = new ArrayList[26];
        for (int i = 0; i < 26; i++) oc[i] = new ArrayList<>();

        for (int i = 0; i < n; i++) oc[s.charAt(i) - 'a'].add(i);

        long total = 0L;

        for (List<Integer> indexes : oc) {
            for (int idx = 0; idx < indexes.size(); idx++) {
                // curr = total number of substrings able  to generate from 0 to idx
                // next = total number of substrings able  to generate from idx to next of SW
                int curr = indexes.get(idx);
                int next = idx < indexes.size() - 1 ? indexes.get(idx + 1) : n;
                int right = next - idx;
                total += (long) (curr + 1) * right;
            }
        }

        return total;
    }

    /*
     * PROBLEM: Maximum White Tiles Covered by a Carpet (LeetCode 2271)
     * Find the maximum number of white tiles a carpet of length len can cover.
     *
     * ALGORITHM: Sliding Window on sorted tile intervals
     * TC: O(n log n) | SC: O(1)
     */
    /*
    Input: tiles = [[1,5],[10,11],[12,18],[20,25],[30,32]], carpetLen = 10
    Output: 9
    Explanation: Place the carpet starting on tile 10.
    It covers 9 white tiles, so we return 9.
    Note that there may be other places where the carpet covers 9 white tiles.
    It can be shown that the carpet cannot cover more than 9 white tiles.
     */
    // Author: Anand

    int maximumWhiteTiles(int[][] tiles, int len) {

        int result = 0, si = 0, covered = 0;

        for (int ei = 0; result < len && ei < tiles.length; ) {
            if (si == ei || tiles[si][end] + len > tiles[ei][end]) {
                covered += Math.min(len, tiles[ei][end] - tiles[ei][start] + 1);
                result = Math.max(result, covered);
                ++ei;
            } else {
                int partial = Math.max(tiles[si][start] + len - tiles[ei][start], 0);
                result = Math.max(result, covered + partial);
                covered -= (tiles[si][end] - tiles[si][start] + 1);
                ++si;
            }
        }

        return result;
    }


    /*
     * PROBLEM: Subarray Product Less Than K (LeetCode 713)
     * Count the number of contiguous subarrays whose product is strictly less than k.
     *
     * ALGORITHM: Sliding Window (two pointers)
     * TC: O(n) | SC: O(1)
     */
    // Author: Anand
    //    Sliding window approach
    //    The idea is to keep moving i till condition is satisfied. The moment it fails
    //    start moving j and update result. Then evaluate the number of subarrays possible

    public int numSubarrayProductLessThanK(int[] nums, int k) {
        long prod = 1;
        int cnt = 0;

        if (k <= 1) return 0;

        for (int i = 0, j = 0; i < nums.length; i++) {
            prod *= nums[i];

            while (prod >= k) {
                prod /= nums[j++];
            }

            cnt += (i - j + 1);
        }
        return cnt;
    }


    /*
     * PROBLEM: Count Continuous Subarrays (LeetCode 2762)
     * Count subarrays where max - min <= 2.
     *
     * ALGORITHM: Sliding Window with TreeMap (maintains sorted frequency)
     * TC: O(n log n) | SC: O(n)
     */
    public long continuousSubarrays(int[] nums) {
        long cnt = 0L;
        int smallest = Integer.MAX_VALUE, largest = Integer.MIN_VALUE;
        int i, j;
        TreeMap<Integer, Integer> freq = new TreeMap<>(); // Used because it will maintain sorted order
        for (i = 0, j = 0; i < nums.length; ++i) {
            smallest = Math.min(smallest, nums[i]);
            largest = Math.max(largest, nums[i]);
            freq.put(nums[i], freq.getOrDefault(nums[i], 0) + 1);

            while (largest - smallest > 2 && j <= i) {
                j++;
                freq.put(nums[j - 1], freq.getOrDefault(nums[j - 1], 0) - 1);
                if (freq.get(nums[j - 1]) <= 0) freq.remove(nums[j - 1]);
                smallest = freq.firstKey();
                largest = freq.lastKey();
            }
            cnt += i - j + 1;
        }

        return cnt;
    }

    /*
     * PROBLEM: Count Subarrays with Score Less Than K (LeetCode 2302)
     * Count subarrays whose score (sum × length) is strictly less than k.
     *
     * ALGORITHM: Sliding Window (two pointers)
     * TC: O(n) | SC: O(1)
     */
    /*
    Input: nums = [2,1,4,3,5], k = 10
    Output: 6
    Explanation:
    The 6 subarrays having scores less than 10 are:
    - [2] with score 2 * 1 = 2.
    - [1] with score 1 * 1 = 1.
    - [4] with score 4 * 1 = 4.
    - [3] with score 3 * 1 = 3.
    - [5] with score 5 * 1 = 5.
    - [2,1] with score (2 + 1) * 2 = 6.
    Note that subarrays such as [1,4] and [4,3,5] are not considered because their scores are 10 and 36 respectively, while we need scores strictly less than 10.
     */
    // Author : Anand
    public long countSubarrays(int[] nums, long k) {
        long sum = 0L;
        long cnt = 0L;

        for (int i = 0, j = 0; i < nums.length; ++i) {
            sum += nums[i];
            while ((i - j + 1) * sum >= k) {
                sum -= nums[j++];
            }
            cnt += i - j + 1;
        }

        return cnt;
    }

    /*
     * PROBLEM: Minimum Size Subarray in Infinite Array (LeetCode 2875)
     * Find the minimum length subarray of the infinite repeated array that sums to target.
     *
     * ALGORITHM: Sliding Window with modular target
     * TC: O(n) | SC: O(n)
     */
    public int minSizeSubarray(int[] nums, int target) {

        int ts = Arrays.stream(nums).sum();
        int nt = ts < target ? target % ts : target;

        int sum = 0;
        List<Integer> nn = Arrays.stream(nums).boxed().collect(Collectors.toList());
        nn.addAll(nn);

        int ans = Integer.MAX_VALUE;
        for (int i = 0, j = 0; i < nn.size(); ++i) {
            sum += nn.get(i);
            while (sum > nt) sum -= nn.get(j++);
            if (sum == nt) ans = Math.min(ans, i - j + 1);
        }

        if (ans == Integer.MAX_VALUE) return -1;
        int factor = target / ts;
        return ts < target ? (ans + factor * nums.length) : ans;
    }

    //TODO: complete
    class Solution {
        public int maximumBeauty(int[] nums, int k) {
            int maxLen = 0, len = 0;
            Arrays.sort(nums);

            int i = 0, j = 0;
            int[] range = new int[]{nums[0] - k, nums[0] + k};

            for (i = 0, j = 0; i < nums.length; ++i) {
                // condition fails
                while (nums[i] < range[0] || nums[i] < range[1]) {
                    j++;
                }
            }

            return len;
        }
    }

    /*
     * PROBLEM: Count Interesting Subarrays (LeetCode 2845)
     * Count subarrays where (count of nums[i] % modulo == k) % modulo == k.
     *
     * ALGORITHM: Sliding Window with prefix count modulo
     * TC: O(n) | SC: O(1)
     */
    /*
     nums[i] % modulo == k. Then, cnt % modulo == k.
     Solve using sliding window algorithm
     */
    public long countInterestingSubarrays(List<Integer> nums, int modulo, int k) {
        long sum = 0L;
        long cnt = 0L;

        for (int i = 0, j = 0; i < nums.size(); ++i) {
            sum += nums.get(i);
            while ((i - j + 1) * sum >= k) {
                sum -= nums.get(j++);
            }
            cnt += i - j + 1;
        }

        return cnt;
    }

    int MOD = (int) (1e9 + 7);

    /*
     * PROBLEM: Count Number of Homogenous Substrings (LeetCode 1759)
     * Count substrings where all characters are the same.
     *
     * ALGORITHM: Sliding Window (count consecutive same chars)
     * TC: O(n) | SC: O(1)
     */
    public int countHomogenous(String s) {
        int cnt = 0;
        for (int i = 0, j = 0; i < s.length(); ++i) {
            while (s.charAt(j) != s.charAt(i)) j++;
            cnt = (cnt + i - j + 1) % MOD;
        }
        return cnt;
    }

    /*
     * PROBLEM: Number of 1 Bits Substrings (LeetCode 1513)
     * Count substrings that consist only of '1's.
     *
     * ALGORITHM: Sliding Window (count consecutive '1' chars)
     * TC: O(n) | SC: O(1)
     */
    public int numSub(String s) {
        int cnt = 0;
        for (int i = 0, j = 0; i < s.length(); ++i) {
            while (s.charAt(j) != s.charAt(i)) j++;
            if (s.charAt(i) == '1') cnt = (cnt + i - j + 1) % MOD;
        }
        return cnt;
    }

    /*
     * PROBLEM: Sum of Subarray Minimums (Helper)
     * Compute sum of (max - min) for all subarrays.
     *
     * ALGORITHM: Brute Force O(n²)
     * TC: O(n²) | SC: O(1)
     */
    public int sumSubarrayMins(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            int max = Integer.MIN_VALUE, min = Integer.MAX_VALUE;
            for (int j = i; j < nums.length; j++) {
                max = Math.max(max, nums[j]);
                min = Math.min(min, nums[j]);
                sum += (max - min);
            }
        }

        return sum;
    }


    /*
     * PROBLEM: Maximum Ascending Subarray Sum (LeetCode 1800)
     * Find the maximum sum of a contiguous strictly ascending subarray.
     *
     * ALGORITHM: Greedy Linear Scan
     * TC: O(n) | SC: O(1)
     */
    public int maxAscendingSum(int[] nums) {
        int max = 0, sum = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (i == 0) sum += nums[i];
            else {
                if (nums[i - 1] < nums[i]) sum += nums[i];
                else {
                    max = Math.max(max, sum);
                    sum = nums[i];
                }
            }
        }
        max = Math.max(max, sum);
        return max;
    }


    /*
     * PROBLEM: Consecutive Characters (LeetCode 1446)
     * Find the length of the longest substring with all identical characters.
     *
     * ALGORITHM: Sliding Window
     * TC: O(n) | SC: O(1)
     */
    public int maxPower(String s) {
        int max = 0;
        for (int i = 0, j = 0; i < s.length(); ++i) {
            while (s.charAt(j) != s.charAt(i)) j++;
            max = Math.max(max, i - j + 1);
        }
        return max;
    }

    /*
     * PROBLEM: Subarray LCM Equal K (LeetCode 2470)
     * Count subarrays whose LCM equals k.
     *
     * ALGORITHM: Brute Force with LCM computation
     * TC: O(n² log k) | SC: O(1)
     */
    public int subarrayLCM(int[] nums, int k) {
        int cnt = 0;
        int lcm = -1;
        int sz = 0;
        for (int i = 0; i < nums.length; i++) {
            sz = 1;
            lcm = -1;
            if (nums[i] == k) cnt++;
            for (int j = i + 1; j < nums.length; j++) {
                sz++;
                lcm = lcm != -1 ? lcm(lcm, nums[j]) : lcm(nums[i], nums[j]);
                if (lcm == k) cnt++;
                else if (lcm > k) break;
            }
        }
        return cnt;
    }

    /*
     * PROBLEM: LCM Computation (Helper)
     * Compute the least common multiple of two integers.
     *
     * ALGORITHM: Math (LCM = a*b / GCD)
     * TC: O(log(min(a,b))) | SC: O(1)
     */
    public int lcm(int a, int b) {
        return (a * b) / gcd(a, b);
    }

    /*
     * PROBLEM: GCD Computation (Helper)
     * Compute the greatest common divisor of two integers.
     *
     * ALGORITHM: Euclidean Algorithm (recursive)
     * TC: O(log(min(a,b))) | SC: O(log(min(a,b)))
     */
    //int version for gcd
    public int gcd(int a, int b) {
        if (b == 0)
            return a;

        return gcd(b, a % b);

    }

    /*
     * PROBLEM: Number of Zero-Filled Subarrays (LeetCode 2348)
     * Count the total number of subarrays filled with zeros.
     *
     * ALGORITHM: Sliding Window (count consecutive zeros)
     * TC: O(n) | SC: O(1)
     */
    /*
    Input: nums = [1,3,0,0,2,0,0,4]
    Output: 6
    Explanation:
    There are 4 occurrences of [0] as a subarray.
    There are 2 occurrences of [0,0] as a subarray.
    There is no occurrence of a subarray with a size more than 2 filled with 0. Therefore, we return 6.
     */
    //Author: Anand
    public long zeroFilledSubarray(int[] nums) {
        long cnt = 0L;
        for (int i = 0, j = 0; i < nums.length; ++i) {
            if (nums[i] != 0) j = i + 1;
            cnt += i - j + 1;
        }
        return cnt;
    }

    /*
     * PROBLEM: Maximum Subarray Sum With Length Divisible by K (LeetCode 2461)
     * Find the maximum sum of a subarray of exactly length k with all distinct elements.
     *
     * ALGORITHM: Sliding Window with HashMap (frequency tracking)
     * TC: O(n) | SC: O(k)
     */
    public long maximumSubarraySum(int[] nums, int k) {
        long sum = 0L;
        boolean duplicate = false;

        long maxSum = 0L;
        Map<Integer, Integer> freq = new HashMap<>();
        for (int i = 0, j = 0; i < nums.length; ++i) {
            sum += nums[i];
            freq.put(nums[i], freq.getOrDefault(nums[i], 0) + 1);

            if (freq.get(nums[i]) >= 2) duplicate = true;

            if (freq.values().size() == k && !duplicate) maxSum = Math.max(maxSum, sum);
            else if (freq.values().size() > k && !duplicate) {
                freq.put(nums[j], freq.getOrDefault(nums[j], 0) - 1);
                if (freq.get(nums[j]) <= 0) freq.remove(nums[j]);
                sum -= nums[j++];
                maxSum = Math.max(maxSum, sum);
            } else {
                while (freq.containsKey(nums[i]) && freq.get(nums[i]) >= 2) {
                    freq.put(nums[j], freq.getOrDefault(nums[j], 0) - 1);
                    if (freq.get(nums[j]) <= 0) freq.remove(nums[j]);
                    sum -= nums[j++];
                }

                if (freq.get(nums[i]) <= 1) duplicate = false;
                if (i - j + 1 == k) maxSum = Math.max(maxSum, sum);
            }
        }

        return maxSum;
    }

    //TBD- wrong answer
    public int maximumCostSubstring(String s, String chars, int[] vals) {
        int sum = 0;
        int maxSum = 0;
        Map<Character, Integer> cm = new HashMap<>();
        for (int i = 0; i < chars.length(); i++) cm.put(chars.charAt(i), i);

        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            sum += cm.containsKey(c) ? vals[cm.get(c)] : (int) (c - 'a') + 1;
            maxSum = Math.max(sum, maxSum);
        }


        System.out.println(maxSum);

        sum = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            int cal = cm.containsKey(c) ? vals[cm.get(c)] : (int) (c - 'a') + 1;
            if (cal < 0) maxSum = Math.max(maxSum, maxSum - cal);
        }


        return maxSum;

    }
}
