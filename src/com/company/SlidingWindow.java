package com.company;

import java.util.ArrayList;
import java.util.List;

public class SlidingWindow {
    // Sliding window
    // TC = O(26N)
    //Author: Anand
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
}
