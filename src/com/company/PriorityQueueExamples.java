package com.company;

import javafx.util.Pair;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Collections;
import java.util.PriorityQueue;
import java.util.Queue;

public class PriorityQueueExamples {

    /*
    Input: nums = [2,4,-2], k = 5
    Output: 2
    Explanation: All the possible subsequence sums that we can obtain are the following sorted in decreasing order:
    - 6, 4, 4, 2, 2, 0, 0, -2.
    The 5-Sum of the array is 2.

    Based on idea to either include/exclude a +/- number  from max sum.
     */
    public long kSum(int[] nums, int k) {
        long sum = 0L;
        for (int num : nums) sum += Math.max(num, 0);

        for (int i = 0; i < nums.length; ++i) nums[i] = Math.abs(nums[i]);
        Arrays.sort(nums);

        //max PQ
        Queue<Pair<Long, Integer>> queue = new PriorityQueue<>((a, b) -> Long.compare(b.getKey(), a.getKey()));
        queue.offer(new Pair<>(sum - nums[0], 0));
        long result = sum;
        while (--k > 0) {
            Pair<Long, Integer> pair = queue.poll();
            result = pair.getKey();
            int idx = pair.getValue();
            if (idx < nums.length - 1) {
                queue.offer(new Pair<>(sum - nums[idx + 1] + nums[idx], idx + 1));
                queue.offer(new Pair<>(sum - nums[idx + 1], idx + 1));
            }
        }

        return result;
    }

    // Author: Anand
    public int halveArray(int[] nums) {
        PriorityQueue<BigDecimal> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int num : nums) pq.add(BigDecimal.valueOf((double) num).setScale(2, RoundingMode.HALF_UP));

        long ls = 0;
        for (int num : nums) ls += num;
        BigDecimal sum = BigDecimal.valueOf(ls);
        BigDecimal ns = sum;
        int cnt = 0;
        while (!pq.isEmpty()) {
            if (ns.compareTo(sum.divide(BigDecimal.valueOf(2))) <= 0) {
                return cnt;
            }

            BigDecimal greatest = pq.poll();
            ns = ns.subtract(greatest.divide(BigDecimal.valueOf(2)));
            pq.offer(greatest.divide(BigDecimal.valueOf(2)));
            cnt++;
        }
        return cnt;
    }

    // Author: Anand
    public int maximumProduct(int[] nums, int k) {
        int MOD = 1_000_000_000 + 7;

        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int num : nums) pq.offer(num);
        while (k > 0) {
            int num = pq.poll();
            num += 1;
            k--;
            pq.offer(num);
        }

        long prod = 1;
        while (!pq.isEmpty()) {
            prod = mod_mul(prod, pq.poll(), MOD);
        }
        return (int) prod;
    }

    public long mod_mul(long a, long b, long m) {
        a = a % m;
        b = b % m;
        return (((a * b) % m) + m) % m;
    }
}
