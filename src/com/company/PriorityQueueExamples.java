package com.company;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.PriorityQueue;

public class PriorityQueueExamples {

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
