package com.company;

import java.util.HashMap;
import java.util.Map;

public class XORProblemSet {
    // Author: Anand
    // Use XOR property a^a=0 then for all segments between i to j there must be a pivot (p) such that i^...p == p+1^...k.
    // Hence the total number of pairs would be count of those pivot ie. j-i;
    // TC = O(n2), SC = O(1)
    public int countTriplets(int[] arr) {
        int n = arr.length;
        int cnt = 0;
        int xor = 0;
        for (int i = 0; i < n; i++) {
            xor = arr[i];
            for (int j = i + 1; j < n; j++) {
                xor ^= arr[j];
                if (xor == 0) {
                    cnt += j - i;
                }
            }
        }

        return cnt;
    }

    /*
    Explanation for approach:

    you have an array : a[0], a[1].... a[n - 1]

    First things first:
    We need to understand small fact, if xor(a[0....i]) has appeared before at index j then it means xor(a[j+1.....i]) = 0
    Another fact, if xor(a[i....j]) = 0 so this subarray will add (j - i - 1) to the answer.

    Now say currently we are at index i and let xor([0...i]) = x.

    Now say x has occurred 3 times previously at indices (i1, i2, i3)

    our answer for i will be = (i - i1 - 1) + (i - i2 - 1) + (i - i3 - 1)

    if you simplify this further you get f * i - (i1 + i2 + i3) - f = (i - 1) * f - (i1 + i2 + i3)

    f = no. of times x has occurred previously.

    (i1 + i2 + i3) = sum of all the indices where x has occurred previously.
     */
    // TC = O(N), SC = O(N)
    public int countTripletsOptimised(int[] arr) {
        int n = arr.length;
        int cnt = 0;
        Map<Integer, Integer> count = new HashMap<>(), total = new HashMap<>();
        int prefix = 0, c, t;
        count.put(0, 1);

        for (int i = 0; i < n; i++) {
            prefix ^= arr[i];
            c = count.getOrDefault(prefix, 0);
            t = total.getOrDefault(prefix, 0);
            cnt += c * i - t;

            count.put(prefix, c + 1);
            total.put(prefix, t + i + 1);
        }

        return cnt;
    }

}
