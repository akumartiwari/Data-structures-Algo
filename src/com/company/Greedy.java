package com.company;

import java.util.Arrays;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Queue;

public class Greedy {
    /*
    Input: plantTime = [1,4,3], growTime = [2,3,1]
    Output: 9

    Greedy approach:-
    Sort in  increasing order of growtime-plantime and then
    calculate total full blow time after traversal
    TC = O(nlogn), SC = O(n)
*/
    public int earliestFullBloom(int[] plantTime, int[] growTime) {
        int n = plantTime.length;
        int[][] fs = new int[n][2];
        for (int i = 0; i < n; i++) {
            fs[i] = new int[]{plantTime[i], growTime[i]};
        }
        // sort in increasing order of grow time
        Arrays.sort(fs, (f1, f2) -> f2[1] - f1[1]);

        int btime = 0, ptime = 0;
        for (int[] f : fs) {
            int fbtime = ptime + f[0] + f[1];
            btime = Math.max(fbtime, btime);
            ptime += f[0];
        }
        return btime;
    }

    // Author: Anand
    // TC= O(n)
    public String[] divideString(String s, int k, char fill) {
        int n = s.length();
        String[] ans = new String[(int) (Math.floor(n / k)) + (n % k == 0 ? 0 : 1)];
        int index = 0;
        for (int i = 0; i < n; i += k) {
            StringBuilder group = new StringBuilder("");
            if (i >= n - k) {
                group.append(s.substring(i));
                int loop = i - (n - k);
                while (loop-- > 0) {
                    group.append(fill);
                }
            } else {
                group.append(s, i, i + k);
            }
            ans[index++] = group.toString();
        }
        return ans;
    }

   /*
    Input: target = 19, maxDoubles = 2
    Output: 7
    Explanation: Initially, x = 1
    Increment 3 times so x = 4
    Double once so x = 8
    Increment once so x = 9
    Double again so x = 18
    Increment once so x = 19

    Greedy Approach
     */

    // TC = O(logn)
    // Author: Anand
    public int minMoves(int target, int maxDoubles) {
        if (maxDoubles == 0) return target - 1;
        int ans = 0;
        while (target > 1) {
            if (maxDoubles > 0) {
                maxDoubles--;
                ans += 1 + (target % 2 == 0 ? 0 : 1); // Due to 1 operation to convert to even number
                target /= 2;
            } else {
                ans += target - 1;
                target = 1;
            }
        }
        return ans;
    }

    // TC = O(nlogn)
    public long maxRunTime(int n, int[] batteries) {
        long sum = 0L;
        for (int b : batteries) sum += b;
        Arrays.sort(batteries);
        int k = 0;
        // while max-power battery can be used for 1 of the computer
        while (batteries[batteries.length - 1 - k] > sum / (n - k))
            sum -= batteries[batteries.length - 1 - k++];

        return sum / (n - k);
    }
}
