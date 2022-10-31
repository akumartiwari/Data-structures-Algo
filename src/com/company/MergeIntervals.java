package com.company;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public class MergeIntervals {
    /*
input = [(1,4), (2,3)]
return 3

input = [(4,6), (1,2)]

{[1,2],[4,6]}

return 3

[[1,4],[4,6],[6,8],[10,15]]
3+2+2+5=12
{{1,3}, {2,4}, {5,7}, {6,8}}.

s = 1 ,e=4

r[0][0]=1
r[0][1]=4
[1,4][5,8]
 */
    public static int mergeSegments(int[][] segments) {
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int result = 0;
        int last = 0;
        for (int[] seg : segments) {
            result += Math.max(seg[1] - Math.max(last, seg[0]), 0);
            last = Math.max(last, seg[1]);
        }
        return result;
    }

    public static int[][] mergeOverlapingSegments(int[][] segments) {
        // O(n), O(n*n)
        /*
        int n = segments.length;
        if (n == 0) return new int[n][n];
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int[][] result = new int[n][n];
        int index = 0;

        int start = Integer.MAX_VALUE;
        int end = Integer.MAX_VALUE;
        for (int[] seg : segments) {
            if (seg[0] < end) { // the value of start will not change
                end = seg[1];
                if (start == Integer.MAX_VALUE) start = seg[0];
            } else {
                result[index][0] = start;
                result[index][1] = end;
                index++;
                start = seg[0];
                end = seg[1];
            }
        }
        result[index][0] = start;
        result[index][1] = end;
        return result;

         */
        // Using stack
        int n = segments.length;
        if (n == 0) return new int[n][n];
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int[][] result = new int[n][n];


        return result;
    }

    // O(n+m), O(1)
	/*
     This problem is called Merge Sorted Array
     Example:-
     Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
     Output: [1,2,2,3,5,6]
     Algo :-
     We start filling the array from right end till all elements of nums1 is consumed
     After that remaining element of nums2 is utitlised
     */

    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
        while (j > -1) {
            if (i > -1) {
                nums1[k--] = nums1[i] > nums2[j] ? nums1[i--] : nums2[j--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
    }

    class pair {
        int l, r, index;

        pair(int l, int r, int index) {
            this.l = l;
            this.r = r;
            this.index = index;
        }
    }

    void findOverlapSegement(int N, int[] a, int[] b) {

        ArrayList<pair> tuple = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            int x, y;
            x = a[i];
            y = b[i];
            tuple.add(new pair(x, y, i));
        }

        // sorted the tuple base on l values -> (leftmost value of tuple)
        Collections.sort(tuple, (aa, bb) -> (aa.l != bb.l) ? aa.l - bb.l : aa.r - bb.r);
        // store r-value 0f current
        int curr = tuple.get(0).r;
        // store index of current
        int curpos = tuple.get(0).index;

        for (int i = 1; i < N; i++) {
            pair currpair = new pair(tuple.get(i).l, tuple.get(i).r, tuple.get(i).index);

            // get L-value of prev
            int L = tuple.get(i - 1).l;
            int R = currpair.r;
            if (L == R) {
                if (tuple.get(i - 1).index < currpair.index)
                    System.out.println(tuple.get(i).index + " " + currpair.index);
                else System.out.println(currpair.index + " " + tuple.get(i).index);
                return;
            }

            if (currpair.r < curr) {
                System.out.print(tuple.get(i).index + " " + curpos);
                return;
            }
            // update the pos and the index
            else {
                curpos = currpair.index;
                curr = currpair.r;
            }

            // If such intervals found
            System.out.print("-1 -1");

        }

    }
}
