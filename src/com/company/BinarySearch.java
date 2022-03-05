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
}
