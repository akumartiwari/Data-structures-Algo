package com.company;

import java.util.TreeMap;

public class TreeMapExample {

    /*
       Input
       ["CountIntervals", "add", "add", "count", "add", "count"]
       [[], [2, 3], [7, 10], [], [5, 8], []]
       Output
       [null, null, null, 6, null, 8]

       Explanation
       CountIntervals countIntervals = new CountIntervals(); // initialize the object with an empty set of intervals.
       countIntervals.add(2, 3);  // add [2, 3] to the set of intervals.
       countIntervals.add(7, 10); // add [7, 10] to the set of intervals.
       countIntervals.count();    // return 6
                                  // the integers 2 and 3 are present in the interval [2, 3].
                                  // the integers 7, 8, 9, and 10 are present in the interval [7, 10].
       countIntervals.add(5, 8);  // add [5, 8] to the set of intervals.
       countIntervals.count();    // return 8
                                  // the integers 2 and 3 are present in the interval [2, 3].
                                  // the integers 5 and 6 are present in the interval [5, 8].
                                  // the integers 7 and 8 are present in the intervals [5, 8] and [7, 10].
                                  // the integers 9 and 10 are present in the interval [7, 10].

    */
    //Author: Anand
    // Interval treemap start -> finish.
    TreeMap<Integer, Integer> s;
    int count;

    public TreeMapExample() {
        s = new TreeMap<Integer, Integer>();
        count = 0;
    }

    public void add(int left, int right) {
        // Add interval if there is no overlapping.
        if (s.floorKey(right) == null || s.get(s.floorKey(right)) < left) {
            s.put(left, right);
            count += (right - left + 1);
        } else {
            int start = left;
            int end = right;

            // Remove overlapping intervals and update count.
            while (true) {
                int l = s.floorKey(end);
                int r = s.get(l);
                start = Math.min(start, l);
                end = Math.max(end, r);
                count -= (r - l + 1);
                s.remove(l);
                // Break the loop until there is no overlapping with interval (start, end).
                if (s.floorKey(end) == null || s.get(s.floorKey(end)) < start) {
                    break;
                }
            }
            // Add (start, end) to TreeMap and update count.
            s.put(start, end);
            count += (end - start + 1);
        }
    }

    public int count() {
        return count;
    }
/**
 * Your TreeMapExample object will be instantiated and called as such:
 * TreeMapExample obj = new TreeMapExample();
 * obj.add(left,right);
 * int param_2 = obj.count();
 */
}
