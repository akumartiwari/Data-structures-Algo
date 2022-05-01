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
}
