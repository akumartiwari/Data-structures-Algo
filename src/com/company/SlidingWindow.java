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
}
