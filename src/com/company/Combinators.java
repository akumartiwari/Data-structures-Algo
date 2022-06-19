package com.company;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;

public class Combinators {
    /*
    Any idea = first letter + postfix string.
    We can group all ideas by their first letter.

    If two ideas ideas[i] and ideas[j] share a common postfix string,
    then ideas[i] can not pair with any idea starts with ideas[j][0]
    and ideas[j] can not pair with any idea starts with ideas[i][0].
   */
    public long distinctNames(String[] ideas) {
        HashSet<Integer>[] count = new HashSet[26];

        for (int i = 0; i < count.length; i++) count[i] = new HashSet<>();

        for (String s : ideas) {
            count[s.charAt(0) - 'a'].add(s.substring(1).hashCode());
        }

        long res = 0L;
        for (int i = 0; i < 26; i++) {
            for (int j = i + 1; j < 26; j++) {
                int c1 = 0, c2 = 0;
                for (int c : count[i]) {
                    if (!count[j].contains(c)) c1++;
                }

                for (int c : count[j]) {
                    if (!count[i].contains(c)) c2++;
                }

                res += (long) c1 * c2;
            }
        }
        return res * 2;
    }

    // Author: Anand
    public long numberOfWays(String s) {
        long ans = 0;

        int t0 = 0, t1 = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '0') t0++;
            else t1++;
        }

        int c0 = 0, c1 = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '0') {
                ans += (long) c1 * (t1 - c1);
                c0++;
            } else {
                ans += (long) c0 * (t1 - c0);
                c1++;
            }
        }

        return ans;
    }

    //Author: Anand
    public long maximumSubsequenceCount(String text, String pattern) {
        int n = text.length();
        char f = pattern.charAt(0);
        char l = pattern.charAt(1);
        long ans = 0;

        if (f == l) {
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                if (text.charAt(i) == f) cnt++;
            }
            return (long) cnt * (cnt + 1) / 2;
        }
        // Adding f at start
        Map<Integer, Integer> map = new LinkedHashMap<>(); // cnt of f before lth index
        int cntf = 1;
        for (int i = 0; i < n; i++) {
            if (text.charAt(i) == f) cntf++;
            else if (text.charAt(i) == l) {
                map.put(i, cntf);
                cntf = 0;
            }
        }

        int size = map.size();
        int curr = 0;
        for (Map.Entry entry : map.entrySet()) {
            ans += (long) (size - curr) * (int) entry.getValue();
            curr++;
        }

        // Adding l at last
        Map<Integer, Integer> mapl = new LinkedHashMap<>();// cnt of f before lth index
        int cntl = 0;
        String nt = text.concat(String.valueOf(l));
        for (int i = 0; i < nt.length(); i++) {
            if (nt.charAt(i) == f) cntl++;
            else if (nt.charAt(i) == l) {
                mapl.put(i, cntl);
                cntl = 0;
            }
        }

        long ansl = 0;
        int szl = mapl.size();
        int currl = 0;
        for (Map.Entry entry : mapl.entrySet()) {
            ansl += (long) (szl - currl) * (int) entry.getValue();
            currl++;
        }

        return Math.max(ansl, ans);
    }

}
