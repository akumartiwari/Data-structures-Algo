package com.company;

public class ToBeDiscovered {
    // Author: Anand
    public int minBitFlips(int start, int goal) {
        String bstart = Integer.toBinaryString(start);
        String bgoal = Integer.toBinaryString(goal);
        int len = Math.max(bstart.length(), bgoal.length());

        if (bstart.length() < len) {
            int cnt = len - bstart.length();
            StringBuilder newbstart = new StringBuilder(bstart);
            while (cnt-- > 0) {
                newbstart.insert(0, '0');
            }
            bstart = newbstart.toString();
        } else if (bgoal.length() < len) {
            int cnt = len - bgoal.length();
            StringBuilder newbgoal = new StringBuilder(bgoal);
            while (cnt-- > 0) {
                newbgoal.insert(0, '0');
            }
            bgoal = newbgoal.toString();
        }

        int ans = 0;
        for (int i = bgoal.length() - 1; i >= 0; i--) {
            if (bgoal.charAt(i) != bstart.charAt(i)) ans++;
        }
        return ans;
    }
}
