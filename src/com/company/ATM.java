package com.company;

import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

public class ATM {
    Map<Integer, Long> notes;

    public ATM() {
        notes = new TreeMap<>();
        notes.put(20, 0L);
        notes.put(50, 0L);
        notes.put(100, 0L);
        notes.put(200, 0L);
        notes.put(500, 0L);
    }

    public void deposit(int[] banknotesCount) {
        int idx = 0;
        for (Map.Entry entry : notes.entrySet()) {
            notes.put((int) entry.getKey(), (long) entry.getValue() + banknotesCount[idx++]);
        }
    }

    public int[] withdraw(int amount) {
        int[] ans = new int[5];
        int idx = 4;
        // Use reverseOrder() method in the constructor
        TreeMap<Integer, Long> treeMap = new TreeMap<>(Collections.reverseOrder());
        treeMap.putAll(notes);
        for (Map.Entry entry : treeMap.entrySet()) {
            if (amount > 0) {
                int key = (int) entry.getKey();
                long value = (long) entry.getValue();
                long cnt = Math.min(amount / key, value);
                amount -= key * cnt;
                ans[idx--] = (int) cnt;
            } else break;
        }

        if (amount == 0) {
            int i = 0;
            for (Map.Entry entry : notes.entrySet()) {
                notes.put((int) entry.getKey(), (long) entry.getValue() - ans[i++]);
            }
            return ans;
        }
        return new int[]{-1};
    }
}

/**
 * Your ATM object will be instantiated and called as such:
 * ATM obj = new ATM();
 * obj.deposit(banknotesCount);
 * int[] param_2 = obj.withdraw(amount);
 */
