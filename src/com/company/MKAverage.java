package com.company;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.TreeMap;

public class MKAverage {

    public static void main(String[] args) {
        int m = 3, k = 1, num = 3;
        MKAverage obj = new MKAverage(m, k);
        obj.addElement(num);
        System.out.println(obj.calculateMKAverage());
/*
["MKAverage","addElement","addElement","calculateMKAverage","addElement","calculateMKAverage","addElement","addElement","addElement","calculateMKAverage"]
[[3,1],[3],[1],[],[10],[],[5],[5],[5],[]]
 */
    }

    // TC = O(mklogm)
    // Author: Anand
    int tot = 0, sum = 0, m, k;
    Deque<Integer> deque = new ArrayDeque<>();
    TreeMap<Integer, Integer> map = new TreeMap<>(); // to store sorted values;

    public MKAverage(int m, int k) {
        this.m = m;
        this.k = k;
    }

    // TC = O(1)
    public void addElement(int num) {
        deque.offerLast(num);
        map.put(num, map.getOrDefault(num, 0) + 1);
        tot++;
        sum += num;

        if (tot > m) {
            int v = deque.pollFirst();
            sum -= v;
            tot--;
            if (map.get(v) == 1) map.remove(v);
            else map.put(v, map.get(v) - 1);
        }
    }

    // TC = O(k)
    public int calculateMKAverage() {
        if (tot < m) return -1;
        int totLess = k;
        int s = sum;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (totLess == 0) break;
            int v = Math.min(entry.getValue(), totLess);
            s -= v * entry.getKey();
            totLess -= v;
        }

        totLess = k;
        for (Map.Entry<Integer, Integer> entry : map.descendingMap().entrySet()) {
            if (totLess == 0) break;
            int v = Math.min(entry.getValue(), totLess);
            s -= v * entry.getKey();
            totLess -= v;
        }

        return Math.abs(s / (m - 2 * k));
    }
}
