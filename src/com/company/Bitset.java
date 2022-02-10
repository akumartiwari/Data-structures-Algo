package com.company;

import java.util.ArrayList;
import java.util.List;

// Author: Anand
public class Bitset {
    int cntone;
    boolean flip;
    List<Integer> bitset;
    int sz;


    public Bitset(int size) {
        bitset = new ArrayList<>();
        sz = size;
        for (int i = 0; i < sz; i++) bitset.add(0); // Initialise
        flip = false;
        cntone = 0;
    }

    // Updates the value of the bit at the index idx to 1. If the value was already 1, no change occurs.
    public void fix(int idx) {
        if (!flip) {
            if (bitset.get(idx) == 0) {
                bitset.set(idx, 1);
                cntone++;
            }
        } else {
            if (bitset.get(idx) == 1) {
                bitset.set(idx, 0);
                cntone++;
            }
        }
    }

    // Updates the value of the bit at the index idx to 0. If the value was already 0, no change occurs.
    public void unfix(int idx) {
        if (!flip) {
            if (bitset.get(idx) == 1) {
                bitset.set(idx, 0);
                cntone--;
            }
        } else {
            if (bitset.get(idx) == 0) {
                bitset.set(idx, 1);
                cntone--;
            }
        }
    }

    public void flip() {
        flip = !flip;
        cntone = sz - cntone;
    }

    public boolean all() {
        return cntone == sz;
    }

    public boolean one() {
        return cntone > 0;
    }

    public int count() {
        return cntone;
    }

    public String toString() {
        StringBuilder s = new StringBuilder();
        if (!flip) {
            for (int e : bitset) {
                s.append(e);
            }
        } else {
            for (int e : bitset) {
                s.append(e == 0 ? 1 : 0);
            }
        }
        return s.toString();
    }
}

/**
 * Your Bitset object will be instantiated and called as such:
 * Bitset obj = new Bitset(size);
 * obj.fix(idx);
 * obj.unfix(idx);
 * obj.flip();
 * boolean param_4 = obj.all();
 * boolean param_5 = obj.one();
 * int param_6 = obj.count();
 * String param_7 = obj.toString();
 */

