package com.company;

import java.util.ArrayList;
import java.util.List;

/**
 * ============================================================
 * PROBLEM: Design Bitset (LeetCode 2166)
 * ============================================================
 *
 * GOAL:
 *   Implement a fixed-size Bitset data structure that supports
 *   setting/unsetting individual bits, flipping ALL bits, and
 *   querying the count of 1s — all efficiently.
 *
 * KEY CHALLENGE:
 *   A naive flip() would require O(n) time to toggle every bit.
 *   We need flip() to run in O(1).
 *
 * CORE IDEA — Lazy Flip with a Boolean Flag:
 *   Instead of physically flipping every element in the array,
 *   we maintain a boolean flag `flip`.
 *
 *   - When flip = false → array values are the TRUE logical values.
 *   - When flip = true  → every stored value is logically INVERTED.
 *                         i.e., logical value = 1 - stored value.
 *
 *   This means flip() just toggles the flag and updates the count
 *   in O(1), without touching the array at all.
 *
 * STATE VARIABLES:
 *   - bitset   : the underlying integer array (stores 0s and 1s)
 *   - sz       : total size of the bitset
 *   - flip     : lazy flip flag (false by default)
 *   - cntone   : count of LOGICAL 1s (always kept up-to-date)
 *
 * OPERATIONS SUMMARY:
 *   ┌─────────────┬──────────────────────────────────────────────────────┐
 *   │ fix(idx)    │ Set logical bit[idx] = 1. Under lazy flip, this means│
 *   │             �� ensuring the stored value represents a logical 1.    │
 *   ├─────────────┼──────────────────────────────────────────────────────┤
 *   │ unfix(idx)  │ Set logical bit[idx] = 0. Mirror logic of fix().     │
 *   ├─────────────┼──────────────────────────────────────────────────────┤
 *   │ flip()      │ Toggle flag, update cntone = sz - cntone. O(1).      │
 *   ├─────────────┼──────────────────────────────────────────────────────┤
 *   │ all()       │ True if all sz bits are logical 1s.                  │
 *   ├─────────────┼──────────────────────────────────────────────────────┤
 *   │ one()       │ True if at least one bit is logical 1.               │
 *   ├─────────────┼──────────────────────────────────────────────────────┤
 *   │ count()     │ Return number of logical 1s.                         │
 *   ├─────────────┼──────────────────────────────────────────────────────┤
 *   │ toString()  │ Build the string; invert values on-the-fly if needed.│
 *   └─────────────┴──────────────────────────────────────────────────────┘
 *
 * TIME COMPLEXITY:
 *   - fix, unfix, all, one, count, toString → O(1) or O(n) for toString
 *   - flip → O(1)  ← the main advantage of this approach
 *
 * SPACE COMPLEXITY: O(n)
 * ============================================================
 */
// Author: Anand
public class Bitset {
    int cntone;   // Number of LOGICAL 1-bits (accounts for flip state)
    boolean flip; // Lazy flip flag: when true, stored values are logically inverted
    List<Integer> bitset; // Backing array storing raw (possibly inverted) bit values
    int sz;       // Total size of the bitset


    public Bitset(int size) {
        bitset = new ArrayList<>();
        sz = size;
        for (int i = 0; i < sz; i++) bitset.add(0); // Initialise all bits to 0
        flip = false;  // No flip applied yet
        cntone = 0;    // Zero 1-bits initially
    }

    /**
     * Sets the logical bit at idx to 1.
     *
     * Lazy Flip Logic:
     *   - flip=false → stored value IS the logical value.
     *                  If stored==0 (logical 0), change to 1 and increment cntone.
     *   - flip=true  → stored value is INVERTED (logical = 1 - stored).
     *                  A logical 1 requires stored==0.
     *                  If stored==1 (i.e., currently logical 0), flip it to 0 and increment cntone.
     */
    public void fix(int idx) {
        if (!flip) {
            if (bitset.get(idx) == 0) {  // Logical 0 → needs to become 1
                bitset.set(idx, 1);
                cntone++;
            }
        } else {
            // Under flip: stored 1 means logical 0, stored 0 means logical 1
            if (bitset.get(idx) == 1) {  // Logical 0 → needs to become logical 1 (store 0)
                bitset.set(idx, 0);
                cntone++;
            }
        }
    }

    /**
     * Sets the logical bit at idx to 0.
     *
     * Lazy Flip Logic (mirror of fix):
     *   - flip=false → stored value IS the logical value.
     *                  If stored==1 (logical 1), change to 0 and decrement cntone.
     *   - flip=true  → A logical 0 requires stored==1.
     *                  If stored==0 (i.e., currently logical 1), flip it to 1 and decrement cntone.
     */
    public void unfix(int idx) {
        if (!flip) {
            if (bitset.get(idx) == 1) {  // Logical 1 → needs to become 0
                bitset.set(idx, 0);
                cntone--;
            }
        } else {
            // Under flip: stored 0 means logical 1, stored 1 means logical 0
            if (bitset.get(idx) == 0) {  // Logical 1 → needs to become logical 0 (store 1)
                bitset.set(idx, 1);
                cntone--;
            }
        }
    }

    /**
     * Flips ALL bits in O(1) using the lazy flag.
     *
     * Instead of iterating over every element:
     *   1. Toggle the flip flag (future reads/writes interpret stored values inverted).
     *   2. Update cntone: after flip, every 0 becomes 1 and every 1 becomes 0,
     *      so new count of 1s = sz - old count of 1s.
     */
    public void flip() {
        flip = !flip;
        cntone = sz - cntone;
    }

    // Returns true if ALL sz bits are logical 1 (i.e., cntone equals total size)
    public boolean all() {
        return cntone == sz;
    }

    // Returns true if AT LEAST ONE bit is logical 1
    public boolean one() {
        return cntone > 0;
    }

    // Returns the exact count of logical 1-bits (maintained incrementally — O(1))
    public int count() {
        return cntone;
    }

    /**
     * Builds and returns the string representation of the bitset.
     *
     * - flip=false → append stored values directly.
     * - flip=true  → invert each stored value on-the-fly (0→1, 1→0).
     *
     * This is the only O(n) operation; all others are O(1).
     */
    public String toString() {
        StringBuilder s = new StringBuilder();
        if (!flip) {
            for (int e : bitset) {
                s.append(e);
            }
        } else {
            for (int e : bitset) {
                s.append(e == 0 ? 1 : 0); // Invert on-the-fly to reflect logical values
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
