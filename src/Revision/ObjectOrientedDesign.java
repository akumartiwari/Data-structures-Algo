package com.company;

import java.util.HashMap;import java.util.*;

public class ObjectOrientedDesign {
    //Author: Anand
    class TextEditor {

        int cursorPos;
        StringBuilder sb;

        public TextEditor() {
            cursorPos = 0;
            sb = new StringBuilder();
        }

        /*
         * PROBLEM: Design a Text Editor (LeetCode 2296)
         * Insert the given text at the current cursor position.
         *
         * ALGORITHM: StringBuilder with cursor index
         * TC: O(n) | SC: O(n)
         */
        public void addText(String text) {
            sb.insert(cursorPos, text);
            cursorPos += text.length();
        }

        /*
         * PROBLEM: Design a Text Editor (LeetCode 2296)
         * Delete k characters to the left of the cursor; return the number of characters actually deleted.
         *
         * ALGORITHM: StringBuilder delete with bounded cursor
         * TC: O(k) | SC: O(1)
         */
        public int deleteText(int k) {
            int min = Math.min(k, cursorPos);
            cursorPos -= min;
            sb.delete(cursorPos, cursorPos + min);
            return min;
        }

        /*
         * PROBLEM: Design a Text Editor (LeetCode 2296)
         * Move the cursor k steps to the left; return the last min(10, cursor) characters to the left of the cursor.
         *
         * ALGORITHM: Bounded cursor movement on StringBuilder
         * TC: O(k) | SC: O(1)
         */
        public String cursorLeft(int k) {
            int min = Math.min(k, cursorPos);
            cursorPos -= min;
            return cursorPos < 10 ? sb.substring(0, cursorPos) : sb.substring(cursorPos - 10, cursorPos);
        }

        /*
         * PROBLEM: Design a Text Editor (LeetCode 2296)
         * Move the cursor k steps to the right; return the last min(10, cursor) characters to the left of the cursor.
         *
         * ALGORITHM: Bounded cursor movement on StringBuilder
         * TC: O(k) | SC: O(1)
         */
        public String cursorRight(int k) {
            cursorPos = Math.min(sb.length(), cursorPos + k);
            return cursorPos < 10 ? sb.substring(0, cursorPos) : sb.substring(cursorPos - 10, cursorPos);
        }
    }

    /**
     * Your TextEditor object will be instantiated and called as such:
     * TextEditor obj = new TextEditor();
     * obj.addText(text);
     * int param_2 = obj.deleteText(k);
     * String param_3 = obj.cursorLeft(k);
     * String param_4 = obj.cursorRight(k);
     */

    class LUPrefix {

        int n, l;
        TreeSet<Integer> set;

        public LUPrefix(int n) {
            this.n = n;
            l = 0;
            set = new TreeSet<>();
        }

        /*
         * PROBLEM: Design Video Sharing Platform (LeetCode 2254)
         * Upload a video and update the longest consecutive uploaded prefix starting from 1.
         *
         * ALGORITHM: TreeSet + linear prefix scan
         * TC: O(log n) | SC: O(n)
         */
        public void upload(int video) {
            set.add(video);
            if (video == 1 && l == 0) l = 1;
            int prev = l;
            while (set.contains(++prev)) {
            }
            l = --prev;
        }

        /*
         * PROBLEM: Design Video Sharing Platform (LeetCode 2254)
         * Return the length of the longest uploaded prefix (longest sequence 1..l all uploaded).
         *
         * ALGORITHM: Cached prefix length field
         * TC: O(1) | SC: O(1)
         */
        public int longest() {
            return l;
        }
    }

    /**
     * Your LUPrefix object will be instantiated and called as such:
     * LUPrefix obj = new LUPrefix(n);
     * obj.upload(video);
     * int param_2 = obj.longest();
     */


    class FindSumPairs {
        int[] nums1;
        int[] nums2;

        Map<Integer, Integer> map2;

        public FindSumPairs(int[] nums1, int[] nums2) {
            this.nums1 = nums1;
            this.nums2 = nums2;
            map2 = new HashMap<>();

            for (int num : nums2) map2.put(num, map2.getOrDefault(num, 0) + 1);
        }

        /*
         * PROBLEM: Finding Pairs With a Certain Sum (LeetCode 1865)
         * Add val to nums2[index] and update the frequency map to reflect the change.
         *
         * ALGORITHM: Frequency HashMap update
         * TC: O(1) | SC: O(1)
         */
        public void add(int index, int val) {
            map2.put(nums2[index], map2.getOrDefault(nums2[index], 0) - 1);
            if (map2.get(nums2[index]) <= 0) map2.remove(nums2[index]);
            nums2[index] += val;
            map2.put(nums2[index], map2.getOrDefault(nums2[index], 0) + 1);
        }

        /*
         * PROBLEM: Finding Pairs With a Certain Sum (LeetCode 1865)
         * Count pairs (i, j) such that nums1[i] + nums2[j] == tot.
         *
         * ALGORITHM: Frequency HashMap lookup
         * TC: O(n) | SC: O(1)
         */
        public int count(int tot) {
            int cnt = 0;
            for (int num : nums1) {
                if (map2.containsKey(tot - num))
                    cnt += map2.get(tot - num);
            }
            return cnt;
        }
    }

/**
 * Your FindSumPairs object will be instantiated and called as such:
 * FindSumPairs obj = new FindSumPairs(nums1, nums2);
 * obj.add(index,val);
 * int param_2 = obj.count(tot);
 */
}


class Allocator {

    int[] allocator;

    TreeMap<Integer, Integer> tm;

    public Allocator(int n) {
        allocator = new int[n];
        tm = new TreeMap<>();
        tm.put(0, n);
    }

    /*
     * PROBLEM: Design Memory Allocator (LeetCode 2502)
     * Allocate the first contiguous free block of the given size and tag it with mID; return its start index or -1.
     *
     * ALGORITHM: TreeMap of free ranges
     * TC: O(n) | SC: O(n)
     */
    public int allocate(int size, int mID) {
        int ind = -1;
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {
            if ((entry.getValue() - entry.getKey()) >= size) {
                ind = entry.getKey();
                break;
            }
        }

        if (ind != -1) {

            // place the next freed index
            int ni = ind + size;

            if (tm.ceilingKey(ni) == null) {
                tm.put(ni, allocator.length);
            }

            if (ind != -1 && tm.get(ind) > ni) {
                tm.put(ni, tm.get(ind));
            }

            tm.remove(ind);

            // maintain allocator
            for (int i = ind; i < ind + size; i++) {
                allocator[i] = mID;
            }
        }
        System.out.println(tm);
        return ind;
    }

    /*
     * PROBLEM: Design Memory Allocator (LeetCode 2502)
     * Free all memory blocks tagged with mID, merge adjacent free ranges, and return the count of freed units.
     *
     * ALGORITHM: Linear scan + TreeMap range merge
     * TC: O(n) | SC: O(n)
     */
    public int free(int mID) {
        int cnt = 0;
        System.out.println(Arrays.toString(allocator) + "size=" + allocator.length);
        int i = 0;
        while (i < allocator.length) {
            int ind = i;
            while (ind < allocator.length && allocator[ind] == mID) {
                allocator[ind] = 0;
                ind++;
                cnt++;
            }

            // maintain map
            if (ind > i) tm.put(i, ind);
            i = ind + 1;
        }

        Set<Integer> remove = new HashSet<>();
        //Map Merge operation
        int prevK = -1, prevV = -1;
        for (Map.Entry<Integer, Integer> entry : tm.entrySet()) {

            if (entry.getKey() == entry.getValue()) remove.add(entry.getKey());
            if (prevV == entry.getKey()) {
                tm.remove(prevK);
                tm.put(prevK, entry.getValue());
            }

            prevV = entry.getValue();
            prevK = entry.getKey();
        }
        for (int r : remove) tm.remove(r);
        System.out.println(tm);

        return cnt;
    }


    class DataStream {

        boolean lastKInt;
        int cntK;

        int value;
        int k;

        public DataStream(int value, int k) {
            cntK = 0;
            this.value = value;
            this.k = k;
        }

        /*
         * PROBLEM: Find Consecutive Integers from a Data Stream (LeetCode 2526)
         * Return true if the last k integers in the stream all equal the target value.
         *
         * ALGORITHM: Running counter reset on mismatch
         * TC: O(1) | SC: O(1)
         */
        public boolean consec(int num) {

            if (num == value) cntK++;

            else {
                cntK = 0;
            }

            return cntK >= k;
        }
    }

/**
 * Your DataStream object will be instantiated and called as such:
 * DataStream obj = new DataStream(value, k);
 * boolean param_1 = obj.consec(num);
 */
}
/**
 * Your Allocator object will be instantiated and called as such:
 * Allocator obj = new Allocator(n);
 * int param_1 = obj.allocate(size,mID);
 * int param_2 = obj.free(mID);
 */
