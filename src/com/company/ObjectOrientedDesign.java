package com.company;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeSet;

public class ObjectOrientedDesign {
    //Author: Anand
    class TextEditor {

        int cursorPos;
        StringBuilder sb;

        public TextEditor() {
            cursorPos = 0;
            sb = new StringBuilder();
        }

        public void addText(String text) {
            sb.insert(cursorPos, text);
            cursorPos += text.length();
        }

        public int deleteText(int k) {
            int min = Math.min(k, cursorPos);
            cursorPos -= min;
            sb.delete(cursorPos, cursorPos + min);
            return min;
        }

        public String cursorLeft(int k) {
            int min = Math.min(k, cursorPos);
            cursorPos -= min;
            return cursorPos < 10 ? sb.substring(0, cursorPos) : sb.substring(cursorPos - 10, cursorPos);
        }

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

        public void upload(int video) {
            set.add(video);
            if (video == 1 && l == 0) l = 1;
            int prev = l;
            while (set.contains(++prev)) {
            }
            l = --prev;
        }

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

        public void add(int index, int val) {
            map2.put(nums2[index], map2.getOrDefault(nums2[index], 0) - 1);
            if (map2.get(nums2[index]) <= 0) map2.remove(nums2[index]);
            nums2[index] += val;
            map2.put(nums2[index], map2.getOrDefault(nums2[index], 0) + 1);
        }

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
