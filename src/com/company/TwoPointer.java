package com.company;

import java.util.*;

public class TwoPointer {


    /*
    Example 1:
    Input: str1 = "abc", str2 = "ad"
    Output: true
    Explanation: Select index 2 in str1.
    Increment str1[2] to become 'd'.
    Hence, str1 becomes "abd" and str2 is now a subsequence. Therefore, true is returned.

    Example 2:
    Input: str1 = "zc", str2 = "ad"
    Output: true
    Explanation: Select indices 0 and 1 in str1.
    Increment str1[0] to become 'a'.
    Increment str1[1] to become 'd'.
    Hence, str1 becomes "ad" and str2 is now a subsequence. Therefore, true is returned.

     */
    class Solution {

        //TC = O(N+contant)
        // SC = O(N*26)
        class RelativePos {
            int pos;
            List<Integer> indices;

            RelativePos(int pos, List<Integer> indices) {
                this.pos = pos;
                this.indices = indices;
            }
        }

        public boolean canMakeSubsequence(String str1, String str2) {
            Map<Character, RelativePos> map = new LinkedHashMap<>();
            for (int i = 0; i < str1.length(); i++) {
                char c = str1.charAt(i);
                if (!map.containsKey(c)) map.put(c, new RelativePos(0, new ArrayList<>()));
                RelativePos rp = map.get(c);
                rp.indices.add(i);
                map.put(c, rp);
            }

            System.out.println(map);

            //Traverse and check the remaining string
            //Might require recursion based technique as we need  to check the pending string
            return recursion(str1, str2, map, 0);
        }

        /**
         * This function return if possible or not
         *
         * @return
         */
        private boolean recursion(String str1, String str2, Map<Character, RelativePos> map, int ind) {
            for (int i = ind; i < str2.length(); i++) {
                char c = str2.charAt(i);
                if (!map.containsKey(c) && !map.containsKey((char) (c - 1))) {
                    return false;
                } else if (map.containsKey(c)) {
                    List<Integer> indices = map.get(c).indices;
                    int pos = map.get(c).pos;
                    if (pos > ind) continue;
                    else pos++;
                    map.put(c, new RelativePos(pos, indices));
                } else if (map.containsKey((char) (c - 1))) {
                    List<Integer> indices = map.get((char) (c - 1)).indices;
                    int pos = map.get((char) (c - 1)).pos;
                    if (pos > ind) continue;
                    else pos++;
                    map.put((char) (c - 1), new RelativePos(pos, indices));
                }

                recursion(str1, str2, map, i);
            }

            return true;
        }
    }


    /*

    Given a 0-indexed integer array nums of length n and an integer target,
    return the number of pairs (i, j) where 0 <= i < j < n and nums[i] + nums[j] < target.

    Input: nums = [-6,2,5,-2,-7,-1,3], target = -2
    Output: 10
    Explanation: There are 10 pairs of indices that satisfy the conditions in the statement:
    - (0, 1) since 0 < 1 and nums[0] + nums[1] = -4 < target
    - (0, 3) since 0 < 3 and nums[0] + nums[3] = -8 < target
    - (0, 4) since 0 < 4 and nums[0] + nums[4] = -13 < target
    - (0, 5) since 0 < 5 and nums[0] + nums[5] = -7 < target
    - (0, 6) since 0 < 6 and nums[0] + nums[6] = -3 < target
    - (1, 4) since 1 < 4 and nums[1] + nums[4] = -5 < target
    - (3, 4) since 3 < 4 and nums[3] + nums[4] = -9 < target
    - (3, 5) since 3 < 5 and nums[3] + nums[5] = -3 < target
    - (4, 5) since 4 < 5 and nums[4] + nums[5] = -8 < target
    - (4, 6) since 4 < 6 and nums[4] + nums[6] = -4 < target

     */
    public int countPairs(List<Integer> nums, int target) {
        Collections.sort(nums); // sort the vector nums
        int count = 0; // variable to store the count
        int left = 0; // variable to store the left
        int right = nums.size() - 1; // variable to store the right
        while (left < right) { // loop until left is less than right
            if (nums.get(left) + nums.get(right) < target) { // if nums[left] + nums[right] is less than target
                count += right - left; // update the count
                left++; // increment the left
            } else { // else
                right--; // decrement the right
            }
        }
        return count; // return the count

    }
}
