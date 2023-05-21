package com.company;

import java.util.*;
import java.util.stream.Collectors;
import java.util.HashMap;

public class MapProblems {

    /*
        In question given -50 <= nums[i] <= 50 So, we have only -50 to -1 negative element
        and we need to calculate xth smallest negative element so we only deal with negative element in k size of sliding window.
       But one question like that How to calculate xth smallest negative element in current sliding windows bcz each time smallest element change?
       This can acheived by storing freq of numbers from -50 to -1
     */

    public int[] getSubarrayBeauty(int[] nums, int k, int x) {
        int n = nums.length;
        List<Integer> ans = new ArrayList<>();
        int[] freq = new int[51]; // to store freq of negative numbers from -50 to -1

        for (int i = 0, j = 0; i < n; i++) {
            // if number is -ve the update freq
            if (nums[i] < 0) freq[Math.abs(nums[i])]++;

            // current window of size=k
            if (i - j + 1 >= k) {
                int cnt = 0;
                for (int L = 50; L >= 1; L--) {
                    cnt += freq[L];
                    if (cnt >= x) {
                        ans.add(-L);
                        break;
                    }
                }

                // No xth smallest number was found in the window
                if (cnt < x) ans.add(0);
                if (nums[j] < 0) freq[Math.abs(nums[j])]--;
                j++;
            }
        }

        return ans.stream().mapToInt(Integer::intValue).toArray();

    }

    /*
    Input: nums = [1,2,4,6], operations = [[1,3],[4,7],[6,1]]
    Output: [3,2,7,1]
    Explanation: We perform the following operations on nums:
    - Replace the number 1 with 3. nums becomes [3,2,4,6].
    - Replace the number 4 with 7. nums becomes [3,2,7,6].
    - Replace the number 6 with 1. nums becomes [3,2,7,1].
    We return the final array [3,2,7,1].
 */
    //Author: Anand
    public int[] arrayChange(int[] nums, int[][] operations) {

        Map<Integer, Integer> map = new HashMap<>();

        int idx = 0;
        for (int num : nums) map.put(num, idx++);

        for (int[] o : operations) {
            int n1 = o[0];
            int n2 = o[1];
            int nidx = map.get(n1);
            map.remove(n1);
            map.put(n2, nidx);
        }


        Map<Integer, Integer> tmap = new TreeMap<>(map.entrySet().stream().collect(Collectors.toMap(Map.Entry::getValue, Map.Entry::getKey)));

        idx = 0;
        int[] ans = new int[nums.length];
        for (Map.Entry<Integer, Integer> entry : tmap.entrySet()) ans[idx++] = (int) entry.getValue();
        return ans;
    }

        /*
    Input: matches = [[1,3],[2,3],[3,6],[5,6],[5,7],[4,5],[4,8],[4,9],[10,4],[10,9]]
    Output: [[1,2,10],[4,5,7,8]]
    Explanation:
    Players 1, 2, and 10 have not lost any matches.
    Players 4, 5, 7, and 8 each have lost one match.
    Players 3, 6, and 9 each have lost two matches.
    Thus, answer[0] = [1,2,10] and answer[1] = [4,5,7,8].
     */
    // Author: Anand

    public List<List<Integer>> findWinners(int[][] matches) {
        Map<Integer, Integer> playerWinsMap = new HashMap<>();

        Map<Integer, Integer> playerLoseMap = new HashMap<>();

        for (int[] match : matches) {
            int w = match[0];
            int l = match[1];
            playerWinsMap.put(w, playerWinsMap.getOrDefault(w, 0) + 1);
            playerLoseMap.put(l, playerLoseMap.getOrDefault(l, 0) + 1);
        }

        List<List<Integer>> ans = new ArrayList<>();

        List<Integer> win = new ArrayList<>();
        List<Integer> loseOne = new ArrayList<>();

        for (Map.Entry entry : playerWinsMap.entrySet()) {
            if (!playerLoseMap.containsKey(entry.getKey())) win.add((int) entry.getKey());
        }

        for (Map.Entry entry : playerLoseMap.entrySet()) {
            if ((int) entry.getValue() == 1) loseOne.add((int) entry.getKey());
        }

        Collections.sort(win);
        Collections.sort(loseOne);
        ans.add(win);
        ans.add(loseOne);
        return ans;
    }
    /*
    Input: s = "ilovecodingonleetcode", target = "code"
    Output: 2
    Explanation:
    For the first copy of "code", take the letters at indices 4, 5, 6, and 7.
    For the second copy of "code", take the letters at indices 17, 18, 19, and 20.
    The strings that are formed are "ecod" and "code" which can both be rearranged into "code".
    We can make at most two copies of "code", so we return 2.
     */

    //Author: Anand
    public int rearrangeCharacters(String s, String target) {
        int ans = 0;
        Map<Character, Integer> om = new HashMap<>(), cm;

        for (int i = 0; i < target.length(); i++)
            om.put(target.charAt(i), om.getOrDefault(target.charAt(i), 0) + 1);

        cm = new HashMap<>(om);

        Set<Integer> indexTaken = new HashSet<>();
        for (int i = 0; i < s.length(); i++) {
            int idx = i;
            while (idx < s.length()) {
                char key = s.charAt(idx);
                if (cm.containsKey(key) && !indexTaken.contains(idx)) {
                    cm.put(key, cm.get(key) - 1);
                    if (cm.get(key) <= 0) cm.remove(key);
                    indexTaken.add(idx);
                }

                if (cm.size() == 0) {
                    cm = new HashMap<>(om);
                    ans++;
                    break;
                }
                idx++;
            }
        }
        return ans;
    }



    // Author: Anand
    public int firstUniqChar(String s) {
        Map<Character, List<Integer>> freq = new LinkedHashMap<>();
        for (int i = 0; i < s.length(); i++) {
            Character key = s.charAt(i);
            if (freq.containsKey(key)) {
                List<Integer> exist = freq.get(key);
                exist.add(i);
                freq.put(key, exist);
            } else freq.put(key, new ArrayList<>(Collections.singletonList(i)));
        }

        for (Map.Entry entry : freq.entrySet()) {
            if ((int) ((List<Integer>) entry.getValue()).size() == 1)
                return s.indexOf((Character) entry.getKey());
        }
        return -1;
    }

    // Author: Anand
    public char findTheDifference(String s, String t) {

        Map<Character, Integer> freq = new HashMap<>();

        for (int i = 0; i < s.length(); i++)
            freq.put(s.charAt(i), freq.getOrDefault(s.charAt(i), 0) + 1);

        for (int i = 0; i < t.length(); i++) {
            if (freq.containsKey(t.charAt(i))) {
                freq.put(t.charAt(i), freq.get(t.charAt(i)) - 1);
                if (freq.get(t.charAt(i)) <= 0) freq.remove(t.charAt(i));
            } else return t.charAt(i);
        }

        return '\n';
    }
}
