package com.company;

import java.awt.*;
import java.util.List;
import java.util.*;

public class Greedy {

    // TC = O(n)
    // Author: Anand
    // Ladder based greedy problem
    public static int minJumps(int[] arr) {
        int n = arr.length;

      /*
		    The variable 'curEnd stores the last index reachable from the current index
		    'curFarthest storest the farthest index reachable.
		*/
        int minJump = 0, curEnd = 0, curFarthest = 0;

        for (int i = 0; i < n; i++) {

            curFarthest = Math.max(curFarthest, i + arr[i]);

            if (i == curEnd) {
                // You had exhausted 1 jump completely
                minJump++;
                // update farthest index reachable
                curEnd = curFarthest;
            }

        }
        return minJump;
    }

    /*
    Input: nums = [3,9,3]
    Output: 2
    Explanation: Here are the steps to sort the array in non-decreasing order:
    - From [3,9,3], replace the 9 with 3 and 6 so the array becomes [3,3,6,3]
    - From [3,3,6,3], replace the 6 with 3 and 3 so the array becomes [3,3,3,3,3]
    There are 2 steps to sort the array in non-decreasing order. Therefore, we return 2.

    Greedy Approach :-
    - Reversely traverse the input nums, for each element,
       if it is greater than the previous one, then try to break into parts such that each part is no greater than the previous and as large as possible.
    - Initialize the prev to the largest int value as dummy value;
    - Reversely traverse nums, for each element, if it is greater than the previous one,
      compute how many parts we can break it into,
       then calculate the value of smallest part as prev,
        increase the number of operations by the number of parts - 1;
         if it is not greater than the previous, assign current element value to prev.
    */
    //Author: Anand
    public long minimumReplacement(int[] nums) {
        int prev = Integer.MAX_VALUE;
        long op = 0L;
        for (int i = nums.length - 1; i >= 0; i--) {
            int curr = nums[i];
            if (curr > prev) {
                int times = (curr + prev - 1) / prev;
                prev = curr / times;
                op += times - 1;
            } else prev = curr;
        }
        return op;
    }

    /*
    Input: plantTime = [1,4,3], growTime = [2,3,1]
    Output: 9

    Greedy approach:-
    Sort in  increasing order of growtime-plantime and then
    calculate total full blow time after traversal
    TC = O(nlogn), SC = O(n)
*/
    public int earliestFullBloom(int[] plantTime, int[] growTime) {
        int n = plantTime.length;
        int[][] fs = new int[n][2];
        for (int i = 0; i < n; i++) {
            fs[i] = new int[]{plantTime[i], growTime[i]};
        }
        // sort in increasing order of grow time
        Arrays.sort(fs, (f1, f2) -> f2[1] - f1[1]);

        int btime = 0, ptime = 0;
        for (int[] f : fs) {
            int fbtime = ptime + f[0] + f[1];
            btime = Math.max(fbtime, btime);
            ptime += f[0];
        }
        return btime;
    }

   /*
    Input: target = 19, maxDoubles = 2
    Output: 7
    Explanation: Initially, x = 1
    Increment 3 times so x = 4
    Double once so x = 8
    Increment once so x = 9
    Double again so x = 18
    Increment once so x = 19

    Greedy Approach
     */

    // Author: Anand
    // TC= O(n)
    public String[] divideString(String s, int k, char fill) {
        int n = s.length();
        String[] ans = new String[(int) (Math.floor(n / k)) + (n % k == 0 ? 0 : 1)];
        int index = 0;
        for (int i = 0; i < n; i += k) {
            StringBuilder group = new StringBuilder();
            if (i >= n - k) {
                group.append(s.substring(i));
                int loop = i - (n - k);
                while (loop-- > 0) {
                    group.append(fill);
                }
            } else {
                group.append(s, i, i + k);
            }
            ans[index++] = group.toString();
        }
        return ans;
    }

    // TC = O(logn)
    // Author: Anand
    public int minMoves(int target, int maxDoubles) {
        if (maxDoubles == 0) return target - 1;
        int ans = 0;
        while (target > 1) {
            if (maxDoubles > 0) {
                maxDoubles--;
                ans += 1 + (target % 2 == 0 ? 0 : 1); // Due to 1 operation to convert to even number
                target /= 2;
            } else {
                ans += target - 1;
                target = 1;
            }
        }
        return ans;
    }

    // TC = O(nlogn)
    public long maxRunTime(int n, int[] batteries) {
        long sum = 0L;
        for (int b : batteries) sum += b;
        Arrays.sort(batteries);
        int k = 0;
        // while max-power battery can be used for 1 of the computer
        while (batteries[batteries.length - 1 - k] > sum / (n - k))
            sum -= batteries[batteries.length - 1 - k++];

        return sum / (n - k);
    }

    /*
    Input: s = "aabb"
    Output: 2
    Explanation:
    We can obtain two palindromes from s, "abba" and "baab".
    - We can obtain "abba" from s in 2 moves: "aabb" -> "abab" -> "abba".
    - We can obtain "baab" from s in 2 moves: "aabb" -> "abab" -> "baab".
    Thus, the minimum number of moves needed to make s a palindrome is 2.
     */
    public int minMovesToMakePalindrome(String s) {
        int count = 0;

        while (s.length() > 2) {
            char ch1 = s.charAt(0);
            int len = s.length();
            char ch2 = s.charAt(len - 1);

            if (ch1 == ch2) {
                s = s.substring(1, len - 1);
            } else {
                int id1 = s.lastIndexOf(ch1);
                int id2 = s.indexOf(ch2);

                int steps1 = len - id1 - 1;
                int steps2 = id2;

                StringBuilder sb = new StringBuilder();

                if (steps1 > steps2) {
                    count += steps2;
                    sb.append(s, 0, id2);
                    sb.append(s, id2 + 1, len - 1);
                } else {
                    count += steps1;
                    sb.append(s, 1, id1);
                    sb.append(s.substring(id1 + 1));
                }

                s = sb.toString();
            }
        }

        return count;
    }

    // Author: Anand
    // TODO:- Complete it
    public int minimumWhiteTiles(String floor, int numCarpets, int carpetLen) {
        // Store c
        HashMap<Point, Integer> map = new HashMap<>(); // ((s-e), length)
        int stillVisible = 0;
        for (int i = 0; i < floor.length(); i++) {
            if (floor.charAt(i) == 1) stillVisible++;
        }

        int start = -1, end = -1;
        for (int i = 0; i < floor.length(); i++) {

            if (start == -1) {
                if (floor.charAt(i) == 1) {
                    start = i;
                }
            } else {
                if (floor.charAt(i) == 1) {
                    end = i;
                }
                map.put(new Point(start, end), end - start + 1);
                start = -1;
                end = -1;
            }
        }

        // Sort map based on values
        sortByValue(map);
        int canBeUsed = 0;
        for (Map.Entry<Point, Integer> aa : map.entrySet()) {
            int tiles = aa.getValue() / carpetLen;
            canBeUsed += aa.getValue() % carpetLen;
            stillVisible -= tiles;
            carpetLen -= tiles;
            if (carpetLen <= 0) return stillVisible;
        }
        return -1;
    }

    public HashMap<Point, Integer> sortByValue(HashMap<Point, Integer> hm) {


        List<Map.Entry<Point, Integer>> list = new LinkedList<Map.Entry<Point, Integer>>(hm.entrySet());

        Collections.sort(list, new Comparator<Map.Entry<Point, Integer>>() {
            public int compare(Map.Entry<Point, Integer> o1,
                               Map.Entry<Point, Integer> o2) {
                return (o1.getValue()).compareTo(o2.getValue());
            }
        });

        HashMap<Point, Integer> temp = new LinkedHashMap<Point, Integer>();
        for (Map.Entry<Point, Integer> aa : list) {
            temp.put(aa.getKey(), aa.getValue());
        }

        return temp;
    }

    /*
    Traverse from right to left of string and take all possible zeros
    by updating the count of 1 in the process
    return the longest Subsequence ie, less than k
     */

    public int longestSubsequenceGreedy(String s, int k) {

        int val = 0, cnt = 0, pow = 1;

        for (int i = s.length() - 1; i >= 0 && val + pow <= k; i--) {
            if (s.charAt(i) - '0' == 1) {
                val += pow;
                cnt++;
            }
            pow <<= 1;
        }

        return (int) s.chars().filter(x -> x - '0' == 0).count() + cnt;
    }

    /*
    Input: pattern = "IIIDIDDD"
    Output: "123549876"
    Explanation:
    At indices 0, 1, 2, and 4 we must have that num[i] < num[i+1].
    At indices 3, 5, 6, and 7 we must have that num[i] > num[i+1].
    Some possible values of num are "245639871", "135749862", and "123849765".
    It can be proven that "123549876" is the smallest possible num that meets the conditions.
    Note that "123414321" is not possible because the digit '1' is used more than once.

    Algorithm :-
    The idea is to use sliding window based technique, Keep 2 ptrs {i, j}
    Start with smallest possible string ie. 123456789
      then reverse the substring starts from j to the end of curr ans eg. ans = IIIID && i= 5, j=1 --> ans = ans.substring(0,j) + ans.substring(j).reverse();
      and place j to next starting point ie, j = i+1
    Return ans
     */
    // Author: Anand
    // Greedy approach
    public String smallestNumber(String pattern) {
        StringBuilder ans = new StringBuilder();
        for (int i = 0, j = 0; i <= pattern.length(); ++i) {
            ans.append((1 + i));

            // Reverse the prev substring starts from j
            if (i == pattern.length() || pattern.charAt(i) == 'I') {
                StringBuilder sb = new StringBuilder();
                sb.append(ans.substring(0, j)).append(new StringBuilder(ans.substring(j)).reverse());
                ans = sb;
                j = i + 1;
            }
        }

        return ans.toString();
    }

    /*
    Input: n = 3
    Output: 3
    Explanation: Initially, we have one character 'A'.
    In step 1, we use Copy All operation.
    In step 2, we use Paste operation to get 'AA'.
    In step 3, we use Paste operation to get 'AAA'.
     */
    //Author: Anand
    public int minSteps(int n) {
        if (n == 1) return 0;

        List<Integer> factors = new ArrayList<>();
        gf(n, factors);
        if (factors.size() == 0) return n;
        //Repeat 'A' till first factor times
        int first = factors.get(0);
        int op = first;
        op++; // copy the result
        factors.remove(0); // Removed from factors as this is done

        // For the remaining factors paste the copied result required times
        for (int f : factors) {
            op += (f / first) - 1;
            first = f;
            op++; // copy again
        }

        // paste for the final result
        if (first != n) {
            op += (n / first) - 1;
        }

        return op;
    }

    private void gf(int n, List<Integer> factors) {
        // base case
        if (n == 1) return;
        int i = 2;
        while (n % i != 0) {
            i++;
        }
        gf(n / i, factors);
        if (i != n) factors.add(n / i);
    }


    /*
    Play wisely -
    Always ban your next pos opponent while traversal from left to right
     */
    //Author: Anand
    public String predictPartyVictory(String senate) {

        int n = senate.length();
        if (n == 1) {
            if (senate.charAt(0) == 'R') return "Radiant";
            return "Dire";
        }
        Map<String, TreeMap<Integer, Boolean>> map = new HashMap<>();
        map.put("Radiant", new TreeMap<>());
        map.put("Dire", new TreeMap<>());

        for (int i = 0; i < senate.length(); i++) {
            if (senate.charAt(i) == 'R') map.get("Radiant").put(i, true);
            else map.get("Dire").put(i, true);
        }

        if (map.get("Radiant").size() > 0 && map.get("Dire").size() <= 0) return "Radiant";
        if (map.get("Dire").size() > 0 && map.get("Radiant").size() <= 0) return "Dire";

        while (senate.length() > 0) {
            List<Integer> blocked = new ArrayList<>();
            for (int i = 0; i < senate.length(); i++) {

                if (blocked.contains(i)) continue;

                if (senate.charAt(i) == 'R' && map.get("Dire").size() > 0) {
                    int key;
                    if (map.get("Dire").ceilingKey(i) == null) key = map.get("Dire").firstKey();
                    else key = map.get("Dire").ceilingKey(i);

                    blocked.add(key);
                    map.get("Dire").remove(key);
                } else if (senate.charAt(i) == 'D' && map.get("Radiant").size() > 0) {
                    int key;
                    if (map.get("Radiant").ceilingKey(i) == null) key = map.get("Radiant").firstKey();
                    else key = map.get("Radiant").ceilingKey(i);

                    blocked.add(key);
                    map.get("Radiant").remove(key);
                }
            }

            if (map.get("Radiant").size() > 0 && map.get("Dire").size() <= 0) return "Radiant";
            if (map.get("Dire").size() > 0 && map.get("Radiant").size() <= 0) return "Dire";


            StringBuilder sb = new StringBuilder();
            List<Integer> indR = new ArrayList<>(map.get("Radiant").keySet());
            Collections.sort(indR);
            List<Integer> indD = new ArrayList<>(map.get("Dire").keySet());
            Collections.sort(indD);
            int ind = 0;
            while (indR.size() > ind && indD.size() > ind) {
                if (indR.get(ind) > indD.get(ind)) {
                    sb.append('D');
                } else sb.append('R');
                ind++;
            }

            if (ind < indR.size()) {
                int cnt = indR.size() - ind;
                while (cnt-- > 0) sb.append('R');
            }

            if (ind < indD.size()) {
                int cnt = indD.size() - ind;
                while (cnt-- > 0) sb.append('D');
            }

            senate = sb.toString();
        }

        return "";
    }



}
