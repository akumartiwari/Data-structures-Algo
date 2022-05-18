package com.company;

import java.awt.*;
import java.util.*;
import java.util.List;

public class Greedy {
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

    // Author: Anand
    // TC= O(n)
    public String[] divideString(String s, int k, char fill) {
        int n = s.length();
        String[] ans = new String[(int) (Math.floor(n / k)) + (n % k == 0 ? 0 : 1)];
        int index = 0;
        for (int i = 0; i < n; i += k) {
            StringBuilder group = new StringBuilder("");
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
                    sb.append(s.substring(0, id2));
                    sb.append(s.substring(id2 + 1, len - 1));
                } else {
                    count += steps1;
                    sb.append(s.substring(1, id1));
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
            int tiles = (int) aa.getValue() / carpetLen;
            canBeUsed += (int) aa.getValue() % carpetLen;
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
}
