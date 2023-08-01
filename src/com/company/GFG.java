package com.company;

import java.io.IOException;
import java.util.HashMap;import java.util.*;

class GFG {

    // Pair class
    static class Pair {

        int first;
        int second;

        Pair(int first, int second) {
            this.first = first;
            this.second = second;
        }
    }

    static void SelectActivities(int s[], int f[]) {

        // Vector to store results.
        ArrayList<Pair> ans = new ArrayList<>();

        // Minimum Priority Queue to sort activities in
        // ascending order of finishing time (f[i]).
        PriorityQueue<Pair> p = new PriorityQueue<>(Comparator.comparingInt(p2 -> p2.first));

        for (int i = 0; i < s.length; i++) {
            // Pushing elements in priority queue where the
            // key is f[i]
            p.add(new Pair(f[i], s[i]));
        }

        Pair it = p.poll();
        int start = Objects.requireNonNull(it).second;
        int end = it.first;
        ans.add(new Pair(start, end));

        while (!p.isEmpty()) {
            Pair itr = p.poll();
            if (itr.second >= end) {
                start = itr.second;
                end = itr.first;
                ans.add(new Pair(start, end));
            }
        }
        System.out.println("Following Activities should be selected. \n");

        for (Pair itr : ans) {
            System.out.println("Activity started at: " + itr.first + " and ends at  " + itr.second);
        }
    }

    // Driver Code
    public static void main(String[] args) {

        int s[] = {1, 3, 0, 5, 8, 5};
        int f[] = {2, 4, 6, 7, 9, 9};

        // Function call
        SelectActivities(s, f);
    }

    /*
    Input: nums = [2,3,0,1,4]
Output: 2

   Greedy approach to solve problem
     TC = O(n), SC = O(1)
     njmi= 2
     cj = 1
     cjmi=2
     nj=2
     njmx=4
     */

    public int jump(int[] nums) {

        int n = nums.length;
        if (n == 0) return 0;

        int currjumps = 0, currjumpmaxindex = 0, nextjumps = 1, nextjumpmaxindex = 0;// variables to store jumps indexes

        for (int i = 0; i < n; i++) {
            if (i > currjumpmaxindex) { // case when you have consumed all indexes of currjumps
                currjumps = nextjumps;
                currjumpmaxindex = nextjumpmaxindex;
                nextjumps = currjumps + 1;
                nextjumpmaxindex = 0;
            }
            nextjumpmaxindex = Math.max(nums[i] + i, nextjumpmaxindex);
        }

        return currjumps;
    }

    public void format() throws IOException {
        //Scanner
        Scanner s = new Scanner(System.in);
        String variables = s.nextLine();                 // Reading input from STDIN
        int n = variables.split(",").length;
        HashMap<String, String> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            map.put(variables.split(",")[i], "");
        }

        for (int i = 0; i < n; i++) {
            String expression = s.nextLine();
            String[] kv = expression.split("=");
            String k = kv[0].trim();
            String v = kv[1].trim();
            map.put(map.getOrDefault(k, v), "");
        }
        // Traverse map and put ans
        StringBuilder ans = new StringBuilder();
        int index = 0;
        String lastValue = "";

        for (Map.Entry<String, String> key : map.entrySet()) {
            String expression = map.get(key);
            if (index == 0) {
                ans.append("1").append(key);
                lastValue = expression;
                index++;
            } else {
                // case 2 handled
                if (lastValue.replaceAll("[0-9]|\\s", "").equalsIgnoreCase(expression.replaceAll("[0-9]|\\s", ""))) {
                    int prev = Integer.parseInt(lastValue.replaceAll("[a-z|A-Z|\\s]", ""));
                    int curr = Integer.parseInt(expression.replaceAll("[a-z|A-Z|\\s]", ""));
                    if (key.equals(lastValue.replaceAll("[0-9]|\\s", ""))) ans.append(" = ").append(prev).append(key);
                    else ans.append(" = ").append(prev / curr).append(key);
                }
            }
        }
        System.out.println(ans);
    }

}
