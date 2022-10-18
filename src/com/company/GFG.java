package com.company;

import java.io.IOException;
import java.util.*;

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


    /*
    input = [(1,4), (2,3)]
    return 3

    input = [(4,6), (1,2)]

    {[1,2],[4,6]}

    return 3

    [[1,4],[4,6],[6,8],[10,15]]
    3+2+2+5=12
    {{1,3}, {2,4}, {5,7}, {6,8}}.

    s = 1 ,e=4

    r[0][0]=1
    r[0][1]=4
    [1,4][5,8]
     */
    public static int mergeSegments(int[][] segments) {
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int result = 0;
        int last = 0;
        for (int[] seg : segments) {
            result += Math.max(seg[1] - Math.max(last, seg[0]), 0);
            last = Math.max(last, seg[1]);
        }
        return result;
    }


    public static int[][] mergeOverlapingSegments(int[][] segments) {
        // O(n), O(n*n)
        /*
        int n = segments.length;
        if (n == 0) return new int[n][n];
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int[][] result = new int[n][n];
        int index = 0;

        int start = Integer.MAX_VALUE;
        int end = Integer.MAX_VALUE;
        for (int[] seg : segments) {
            if (seg[0] < end) { // the value of start will not change
                end = seg[1];
                if (start == Integer.MAX_VALUE) start = seg[0];
            } else {
                result[index][0] = start;
                result[index][1] = end;
                index++;
                start = seg[0];
                end = seg[1];
            }
        }
        result[index][0] = start;
        result[index][1] = end;
        return result;

         */
        // Using stack
        int n = segments.length;
        if (n == 0) return new int[n][n];
        Arrays.sort(segments, Comparator.comparingInt(x -> x[0]));
        int[][] result = new int[n][n];


        return result;
    }


    class pair {
        int l, r, index;

        pair(int l, int r, int index) {
            this.l = l;
            this.r = r;
            this.index = index;
        }
    }

    void findOverlapSegement(int N, int[] a, int[] b) {

        ArrayList<pair> tuple = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            int x, y;
            x = a[i];
            y = b[i];
            tuple.add(new pair(x, y, i));
        }

        // sorted the tuple base on l values -> (leftmost value of tuple)
        Collections.sort(tuple, (aa, bb) -> (aa.l != bb.l) ? aa.l - bb.l : aa.r - bb.r);
        // store r-value 0f current
        int curr = tuple.get(0).r;
        // store index of current
        int curpos = tuple.get(0).index;

        for (int i = 1; i < N; i++) {
            pair currpair = new pair(tuple.get(i).l, tuple.get(i).r, tuple.get(i).index);

            // get L-value of prev
            int L = tuple.get(i - 1).l;
            int R = currpair.r;
            if (L == R) {
                if (tuple.get(i - 1).index < currpair.index)
                    System.out.println(tuple.get(i).index + " " + currpair.index);
                else System.out.println(currpair.index + " " + tuple.get(i).index);
                return;
            }

            if (currpair.r < curr) {
                System.out.print(tuple.get(i).index + " " + curpos);
                return;
            }
            // update the pos and the index
            else {
                curpos = currpair.index;
                curr = currpair.r;
            }

            // If such intervals found
            System.out.print("-1 -1");

        }

    }
}
