package com.company;

import java.util.*;
import java.util.stream.Collectors;

public class RecursionPatterns {
    private static Set<List<Integer>> ans;

    public static void main(String[] args) {
        int[] arr = {2, 3, 0, 1, 4};
//        int n = 4;
//        int targetSum = 7;
//        List<Integer> ds = new ArrayList<>();
//        printAllSubsequences(arr, targetSum, 0, 0, ds);
//        printAnyoneSubsequence(arr, targetSum, 0, 0, ds);
//        System.out.println();
//        System.out.println("Count: " + printCountSubsequences(arr, targetSum, 0, 0));
//        combinationSum(arr, targetSum);
//        System.out.println((int) ans.stream().filter(x -> x.size() > 0).count());
//        for (List<Integer> list : ans) {
//            System.out.println(list.stream().map(Object::toString).collect(Collectors.joining(", ")));
//        }

        System.out.println("jumps=" + minJumps(arr));
    }

    /***
     * Function to print all subsequences of a given sum
     * @param arr
     * @param sum
     * @param ds
     */
    private static void printAllSubsequences(int[] arr, int targetSum, int idx, int sum, List<Integer> ds) {
        // base case
        if (idx == arr.length) {
            if (sum == targetSum) {
                ds.forEach(x -> System.out.print(x + " "));
                System.out.println();
            }
            return;
        }


        // take
        sum += arr[idx];
        ds.add(arr[idx]);
        printAllSubsequences(arr, targetSum, idx + 1, sum, ds);
        sum -= arr[idx];
        ds.remove(ds.size() - 1); // remove last element
        // not-take
        printAllSubsequences(arr, targetSum, idx + 1, sum, ds);
    }

    /***
     * Function to print single subsequence of a given sum
     * @param arr
     * @param sum
     * @param ds
     */
    private static boolean printAnyoneSubsequence(int[] arr, int targetSum, int idx, int sum, List<Integer> ds) {
        // base case
        if (idx == arr.length) {
            if (sum == targetSum) {
                ds.forEach(x -> System.out.print(x + " "));
                return true;
            }
            return false;
        }


        // take
        sum += arr[idx];
        ds.add(arr[idx]);
        if (printAnyoneSubsequence(arr, targetSum, idx + 1, sum, ds)) return true;
        sum -= arr[idx];
        ds.remove(ds.size() - 1); // remove last element
        // not-take
        return printAnyoneSubsequence(arr, targetSum, idx + 1, sum, ds);
    }

    /***
     * Function to count all subsequences of a given sum
     * @param arr
     * @param sum
     */
    private static int printCountSubsequences(int[] arr, int targetSum, int idx, int sum) {
        // base case
        // Only can be done if array contains positive elements
        if (sum > targetSum) return 0;
        if (idx == arr.length) {
            if (sum == targetSum) {
                return 1;
            }
            return 0;
        }


        // take
        sum += arr[idx];
        int take = printCountSubsequences(arr, targetSum, idx + 1, sum);
        sum -= arr[idx];
        // not-take
        int notTake = printCountSubsequences(arr, targetSum, idx + 1, sum);
        return take + notTake;
    }

    // Author : Anand
    // TC = O(n2^n), SC=O(n)

    /****
     * This is based on the fact that same elem can be taken as many times as we want
     * till sum <  targetSum
     * @param candidates
     * @param target
     * @return
     */
//    Input: candidates = [2,3,6,7], target = 7
//    Output: [[2,2,3],[7]]
//    Input: candidates = [2,3,5], target = 8
//    Output: [[2,2,2,2],[2,3,3],[3,5]]
    public static List<List<Integer>> combinationSum(int[] candidates, int target) {

        ans = new HashSet<>();
        List<Integer> ds = new ArrayList<>();
        combinations(candidates, target, 0, 0, ds);
        return new ArrayList<>(ans);
    }

    private static void combinations(int[] arr, int targetSum, int idx, int sum, List<Integer> ds) {

        // base case
        if (sum == targetSum) {
            ans.add(new ArrayList<>(ds));
        } else if (sum > targetSum) return;
        else {
            for (int i = idx; i < arr.length; i++) {
                // take
                sum += arr[i];
                ds.add(arr[i]);
                combinations(arr, targetSum, i, sum, ds);
                sum -= arr[i];
                ds.remove(new Integer(arr[i])); // remove last element
            }
        }
    }

    /****
     * Get all factors of a number
     * @param n
     * @return
     */

//    Input: n = 12
//    Output: [[2,6],[3,4],[2,2,3]]


    /*
    ds = [2]
    fact = 2
    2 to 6

     */

    // Author: Anand
    // TC = O(n), SC = O(n)

    // Important thing is to get all factors greater than the current one
    // for that divide the current num from idx and the count all factors recursively for remaining iterations
    public List<List<Integer>> getFactors(int n) {
        if (n <= 2) return new ArrayList<>();

        return factors(n, 2);
    }

    private List<List<Integer>> factors(int num, int fact) {

        List<List<Integer>> ans = new ArrayList<>();

        // Iterate from 2 till sqrt of num as factors can only exist upto sqrt of num

        int end = (int) Math.sqrt(num);
        for (int idx = fact; idx <= end; idx++) {
            int y = num / idx;
            int z = num % idx;

            // Always look for greater number while finding index
            if (y < idx) break;
            else if (z == 0) {
                List<Integer> parts = new ArrayList<>();
                parts.add(idx);
                parts.add(y);
                ans.add(parts);
                List<List<Integer>> list = factors(y, idx);
                for (List<Integer> factorsRelativeToPart : list) {
                    factorsRelativeToPart.add(idx);
                    ans.add(factorsRelativeToPart);
                }
            }
        }
        return ans;
    }

    //Author: Anand
    public boolean checkString(String s) {
        int n = s.length();
        boolean aFlag = true;
        for (int i = 0; i < n; i++) {
            if (aFlag && s.charAt(i) == 'b') {
                aFlag = false;
                continue;
            }
            if (!aFlag && s.charAt(i) == 'a') return false;
        }
        return true;
    }

    // Author : Anand
    public int numberOfBeams(String[] bank) {
        int n = bank.length;
        int ans = 0;
        HashMap<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int one = countOne(bank[i]);
            map.put(i, one);
        }

        int prev = map.get(0);
        for (int i = 1; i < n; i++) {
            if (map.get(i) > 0) {
                ans += prev * map.get(i);
                prev = map.get(i);
            }
        }

        return ans;
    }


    private int countOne(String str) {
        int cnt = 0;
        for (int i = 0; i < str.length(); i++) {
            if (str.charAt(i) == '1') cnt++;
        }
        return cnt;
    }

    // Author : Anand
    // It is based on BS of closest matching value recursively
    // TC = O(nlogn), SC = O(n)
    public boolean asteroidsDestroyed(int mass, int[] asteroids) {
        int n = asteroids.length;
        List<Integer> coll = Arrays.stream(asteroids).boxed().sorted().collect(Collectors.toList());

        long nMass = mass;
        while (n-- > 0) {
            int idx = closestMass(coll, nMass);
            if ((long) coll.get(idx) > nMass) {
                return false;
            }
            nMass += (long) coll.get(idx);
            coll.remove(idx); // O(1)
        }

        return n <= 0;
    }

    private int closestMass(List<Integer> asteroids, long mass) {
        int l = 0, r = asteroids.size() - 1;
        while (l <= r) {
            int mid = (int) Math.abs(l + (r - l) / 2);
            if ((long) asteroids.get(mid) > mass) {
                r = mid - 1;
            } else if ((long) asteroids.get(mid) < mass) {
                if (l != mid) l = mid;
                else return l;
            } else {
                return mid;
            }
        }

        return l;
    }

    // TODO:- REDO it
    // ALGO:- find cycle + DFS
    //Author:Anand
    //Directed acyclic graph based problem
    // TC = O(n)
    public int maximumInvitations(int[] favorite) {
        int n = favorite.length;
        Set<Integer> visited = new HashSet<>();
        // initiate dependencies
        Map<Integer, Set<Integer>> childToParents = new HashMap<>();
        for (int i = 0; i < favorite.length; i++) {
            childToParents.computeIfAbsent(favorite[i], k -> new HashSet<>());
            childToParents.get(favorite[i]).add(i);
        }

        int max = 0;
        // all the cycles with size 2, along with its connected chain, can fit into a table
        int size2Together = 0;
        for (int i = 0; i < n; i++) {
            if (visited.contains(i))
                continue;

            // cycleSize & cycleEntryPoint
            int[] tableMeta = findCycle(i, favorite, visited);
            if (tableMeta[0] == 2) {
                childToParents.get(tableMeta[1]).remove(favorite[tableMeta[1]]);
                childToParents.get(favorite[tableMeta[1]]).remove(tableMeta[1]);

                tableMeta[0] = dfs(tableMeta[1], childToParents, visited) + dfs(favorite[tableMeta[1]], childToParents, visited);
                size2Together += tableMeta[0];
            }
            max = Math.max(max, tableMeta[0]);
        }

        return Math.max(max, size2Together);

    }

    // return : new int[] {cycleSize, entryPoint}
    private int[] findCycle(int startPoint, int[] favorite, Set<Integer> visited) {
        int next = startPoint;
        int entryPoint = -1;
        int cycleSize = 0;

        // find entry point of cycle
        while (entryPoint == -1) {
            visited.add(next);
            cycleSize++;
            next = favorite[next];
            if (visited.contains(next))
                entryPoint = next;
        }

        // remove the segment from startPoint to entryPoint
        next = startPoint;
        while (next != entryPoint) {
            cycleSize--;
            next = favorite[next];
        }

        return new int[]{cycleSize, entryPoint};
    }

    private int dfs(int child, Map<Integer, Set<Integer>> childToParents, Set<Integer> visited) {
        visited.add(child);

        if (!childToParents.containsKey(child) || childToParents.get(child).isEmpty())
            return 1;

        int max = 0;
        for (int parent : childToParents.get(child)) {
            max = Math.max(max, dfs(parent, childToParents, visited) + 1);
        }

        return max;
    }

    //  Author : Anand
//        Approach is base don the fact that if we can segregate the array into k subparts and
//        each part will have min operations  (length-LIS)
//        to make it non-decreasing
//        then the final answer is overall combination of parts operations.
//        [12,6,12,6,14,2,13,17,3,8,11,7,4,11,18,8,8,3]
//        1
    // TC = O(nlogn), SC = O(n)
    public int kIncreasing(int[] arr, int k) {
        int n = arr.length;
        int total = 0;
        List<List<Integer>> subsequences = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            List<Integer> part = new ArrayList<>();
            for (int j = i; j < n; j += k) {
                part.add(arr[j]);
            }
            subsequences.add(part);
        }

        for (List<Integer> part : subsequences) {
            total += part.size() - LIS(part);
        }

        return total;
    }

    private int LIS(List<Integer> part) {
        List<Integer> ans = new ArrayList<>();
        int lastItem = part.get(0);
        for (Integer integer : part) {
            if (integer >= lastItem) {
                ans.add(integer);
            } else {
                // next greater element than current one in the ans list
                int idx = nextGreaterElement(ans, integer);
                ans.set(idx, integer);
            }
            lastItem = ans.get(ans.size() - 1);
        }

        return ans.size();
    }

    private int nextGreaterElement(List<Integer> ans, Integer item) {

        int l = 0, r = ans.size() - 1;
        while (l < r) {
            int mid = (int) Math.abs(l + (r - l) / 2);
            if (ans.get(mid) <= item) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        return l;
    }

    // Recursion
    // Apply memoisation
    // TC = O(2^n)
    public static int frogJumpRecursion(int n, int heights[]) {
        return helper(0, heights);
    }

    private static int helper(int index, int[] heights) {
        if (index == (heights.length - 1)) {
            // Reached the last stair.
            return 0;
        }

        // Two available choices.
        int oneJump = Integer.MAX_VALUE;
        int twoJump = Integer.MAX_VALUE;

        if ((index + 1) < heights.length) {
            oneJump = Math.abs(heights[index] - heights[index + 1]) + helper(index + 1, heights);
        }

        if ((index + 2) < heights.length) {
            twoJump = Math.abs(heights[index] - heights[index + 2]) + helper(index + 2, heights);
        }

        return Math.min(oneJump, twoJump);
    }

    /*
   Input: questions = [[3,2],[4,3],[4,4],[2,5]]
   Output: 5
   Explanation: The maximum points can be earned by solving questions 0 and 3.
   - Solve question 0: Earn 3 points, will be unable to solve the next 2 questions
   - Unable to solve questions 1 and 2
   - Solve question 3: Earn 2 points
   Total points earned: 3 + 2 = 5. There is no other way to earn 5 or more points.
   TC = O(n)
    Author: Anand
  */
    public long mostPoints(int[][] questions) {
        if (questions.length == 0) return 0L;
        long[] dp = new long[questions.length];
        Arrays.fill(dp, -1);
        return recursive(questions, 0, dp);
    }

    /***
     * Recursive function to get max points
     * @param questions
     * @param index
     * @return
     */
    private long recursive(int[][] questions, int index, long[] dp) {
        // base case
        if (index >= questions.length) return 0;

        if (dp[index] != -1) return dp[index];
        // take
        long left = questions[index][0] + recursive(questions, index + 1 + questions[index][1], dp);

        //skip
        long right = recursive(questions, index + 1, dp);

        return dp[index] = Math.max(left, right);
    }

    // Author: Anand
    // TC = O(n2)
    public static int minJumps(int[] arr) {
        if (arr.length == 0) return 0;
        int[] dp = new int[arr.length];
        Arrays.fill(dp, -1);
        return dfs(arr, 0, dp);
    }


    private static int dfs(int[] arr, int index, int[] dp) {
        // base-case
        // If you move out of array that jump is invalid and hence count=0
        if (index >= arr.length - 1) return 0;

        // If precomputed use it
        if (dp[index] != -1) return dp[index];

        int min = 1_0000_000;
        int ca = 0;
        // For all possible jumps we need to consider all possible choices to be made
        for (int i = 1; i <= arr[index]; i++) {

            int nextPos = index + i;

            if (nextPos < arr.length) {
                // Take
                ca = 1 + dfs(arr, nextPos, dp);
            }

            min = Math.min(ca, min);
            dp[index] = min;
        }

        return min;
    }

    /*

     Input: n = 5, k = 2
     Output: 3

     Sol:-
     friends = [1 3 4 5]
     ind = 0

     TC = O(n). SC = O(n) ie. For arraylist
     Author: Anand
   */
    public int findTheWinner(int n, int k) {
        List<Integer> friends = new ArrayList<>();
        // Store all person in an ArrayList
        for (int i = 1; i <= n; i++) friends.add(i);

        return dfs(n, k, friends, 0);
    }

    private int dfs(int n, int k, List<Integer> friends, int ind) {
        if (friends.size() == 1) return friends.get(0);
        int ni = (ind + k - 1) % friends.size();
        friends.remove(ni);
        return dfs(n, k, friends, ni);
    }

    // Author : Anand
    List<Long> res = new ArrayList<>();
    boolean solFound = false;

    public List<Long> maximumEvenSplit(long finalSum) {
        if (finalSum % 2 != 0) return res;
        if (finalSum == 2) {
            res.add(2L);
            return res;
        }

        mes(finalSum, 2, 0, new ArrayList<>());
        return res;
    }

    private void mes(long finalSum, long num, long sum, List<Long> ds) {
        // base case
        if (solFound) {
            return;
        }

        if (sum == finalSum) {
            res = new ArrayList<>(ds);
            solFound = true;
            return;
        }


        if (sum + num > finalSum) return;

        // not-take
        if (sum + num <= finalSum) {
            //take
            ds.add(num);
            mes(finalSum, num + 2, sum + num, ds);
            // backtrack
            ds.remove(ds.size() - 1);
        }

        mes(finalSum, num + 2, sum, ds);
    }



}