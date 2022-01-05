package com.company;

import java.util.*;
import java.util.stream.Collectors;

public class RecursionPatterns {
    private static Set<List<Integer>> ans;

    public static void main(String[] args) {
        int[] arr = {2, 3, 6, 7};
        int n = 4;
        int targetSum = 7;
        List<Integer> ds = new ArrayList<>();
        printAllSubsequences(arr, targetSum, 0, 0, ds);
        printAnyoneSubsequence(arr, targetSum, 0, 0, ds);
        System.out.println();
        System.out.println("Count: " + printCountSubsequences(arr, targetSum, 0, 0));
        combinationSum(arr, targetSum);
        System.out.println((int) ans.stream().filter(x -> x.size() > 0).count());
        for (List<Integer> list : ans) {
            System.out.println(list.stream().map(Object::toString).collect(Collectors.joining(", ")));
        }
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
    // TODO: Optimise it for nlogn  solution
    // TC = O(N^2), SC = O(n)
    public boolean asteroidsDestroyed(int mass, int[] asteroids) {
        int n = asteroids.length;
        Arrays.sort(asteroids);

        List<Integer> coll = Arrays.stream(asteroids).boxed().collect(Collectors.toList());
        while (n-- > 0) {
            int idx = closestMass(coll, mass);
            if (idx >= 0) {
                if (coll.get(idx) > mass) return false;
                mass += coll.get(idx);
                coll.remove(coll.get(idx)); // O(n)
            }
        }

        return n <= 0;
    }

    private int closestMass(List<Integer> asteroids, int mass) {
        int l = 0, r = asteroids.size() - 1;
        while (l < r) {
            int mid = (int) Math.abs(l + (r - l) / 2);
            if (asteroids.get(mid) > mass) {
                r = mid - 1;
            } else if (asteroids.get(mid) < mass) {
                if (l != mid) l = mid;
                else return l;
            } else {
                return mid;
            }
        }

        return l;
    }
}
