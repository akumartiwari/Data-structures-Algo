package com.company;

import java.util.*;

public class LIS {
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


    // Author: Anand
    // Tabulated solution
    // Rules
    /*
       - Fetch changing parameters and write them in rev order
       - Copy recurrence
       - Write base cases
       TC = O(n2), SC=O(n*2)
     */
    public int lengthOfLISTabulated(int[] nums) {
        int[] curr = new int[nums.length + 1], next = new int[nums.length + 1];
        for (int idx = nums.length - 1; idx >= 0; idx--) {
            for (int prev_idx = idx - 1; prev_idx >= -1; prev_idx--) {
                // not-take
                int len = next[prev_idx + 1];
                if (prev_idx == -1 || nums[idx] > nums[prev_idx]) {
                    len = Math.max(len, 1 + next[idx + 1]);
                }
                curr[prev_idx + 1] = len;
            }
            next = curr;
        }

        return curr[-1 + 1];
    }

    // print the LIS
    /*
        dp[n] and initialise with size = 1
        dp[i] signifies the length of  LIS till index i
        i -> 0 to n
        prev -> 0 to i
        Recurrence :-
        dp[i] = Math.max(dp[prev]+1, dp[i]);
       return dp[n-1] as max length of LIS
     */
    public void printLIS(int[] arr) {
        int[] dp = new int[arr.length];
        int[] hash = new int[arr.length];
        Arrays.fill(dp, 1);
        int len = -1;
        int li = 0;
        for (int i = 0; i < arr.length; i++) {
            hash[i] = i;
            for (int prev = 0; prev < i; prev++) {
                if (arr[prev] < arr[i]) {
                    hash[i] = prev;
                    dp[i] = Math.max(dp[prev] + 1, dp[i]);
                }
            }

            if (len < dp[i]) {
                len = dp[i];
                li = i;
            }
        }

        // hash is ready to backtrack
        List<Integer> temp = new ArrayList<>();
        temp.add(arr[li]);
        while (hash[li] != li) {
            li = hash[li];
            temp.add(arr[li]);
        }
        Collections.reverse(temp);
        System.out.println(temp);
    }


    // print the Longest Divisible Subsequence
    /*
       I = [ 1 2 4 7 8 9]
       O = [1 2 4 8]
       ie. either arr[i]%arr[j]==0 || arr[j]%arr[i]==0
     */
    public void printLongestDivisibleSubsequence(int[] arr) {
        Arrays.sort(arr);
        int[] dp = new int[arr.length];
        int[] hash = new int[arr.length];
        Arrays.fill(dp, 1);
        int len = -1;
        int li = 0;
        for (int i = 0; i < arr.length; i++) {
            hash[i] = i;
            for (int prev = 0; prev < i; prev++) {
                if (arr[i] % arr[prev] == 0 && dp[prev] + 1 > dp[i]) {
                    hash[i] = prev;
                    dp[i] = dp[prev] + 1;
                }
            }

            if (len < dp[i]) {
                len = dp[i];
                li = i;
            }
        }

        // hash is ready to backtrack
        List<Integer> temp = new ArrayList<>();
        temp.add(arr[li]);
        while (hash[li] != li) {
            li = hash[li];
            temp.add(arr[li]);
        }
        Collections.reverse(temp);
        System.out.println(temp);
    }


    // print the Longest Chain String
    /*
       I = ["a", "ab", "acb", "acd", "abcd"]
       O = ["a", "ab", "acb"]
       TC = O(n2*(max length of words))
       SC = O(n)
     */

    public void printLongestChainString(String[] arr) {
        // compare based on length of words
        Arrays.sort(arr, (a, b) -> Integer.compare(a.length(), b.length()));
        int[] dp = new int[arr.length];
        int[] hash = new int[arr.length];
        Arrays.fill(dp, 1);
        int len = -1;
        int li = 0;
        for (int i = 0; i < arr.length; i++) {
            hash[i] = i;
            for (int prev = 0; prev < i; prev++) {
                if (checkLength(arr[i], arr[prev]) && dp[prev] + 1 > dp[i]) {
                    hash[i] = prev;
                    dp[i] = dp[prev] + 1;
                }
            }

            if (len < dp[i]) {
                len = dp[i];
                li = i;
            }
        }

        // hash is ready to backtrack
        List<String> temp = new ArrayList<>();
        temp.add(arr[li]);
        while (hash[li] != li) {
            li = hash[li];
            temp.add(arr[li]);
        }
        Collections.reverse(temp);
        System.out.println(temp);
    }

    private boolean checkLength(String word1, String word2) {
        // use 2 pointer technique
        int first = 0, second = 0;
        while (first < word1.length() && second < word2.length()) {
            if (word1.charAt(first) == word2.charAt(second)) {
                first++;
                second++;
            } else {
                first++;
            }
        }

        return first == word1.length() && second == word2.length();
    }

    /*
       I = [ 1 2 4 7 8 9]
       O = [1 2 4 8]
       Bitonic means increasing then descreaing OR only increasing OR only descreaing
       Find LIS from l-> r
       Find LIS from r-> l
       Get the point where max lenth was found
       return max_length
       TC = O(n2)
       SC = O(n)
     */

    public int longestBitonicSubsequence(int[] arr) {
        int n = arr.length;
        // compare based on length of words
        int[] dp1 = new int[arr.length];
        int[] dp2 = new int[arr.length];
        Arrays.fill(dp1, 1);
        Arrays.fill(dp2, 1);

        for (int i = 0; i < arr.length; i++) {
            for (int prev = 0; prev < i; prev++) {
                if (arr[i] > arr[prev] && dp1[prev] + 1 > dp1[i]) {
                    dp1[i] = dp1[prev] + 1;
                }
            }
        }

        int maxi = 0;
        for (int i = n - 1; i >= 0; i--) {
            for (int prev = n - 1; prev > i; prev--) {
                if (arr[i] > arr[prev] && dp2[prev] + 1 > dp2[i]) {
                    dp2[i] = dp2[prev] + 1;
                }
            }
            maxi = Math.max(maxi, dp1[i] + dp2[i] - 1);
        }

        return maxi;
    }

    /*
        I = [1 3 6 5 7]
        O = 2 ## { [1 3 5 7], [1 3 6 7]} ##
        This gives count of longest length LIS subsequence
        TC = O(n2)
        SC = O(n)
     */
    public int countLISSubsequence(int[] arr) {
        int n = arr.length;
        int[] dp = new int[arr.length];
        int[] count = new int[arr.length];
        Arrays.fill(dp, 1);
        Arrays.fill(count, 1);

        int maxi = 1; // length of LIS
        for (int i = 0; i < arr.length; i++) {
            for (int prev = 0; prev < i; prev++) {
                if (arr[i] > arr[prev] && dp[prev] + 1 > dp[i]) {
                    dp[i] = dp[prev] + 1;
                    count[i] = count[prev];
                } else if (dp[prev] + 1 > dp[i]) {
                    count[i] += count[prev];
                }
            }

            maxi = Math.max(maxi, dp[i]);
        }

        int cntOfLIS = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == maxi) cntOfLIS += count[i];
        }

        return cntOfLIS;
    }
}


