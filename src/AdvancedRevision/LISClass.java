package AdvancedRevision;

import java.util.*;

public class LISClass {

    /*
     * -----------------------------------------------------------------------
     * PROBLEM: Longest Increasing Subsequence — Optimised
     * (LeetCode 300 - Medium, O(N log N) approach)
     * -----------------------------------------------------------------------
     * Given a list of integers, return the LENGTH of the Longest Strictly
     * Increasing Subsequence (LIS). Elements don't need to be contiguous.
     *
     * Example:
     *   Input : [1, 3, 2, 4]
     *   Output: 3    (one valid LIS: [1, 2, 4] or [1, 3, 4])
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Greedy + Binary Search — Patience Sorting concept)
     * -----------------------------------------------------------------------
     * Maintain a list `ans` where ans[k] holds the SMALLEST possible tail
     * element of all increasing subsequences of length k+1 seen so far.
     * This greedy choice maximises our chances of extending the subsequence.
     *
     * For each element `integer` in the input:
     *   - If integer >= lastItem (last element of ans):
     *       → Append it — extends the longest subsequence found so far.
     *   - Else:
     *       → Find the leftmost position in `ans` where ans[pos] >= integer
     *         using binary search (nextGreaterElement).
     *       → Replace ans[pos] with integer — this keeps the tail as small
     *         as possible without changing the subsequence length.
     *   - Update lastItem = ans.last().
     *
     * The LENGTH of `ans` at the end equals the LIS length.
     * (Note: `ans` itself may NOT be a valid subsequence — it only tracks tails.)
     *
     * Step-by-step (input = [1, 3, 2, 4]):
     *
     *   elem=1: ans empty → append 1.         ans=[1],    lastItem=1
     *   elem=3: 3>=1 → append 3.              ans=[1,3],  lastItem=3
     *   elem=2: 2<3  → bsearch in [1,3] for first >=2 → idx=1, replace 3 with 2.
     *                                          ans=[1,2],  lastItem=2
     *   elem=4: 4>=2 → append 4.              ans=[1,2,4],lastItem=4
     *
     *   LIS length = ans.size() = 3 ✓
     *
     * TC = O(N log N)  — N elements, binary search each time
     * SC = O(N)        — `ans` list holds at most N elements
     * -----------------------------------------------------------------------
     */
    private int LIS(List<Integer> part) {
        List<Integer> ans = new ArrayList<>();
        int lastItem = part.get(0);

        for (Integer integer : part) {
            if (integer >= lastItem) {
                ans.add(integer);
            } else {
                // next greater element than current one in the ans list
                int idx = nextGreaterElement(ans, integer);
                if (idx < 0) idx = -(idx + 1);
                if (idx == ans.size()) {
                    ans.add(integer);
                } else ans.set(idx, integer);
            }
            lastItem = ans.get(ans.size() - 1);
        }

        return ans.size();
    }

    /*
     * -----------------------------------------------------------------------
     * HELPER: nextGreaterElement (Binary Search — Lower Bound)
     * -----------------------------------------------------------------------
     * Finds the index of the LEFTMOST element in the sorted list `ans` that
     * is >= `item`. This is the standard "lower bound" binary search used by
     * the Greedy LIS algorithm to decide which position to replace.
     *
     * Example:
     *   ans=[1, 2, 4], item=3
     *   - l=0, r=2, mid=1: ans[1]=2 < 3 → l=2
     *   - l=2, r=2: exit loop → return 2
     *   → position 2 (where 4 lives) will be replaced with 3.
     *
     * The caller handles the edge case where idx < 0 using -(idx+1),
     * matching the convention of Java's Collections.binarySearch().
     *
     * TC = O(log N)  SC = O(1)
     * -----------------------------------------------------------------------
     */
    private int nextGreaterElement(List<Integer> ans, Integer item) {

        int l = 0, r = ans.size() - 1;
        while (l < r) {
            int mid = (int) Math.abs(l + (r - l) / 2);
            if (ans.get(mid) < item) {
                l = mid + 1;
            } else if (ans.get(mid) == item) {
                return l;
            } else r = mid;
        }

        return l;
    }


    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 300 - Medium): Length of LIS — Space-Optimised DP
     * -----------------------------------------------------------------------
     * Same problem as above but solved with bottom-up (tabulated) DP using
     * two rolling 1D arrays instead of a full 2D table to save space.
     *
     * State:
     *   dp[idx][prev_idx] = max LIS length considering elements from index
     *   `idx` onward, where `prev_idx` is the index of the last chosen element
     *   (-1 means no element chosen yet).
     *
     *   Since prev_idx ranges from -1 to n-1, we shift by +1 to fit in arrays:
     *   curr[prev_idx + 1] and next[prev_idx + 1].
     *
     * Recurrence (bottom-up, idx from n-1 → 0):
     *   not-take: len = next[prev_idx + 1]
     *   take    : if prev_idx == -1 OR nums[idx] > nums[prev_idx]
     *               len = max(len, 1 + next[idx + 1])
     *   curr[prev_idx + 1] = len
     *
     * Example (nums = [1, 2, 3], n=3):
     *
     *   Start: curr=next=[0, 0, 0, 0]  (size n+1)
     *
     *   idx=2 (val=3):
     *     prev=1: 3>2 → curr[2]=max(next[2], 1+next[3])=1
     *     prev=0: 3>1 → curr[1]=max(next[1], 1+next[3])=1
     *     prev=-1:      curr[0]=max(next[0], 1+next[3])=1
     *     next=[1,1,1,0]
     *
     *   idx=1 (val=2):
     *     prev=0: 2>1 → curr[1]=max(next[1], 1+next[2])=max(1, 2)=2
     *     prev=-1:      curr[0]=max(next[0], 1+next[2])=max(1, 2)=2
     *     next=[2,2,1,0]
     *
     *   idx=0 (val=1):
     *     prev=-1: curr[0]=max(next[0], 1+next[1])=max(2, 3)=3
     *     next=[3,2,1,0]
     *
     *   Answer = curr[0] = 3 ✓  (LIS: [1, 2, 3])
     *
     * TC = O(N²)  — two nested loops
     * SC = O(N)   — only two 1D rolling arrays instead of the full N×N table
     * -----------------------------------------------------------------------
     */
    // Author: Anand
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

    /*
     * -----------------------------------------------------------------------
     * PROBLEM: Print the Actual Longest Increasing Subsequence (LIS)
     * -----------------------------------------------------------------------
     * Given an integer array, find and PRINT one valid Longest Strictly
     * Increasing Subsequence (not just its length).
     *
     * Example:
     *   Input : arr = [5, 1, 4, 2, 3]
     *   Output: [1, 2, 3]   (length 3)
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (DP + Hash-Array Backtracking)
     * -----------------------------------------------------------------------
     * Two arrays:
     *   dp[i]   = length of the LIS ending exactly at index i.
     *   hash[i] = index of the previous element in the LIS ending at i
     *             (set to i itself if i is the starting point of its chain).
     *
     * Fill dp[] and hash[] (left to right, O(N²)):
     *   For each i, scan all prev < i:
     *     if arr[prev] < arr[i] and dp[prev]+1 > dp[i]:
     *       dp[i] = dp[prev]+1  ,  hash[i] = prev
     *
     * Track the index `li` with the maximum dp value (end of the longest chain).
     *
     * Backtrack using hash[] starting from li until hash[li] == li (self-loop
     * marks the start of the chain). Collect elements, then reverse.
     *
     * Step-by-step (arr = [5, 1, 4, 2, 3]):
     *
     *   i=0 (5): no prev       → dp[0]=1, hash[0]=0.  li=0, len=1
     *   i=1 (1): 5>1 (skip)   → dp[1]=1, hash[1]=1
     *   i=2 (4): 5>4 (skip); 1<4 → dp[2]=2, hash[2]=1.  li=2, len=2
     *   i=3 (2): 5>2; 1<2 → dp[3]=2, hash[3]=1; 4>2 (skip) → dp[3]=2, hash[3]=1
     *   i=4 (3): 1<3 → dp[4]=2; 2<3 and dp[3]+1=3>2 → dp[4]=3, hash[4]=3. li=4, len=3
     *
     *   dp   = [1, 1, 2, 2, 3]
     *   hash = [0, 1, 1, 1, 3]
     *
     *   Backtrack from li=4:
     *     temp=[arr[4]]=[ 3],  hash[4]=3≠4 → li=3
     *     temp=[3, arr[3]]=[3,2],  hash[3]=1≠3 → li=1
     *     temp=[3, 2, arr[1]]=[3,2,1],  hash[1]=1 → stop (self-loop)
     *
     *   Reverse → [1, 2, 3] ✓
     *
     * TC = O(N²)  — double loop to fill dp[]
     * SC = O(N)   — dp[] and hash[] arrays
     * -----------------------------------------------------------------------
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


    /*
     * -----------------------------------------------------------------------
     * PROBLEM (GFG / LeetCode 368 variant): Print Longest Divisible Subsequence
     * -----------------------------------------------------------------------
     * Given an array of positive integers, find and PRINT the longest
     * subsequence such that for every consecutive pair (a, b) in the
     * subsequence, a divides b (i.e., b % a == 0).
     *
     * Example:
     *   Input : arr = [1, 2, 4, 7, 8, 9]
     *   Output: [1, 2, 4, 8]
     *
     *   Chain: 1 | 2 | 4 | 8  (each element divides the next)
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Sort + DP + Hash-Array Backtracking)
     * -----------------------------------------------------------------------
     * Key insight: Sort the array first. If arr[i] % arr[prev] == 0 and
     * arr[prev] is in a valid chain, then arr[i] can extend that chain.
     * (Sorting ensures we always check smaller potential divisors first.)
     *
     * Two arrays:
     *   dp[i]   = length of the longest divisible subsequence ending at arr[i].
     *   hash[i] = previous index in the chain ending at i (or i if start).
     *
     * Fill dp[] (left to right, O(N²)):
     *   For each i, scan all prev < i:
     *     if arr[i] % arr[prev] == 0 and dp[prev]+1 > dp[i]:
     *       dp[i] = dp[prev]+1  ,  hash[i] = prev
     *
     * Step-by-step (arr = [1, 2, 4, 7, 8, 9] — already sorted):
     *
     *   i=0 (1): dp[0]=1, hash[0]=0
     *   i=1 (2): 2%1==0 → dp[1]=2, hash[1]=0
     *   i=2 (4): 4%1==0 → dp[2]=2; 4%2==0 and dp[1]+1=3>2 → dp[2]=3, hash[2]=1
     *   i=3 (7): 7%1==0 → dp[3]=2; 7%2≠0; 7%4≠0            → dp[3]=2, hash[3]=0
     *   i=4 (8): 8%1==0 → dp[4]=2; 8%2==0 → dp[4]=3;
     *            8%4==0 and dp[2]+1=4>3  → dp[4]=4, hash[4]=2
     *   i=5 (9): 9%1==0 → dp[5]=2; rest don't divide        → dp[5]=2
     *
     *   dp   = [1, 2, 3, 2, 4, 2]
     *   hash = [0, 0, 1, 0, 2, 0]
     *
     *   Max dp = dp[4]=4, li=4 (arr[4]=8)
     *
     *   Backtrack from li=4:
     *     temp=[8],   hash[4]=2≠4 → li=2
     *     temp=[8,4], hash[2]=1≠2 → li=1
     *     temp=[8,4,2], hash[1]=0≠1 → li=0
     *     temp=[8,4,2,1], hash[0]=0 → stop
     *
     *   Reverse → [1, 2, 4, 8] ✓
     *
     * TC = O(N² + N log N)  — sort + double DP loop
     * SC = O(N)             — dp[] and hash[] arrays
     * -----------------------------------------------------------------------
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


    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 1048 - Medium): Longest String Chain
     * -----------------------------------------------------------------------
     * Given a list of words, find the LONGEST chain of words where each word
     * is a predecessor of the next. Word A is a predecessor of word B if you
     * can insert exactly one letter anywhere in A to get B.
     *
     * Example:
     *   Input : ["a", "ab", "acb", "acd", "abcd"]
     *   Output: ["a", "ab", "acb"]  (chain length = 3)
     *
     *   "a" → insert 'b' → "ab" → insert 'c' at index 1 → "acb"  ✓
     *   (alternatively "a" → "ab" → "abcd" is length 3 too)
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Sort by length + DP + Hash-Array Backtracking + Two Pointers)
     * -----------------------------------------------------------------------
     * Key insight: Sort words by length. A predecessor always has length
     * exactly one less than its successor. Then apply LIS-style DP where
     * the "strictly increasing" condition is replaced by the "predecessor" check.
     *
     * checkLength(word1, word2) — Two Pointer:
     *   Checks if word2 (shorter) is a subsequence of word1 (longer) AND
     *   all characters of word1 are consumed. This validates that word2 can
     *   be obtained from word1 by removing exactly one character.
     *   - Advance `first` (word1 pointer) always; advance `second` (word2 pointer)
     *     only when characters match.
     *   - Return true if both reach their ends simultaneously.
     *
     * DP arrays:
     *   dp[i]   = length of the longest chain ending at arr[i].
     *   hash[i] = index of the previous word in the chain (i if chain start).
     *
     * Fill dp[] (left to right after sorting, O(N² × L) where L = max word length):
     *   For each i, scan all prev < i:
     *     if checkLength(arr[i], arr[prev]) and dp[prev]+1 > dp[i]:
     *       dp[i] = dp[prev]+1  ,  hash[i] = prev
     *
     * Step-by-step (sorted: ["a","ab","acb","acd","abcd"]):
     *
     *   i=0 ("a"):    no prev → dp[0]=1, hash[0]=0
     *   i=1 ("ab"):   check("ab","a")=true → dp[1]=2, hash[1]=0
     *   i=2 ("acb"):  check("acb","a")=true → dp[2]=2;
     *                 check("acb","ab")=true and dp[1]+1=3>2 → dp[2]=3, hash[2]=1
     *   i=3 ("acd"):  check("acd","a")=true → dp[3]=2;
     *                 check("acd","ab")=false; check("acd","acb")=false → dp[3]=2
     *   i=4 ("abcd"): check("abcd","acb")=true and dp[2]+1=4>1 → dp[4]=4, hash[4]=2
     *                 (etc.)
     *
     *   dp   = [1, 2, 3, 2, 4]
     *   hash = [0, 0, 1, 0, 2]
     *
     *   Max dp = dp[4]=4, li=4 ("abcd")
     *
     *   Backtrack from li=4:
     *     temp=["abcd"], hash[4]=2≠4 → li=2
     *     temp=["abcd","acb"], hash[2]=1≠2 → li=1
     *     temp=["abcd","acb","ab"], hash[1]=0≠1 → li=0
     *     temp=["abcd","acb","ab","a"], hash[0]=0 → stop
     *
     *   Reverse → ["a", "ab", "acb", "abcd"] ✓  (length 4)
     *
     * TC = O(N² × L + N log N)  — sort + N² pairs × length comparison
     * SC = O(N)                  — dp[] and hash[] arrays
     * -----------------------------------------------------------------------
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
     * -----------------------------------------------------------------------
     * PROBLEM (GFG): Longest Bitonic Subsequence
     * -----------------------------------------------------------------------
     * A subsequence is BITONIC if it first strictly increases and then strictly
     * decreases (either part can be empty — purely increasing or purely
     * decreasing also counts).
     * Given an integer array, return the length of the Longest Bitonic Subsequence.
     *
     * Example:
     *   Input : arr = [1, 4, 2]
     *   Output: 3
     *
     *   The whole array [1, 4, 2] is bitonic: increases 1→4, then decreases 4→2.
     *
     *   Another example:
     *   Input : arr = [1, 2, 4, 7, 8, 9]  →  Output: 6  (purely increasing)
     *   Input : arr = [1, 11, 2, 10, 4, 5, 2, 1]  →  Output: 6  ([1,2,10,4,2,1])
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Two-pass LIS DP)
     * -----------------------------------------------------------------------
     * Key insight: A bitonic subsequence has a PEAK element. Everything to the
     * left of the peak forms an increasing subsequence; everything to the right
     * forms a decreasing subsequence.
     *
     * Step 1 — Compute dp1[] (LIS length ending at each index, left to right):
     *   dp1[i] = length of LIS ending at arr[i] (increasing from left)
     *
     * Step 2 — Compute dp2[] (LIS length starting at each index, right to left):
     *   dp2[i] = length of LDS starting at arr[i] (decreasing to right)
     *   This is equivalent to computing LIS from right to left.
     *
     * Step 3 — For each index i, treating i as the PEAK:
     *   bitonic_length = dp1[i] + dp2[i] - 1  (subtract 1: arr[i] counted twice)
     *   Answer = max over all i.
     *
     * Step-by-step (arr = [1, 4, 2]):
     *
     *   dp1 (LIS left→right):
     *     i=0: dp1[0]=1
     *     i=1: arr[0]=1<4 → dp1[1]=2
     *     i=2: arr[0]=1<2 → dp1[2]=2; arr[1]=4>2 (not <) → dp1[2]=2
     *     dp1 = [1, 2, 2]
     *
     *   dp2 (LDS — LIS right→left):
     *     i=2: dp2[2]=1
     *     i=1: arr[1]=4>arr[2]=2 → dp2[1]=2;  maxi=dp1[1]+dp2[1]-1=2+2-1=3
     *     i=0: arr[0]=1 not > any to its right → dp2[0]=1;  maxi=max(3,1+1-1)=3
     *     dp2 = [1, 2, 1]
     *
     *   Peak candidates:
     *     i=0: 1+1-1=1
     *     i=1: 2+2-1=3  ← maximum (peak at 4)
     *     i=2: 2+1-1=2
     *
     *   Output: 3 ✓  (sequence: [1, 4, 2])
     *
     * TC = O(N²)  — two passes of the standard O(N²) LIS DP
     * SC = O(N)   — dp1[] and dp2[] arrays
     * -----------------------------------------------------------------------
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
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 673 - Medium): Number of Longest Increasing Subsequences
     * -----------------------------------------------------------------------
     * Given an integer array, return the NUMBER of distinct Longest Increasing
     * Subsequences (not the length, but how many such subsequences exist).
     *
     * Example:
     *   Input : arr = [1, 3, 5, 4, 7]
     *   Output: 2
     *
     *   Both [1,3,5,7] and [1,3,4,7] are valid LIS of length 4 → count = 2.
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (DP with parallel count[] array)
     * -----------------------------------------------------------------------
     * Two arrays (same size as input):
     *   dp[i]    = length of the LIS ending at index i.
     *   count[i] = number of distinct LIS of that length ending at index i.
     *
     * Both initialised to 1 (each element alone is a subsequence of length 1).
     *
     * For each i (left to right), scan all prev < i:
     *   CASE 1: arr[i] > arr[prev] AND dp[prev]+1 > dp[i]
     *     → Found a strictly LONGER LIS ending at i.
     *       dp[i]    = dp[prev]+1
     *       count[i] = count[prev]           ← inherit count from prev's chain
     *
     *   CASE 2: arr[i] > arr[prev] AND dp[prev]+1 == dp[i]
     *     → Found another chain of the SAME longest length ending at i.
     *       count[i] += count[prev]           ← accumulate additional chains
     *
     * After filling dp[] and count[], scan for maxi = max(dp[]):
     *   Sum count[i] for all i where dp[i] == maxi → that's the answer.
     *
     * Step-by-step (arr = [1, 3, 5, 4, 7]):
     *
     *   dp    = [1, 1, 1, 1, 1]
     *   count = [1, 1, 1, 1, 1]
     *
     *   i=1 (3): arr[0]=1<3 → dp[1]+1 > dp[1] → dp[1]=2, count[1]=count[0]=1
     *   i=2 (5): arr[0]<5 → dp[2]=2,count[2]=1; arr[1]<5 → dp[2]=3>2,count[2]=count[1]=1
     *   i=3 (4): arr[0]<4 → dp[3]=2,count[3]=1; arr[1]<4 → dp[3]=3>2,count[3]=1;
     *            arr[2]=5>4 (skip)
     *   i=4 (7): arr[0]<7 → dp[4]=2,count[4]=1; arr[1]<7 → dp[4]=3>2,count[4]=1;
     *            arr[2]<7 → dp[4]=4>3,count[4]=count[2]=1;
     *            arr[3]<7 → dp[3]+1=4==dp[4] → count[4]+=count[3]=1 → count[4]=2
     *
     *   dp    = [1, 2, 3, 3, 4]
     *   count = [1, 1, 1, 1, 2]
     *
     *   maxi = 4. Indices where dp[i]==4 → only i=4, count[4]=2.
     *   Output: 2  ✓  ([1,3,5,7] and [1,3,4,7])
     *
     * TC = O(N²)  — two nested loops
     * SC = O(N)   — dp[] and count[] arrays
     * -----------------------------------------------------------------------
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


