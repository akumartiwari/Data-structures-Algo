package com.company;

import java.util.HashMap;
import java.util.*;

public class StackExamples {

    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 739 - Medium): Daily Temperatures
     * -----------------------------------------------------------------------
     * Given an array temperatures[], return an array answer[] where answer[i]
     * is the number of days you must wait after day i to get a warmer day.
     * If no such future day exists, answer[i] = 0.
     *
     * Example:
     *   Input : temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
     *   Output:                 [ 1,  1,  4,  2,  1,  1,  0,  0]
     *
     *   - Day 0 (73°): Day 1 (74°) is warmer → wait 1 day
     *   - Day 2 (75°): Day 6 (76°) is the next warmer → wait 4 days
     *   - Day 6 (76°): No warmer day ahead → 0
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Monotonic Stack — left to right)
     * -----------------------------------------------------------------------
     * Maintain a stack of INDICES whose "next warmer day" hasn't been found yet.
     * The stack is always in DECREASING order of temperatures (monotonic).
     *
     * For each day i:
     *   1. While stack is non-empty AND temperature at stack-top < temperatures[i]:
     *        - Pop index idx → day i is the next warmer day for idx
     *        - Record answer[idx] = i - idx  (number of days to wait)
     *   2. Push i onto the stack.
     *
     * Any indices left in the stack at the end have no warmer future day → remain 0.
     *
     * Step-by-step (temperatures = [73, 74, 75, 71, 69, 72, 76, 73]):
     *
     *   i=0 (73): stack empty → push 0.               stk=[0]
     *   i=1 (74): 74 > 73 → pop 0, ans[0]=1-0=1.      stk=[1]
     *   i=2 (75): 75 > 74 → pop 1, ans[1]=2-1=1.      stk=[2]
     *   i=3 (71): 71 < 75 → push 3.                   stk=[2,3]
     *   i=4 (69): 69 < 71 → push 4.                   stk=[2,3,4]
     *   i=5 (72): 72 > 69 → pop 4, ans[4]=5-4=1
     *             72 > 71 → pop 3, ans[3]=5-3=2
     *             72 < 75 → push 5.                   stk=[2,5]
     *   i=6 (76): 76 > 72 → pop 5, ans[5]=6-5=1
     *             76 > 75 → pop 2, ans[2]=6-2=4.      stk=[6]
     *   i=7 (73): 73 < 76 → push 7.                   stk=[6,7]
     *   End: idx 6 and 7 remain → ans[6]=ans[7]=0.
     *
     *   Final answer: [1, 1, 4, 2, 1, 1, 0, 0] ✓
     *
     * TC = O(N)  — each index is pushed and popped at most once
     * SC = O(N)  — stack holds at most N indices
     * -----------------------------------------------------------------------
     */
    public int[] dailyTemperatures(int[] temperatures) {
        // base cases
           /*
            int n = temperatures.length;

            int[] ans = new int[n];
            // iterate throughout all element
            for (int i = 0; i < n; i++) {
                int count = 0;
                for (int j = i + 1; j < n; j++) {
                    count++;
                    if (temperatures[j] > temperatures[i]) ans[i] = count;
                }
            }

            return ans;
        }
        */


        int n = temperatures.length;
        int[] nextWarmerday = new int[n];
        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack

        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday[idx] = idx - i;
            }
            stk.push(i);
        }
        return nextWarmerday;
    }

    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 496 - Easy): Next Greater Element I
     * -----------------------------------------------------------------------
     * nums1 is a subset of nums2. For each element in nums1, find the first
     * element to its RIGHT in nums2 that is strictly greater. Return -1 if none.
     *
     * Example:
     *   nums1 = [4, 1, 2],  nums2 = [1, 3, 4, 2]
     *   Output: [-1, 3, -1]
     *
     *   - 4 in nums2: no element to its right is greater → -1
     *   - 1 in nums2: next greater to the right is 3    →  3
     *   - 2 in nums2: no element to its right is greater → -1
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Monotonic Stack + HashMap — right to left on nums2)
     * -----------------------------------------------------------------------
     * Pre-process nums2 once using a monotonic stack to build a map:
     *   value → its next greater element in nums2.
     * Then answer each nums1 query in O(1) via the map.
     *
     * Step-by-step on nums2 = [1, 3, 4, 2] (right to left):
     *
     *   i=3 (val=2): stack empty → map[2]=-1, push 3.       stk=[3]
     *   i=2 (val=4): 4 >= nums2[3]=2 → pop 3.
     *                stack empty → map[4]=-1, push 2.       stk=[2]
     *   i=1 (val=3): 3 < nums2[2]=4 → map[3]=4,  push 1.   stk=[2,1]
     *   i=0 (val=1): 1 < nums2[1]=3 → map[1]=3,  push 0.   stk=[2,1,0]
     *
     *   map: {2:-1, 4:-1, 3:4, 1:3}
     *
     * Answer nums1 = [4, 1, 2]:
     *   4 → map[4] = -1
     *   1 → map[1] =  3
     *   2 → map[2] = -1
     *
     *   Output: [-1, 3, -1] ✓
     *
     * TC = O(N + M)  — N = nums2.length, M = nums1.length
     * SC = O(N)      — map and stack both hold at most N entries
     * -----------------------------------------------------------------------
     */
    //Author: Anand
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {

        Map<Integer, Integer> map = new HashMap<>();
        Stack<Integer> stk = new Stack<>();

        for (int i = nums2.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums2[stk.peek()] <= nums2[i]) stk.pop();

            if (!stk.isEmpty()) map.put(nums2[i], nums2[stk.peek()]);
            else map.put(nums2[i], -1);

            stk.push(i);
        }


        int[] ans = new int[nums1.length];
        for (int i = 0; i < nums1.length; i++) ans[i] = map.getOrDefault(nums1[i], -1);

        return ans;
    }



    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 503 - Medium): Next Greater Element II — Alternate
     * (HashMap approach to handle DUPLICATE values in a circular array)
     * -----------------------------------------------------------------------
     * Same problem as nextGreaterElementsOptimised above, but this version
     * explicitly handles arrays that may contain DUPLICATE values by mapping
     * each value to a LIST of its possible next-greater answers rather than
     * relying purely on indices.
     *
     * Example:
     *   Input : nums = [1, 2, 1]
     *   Output: [2, -1, 2]
     *   (Both 1s each have 2 as their next greater; 2 has no greater value.)
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Two-pass Monotonic Stack + HashMap<value, List<nextGreater>>)
     * -----------------------------------------------------------------------
     * PASS 1 (right to left, single traversal — handles non-circular part):
     *   Build map: value → list of next-greater values seen so far.
     *   Drain the list into ans[] left to right. Elements with no answer yet
     *   get a sentinel (Integer.MIN_VALUE) to mark them for Pass 2.
     *
     * PASS 2 (right to left again — handles circular wrap-around):
     *   Using a "visited" map (cm), add only the FIRST next-greater found
     *   per distinct value. Fill remaining MIN_VALUE slots in ans[] from this.
     *
     * Why two passes?
     *   The first pass gives the linear (non-wrap) next-greater.
     *   The second pass covers elements that need to wrap around the array.
     *
     * Step-by-step (nums = [1, 2, 1]):
     *
     *   Pass 1 (right to left):
     *     i=2 (val=1): stk empty → map[1]=[]      push 2.  stk=[2]
     *     i=1 (val=2): 2>=1 pop 2; stk empty → map[2]=[]  push 1. stk=[1]
     *     i=0 (val=1): 1<2  → map[1]=[2],         push 0.  stk=[1,0]
     *
     *   Fill ans[] left to right:
     *     ans[0]=map[1].removeLast()=2, ans[1]=MIN_VALUE(map[2] empty), ans[2]=MIN_VALUE(map[1] now empty)
     *
     *   Pass 2 (right to left, circular):
     *     Adds first next-greater per value using cm guard.
     *     → map[2] stays [], map[1] gets [2] (wrap-around find)
     *
     *   Final fill: ans[1]→map[2] empty→-1, ans[2]→map[1]=[2]→2
     *
     *   Output: [2, -1, 2] ✓
     *
     * TC = O(N)  — two linear passes
     * SC = O(N)  — map, stack, and cm together hold O(N) entries
     * -----------------------------------------------------------------------
     */
    //Author: Anand
    public int[] nextGreaterElements(int[] nums) {

        Map<Integer, List<Integer>> map = new HashMap<>();

        java.util.Stack<Integer> stk = new java.util.Stack<>();

        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[stk.peek()] <= nums[i]) stk.pop();

            if (!map.containsKey(nums[i])) map.put(nums[i], new ArrayList<>());

            if (!stk.isEmpty()) map.get(nums[i]).add(nums[stk.peek()]);

            stk.push(i);
        }

        int[] ans = new int[nums.length];
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(nums[i]) && map.get(nums[i]).size() > 0) {
                ans[i] = map.get(nums[i]).get(map.get(nums[i]).size() - 1);
                map.get(nums[i]).remove(map.get(nums[i]).size() - 1);
            } else ans[i] = Integer.MIN_VALUE;
        }


        Map<Integer, Boolean> cm = new HashMap<>();

        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[stk.peek()] <= nums[i]) stk.pop();

            if (!map.containsKey(nums[i])) map.put(nums[i], new ArrayList<>());

            if (!stk.isEmpty() && !cm.containsKey(nums[i])) {
                map.get(nums[i]).add(nums[stk.peek()]);
                cm.put(nums[i], true);
            }

            stk.push(i);
        }

        for (int i = 0; i < nums.length; i++) {
            if (ans[i] != Integer.MIN_VALUE) continue;

            if (map.containsKey(nums[i]) && map.get(nums[i]).size() > 0)
                ans[i] = map.get(nums[i]).get(map.get(nums[i]).size() - 1);
            else ans[i] = -1;
        }

        return ans;
    }


    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 503 - Medium): Next Greater Element II (Circular Array)
     * -----------------------------------------------------------------------
     * Given a CIRCULAR integer array nums, return the next greater number for
     * every element. "Next greater" means the first element to the right
     * (wrapping around) that is strictly greater. Return -1 if none exists.
     *
     * Example:
     *   Input : nums = [1, 2, 1]
     *   Output: [2, -1, 2]
     *
     *   - nums[0]=1 → next greater (going right, wraps) is 2    → ans[0] = 2
     *   - nums[1]=2 → no element greater than 2 in whole array   → ans[1] = -1
     *   - nums[2]=1 → wraps around, finds 2 at index 0           → ans[2] = 2
     *
     * -----------------------------------------------------------------------
     * ALGORITHM  (Monotonic Stack — traverse the array TWICE)
     * -----------------------------------------------------------------------
     * Key insight: To handle the circular nature without duplicating the array,
     * iterate i from 2*n-1 down to 0 and use (i % n) to map to a real index.
     * This simulates one full loop followed by another, so every element gets
     * a chance to "see" elements that come before it in the original array.
     *
     * We maintain a MONOTONIC DECREASING stack of INDICES.
     *
     * Step-by-step (nums = [1, 2, 1], n = 3):
     *
     *   i=5 (idx=2): stack empty  → ans[2]=-1,  push 2.   stk=[2]
     *   i=4 (idx=1): nums[1]=2 >= nums[2]=1 → pop 2
     *                stack empty  → ans[1]=-1,  push 1.   stk=[1]
     *   i=3 (idx=0): nums[0]=1 <  nums[1]=2 → ans[0]=2,  push 0.   stk=[1,0]
     *   i=2 (idx=2): nums[2]=1 >= nums[0]=1 → pop 0
     *                nums[2]=1 <  nums[1]=2 → ans[2]=2,  push 2.   stk=[1,2]
     *   i=1 (idx=1): nums[1]=2 >= nums[2]=1 → pop 2
     *                nums[1]=2 >= nums[1]=2 → pop 1
     *                stack empty  → ans[1]=-1, push 1.   stk=[1]
     *   i=0 (idx=0): nums[0]=1 <  nums[1]=2 → ans[0]=2,  push 0.   stk=[1,0]
     *
     *   Final answer: [2, -1, 2] ✓
     *
     * Why pop? Any stack element ≤ current can never be the "next greater"
     * for current or anything further left → safely discard it.
     *
     * TC = O(N)  — each index is pushed/popped at most twice over 2N iterations
     * SC = O(N)  — stack holds at most N indices at any time
     * -----------------------------------------------------------------------
     */
    public int[] nextGreaterElementsOptimised(int[] nums) {
        int n = nums.length;
        int[] ans = new int[n];
        Stack<Integer> stk = new Stack<>(); // to store next greater elements in stack
        for (int i = 2 * n - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[i % n] >= nums[stk.peek()]) stk.pop();

            ans[i % n] = stk.isEmpty() ? -1 : nums[stk.peek()];
            stk.push(i % n);
        }
        return ans;
    }


    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 2454 - Hard): Find the Second Greater Element
     * -----------------------------------------------------------------------
     * Given a 0-indexed integer array nums, return an array answer[] where
     * answer[i] is the SECOND distinct greater integer to the RIGHT of nums[i].
     * If it doesn't exist, answer[i] = -1.
     *
     * "Second greater" means: skip the first element greater than nums[i],
     * then find the next element greater than nums[i] after that.
     *
     * Example:
     *   Input : nums = [2, 4, 0, 9, 6]
     *   Output:        [9, 6, 6, -1, -1]
     *
     *   - nums[0]=2: first greater to right is 4(idx=1), second greater is 9(idx=3) → 9
     *   - nums[1]=4: first greater to right is 9(idx=3), second greater is 6(idx=4) → 6
     *   - nums[2]=0: first greater to right is 9(idx=3), second greater is 6(idx=4) → 6
     *   - nums[3]=9: no element greater than 9 to the right                          → -1
     *   - nums[4]=6: no element greater than 6 to the right                          → -1
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Monotonic Stack + Jump Pointer)
     * -----------------------------------------------------------------------
     * Step 1 — Build a "next greater index" map using a monotonic stack (right to left):
     *   map[i] = index of the FIRST element to the right of i that is > nums[i].
     *   map[i] = -1 if no such element exists.
     *
     * Step 2 — For each i, find the SECOND greater element:
     *   Let fgi = map[i]        (index of first greater element)
     *   Start sgi = fgi + 1    (candidate for second greater, just after first)
     *
     *   Key optimisation (jump pointer):
     *   If nums[sgi] <= nums[i], then sgi itself is not ≥ nums[i], and since the
     *   subarray [fgi+1 .. sgi-1] is also ≤ nums[i] (they were popped before fgi),
     *   we can JUMP directly to map[sgi] (next greater of sgi) and skip the interval.
     *   Repeat until nums[sgi] > nums[i] or sgi goes out of bounds.
     *
     * Step-by-step (nums = [2, 4, 0, 9, 6]):
     *
     *   Build map (right to left):
     *     i=4: stk empty → map[4]=-1, push 4.  stk=[4]
     *     i=3: 9>6 → pop 4; stk empty → map[3]=-1, push 3.  stk=[3]
     *     i=2: 0<9 → map[2]=3, push 2.          stk=[3,2]
     *     i=1: 4>0 → pop 2; 4<9 → map[1]=3, push 1.  stk=[3,1]
     *     i=0: 2<4 → map[0]=1, push 0.          stk=[3,1,0]
     *
     *   map: {0→1, 1→3, 2→3, 3→-1, 4→-1}
     *
     *   Find second greater:
     *     i=0: fgi=1 (nums[1]=4>2), sgi=2. nums[2]=0<=2 → jump: sgi=map[2]=3.
     *          nums[3]=9>2 → ans[0]=9
     *     i=1: fgi=3 (nums[3]=9>4), sgi=4. nums[4]=6>4 → ans[1]=6
     *     i=2: fgi=3 (nums[3]=9>0), sgi=4. nums[4]=6>0 → ans[2]=6
     *     i=3: map[3]=-1 → ans[3]=-1
     *     i=4: map[4]=-1 → ans[4]=-1
     *
     *   Output: [9, 6, 6, -1, -1] ✓
     *
     * TC = O(N)  ��� each index visited amortised O(1) times via jump pointers
     * SC = O(N)  — stack and map hold at most N entries
     * -----------------------------------------------------------------------
     */
    public int[] secondGreaterElement(int[] nums) {

        Map<Integer, Integer> map = new HashMap<>(); // ind , NG ind
        Stack<Integer> stk = new Stack<>();

        for (int i = nums.length - 1; i >= 0; i--) {
            while (!stk.isEmpty() && nums[stk.peek()] <= nums[i]) stk.pop();

            if (!stk.isEmpty()) map.put(i, stk.peek());
            else map.put(i, -1);

            stk.push(i);
        }

        int[] ans = new int[nums.length];
        Arrays.fill(ans, -1);

        for (int i = 0; i < nums.length - 2; i++) {
            if (map.get(i) == -1) continue;

            int fgi = map.get(i);

            int sgi = fgi + 1;

            // For eg. if nums[sgi] is <= current element then all the remaining smaller (smaller than sgi) elements will be smaller than current.
            // Hence, we can directly jump to next greater of sgi ie, map.get(sgi)

            while (sgi != -1 && sgi < nums.length && nums[sgi] <= nums[i]) sgi = map.get(sgi);

            if (sgi < nums.length && sgi != -1) ans[i] = nums[sgi];

        }
        return ans;
    }


    /*
    Input: s = "zza"
    Output: "azz"
    Explanation: Let p denote the written string.
    Initially p="", s="zza", t="".
    Perform first operation three times p="", s="", t="zza".
    Perform second operation three times p="azz", s="", t="".
    */
    /*
    The first loop is to record the frequence of every character, the array freq,
    the second loop is to add every character into a stack,
    when adding the character into the stack, decreate the frequency of the character by one in the array freq,
    then the array freq is the frequency of every character in the rest of string.
    When adding one character from the top of the stack to the result, we check if there is one smaller character in the rest of the string,
    if there is, keep pushing the character of the rest of the string into the stack, if there is not,
    then add the top character into the result.
     */
    public String robotWithString(String s) {
        int[] freq = new int[26];
        for (char c : s.toCharArray()) freq[c - 'a']++;

        StringBuilder sb = new StringBuilder();
        Stack<Character> stack = new Stack<>();

        for (char c : s.toCharArray()) {
            stack.add(c);
            freq[c - 'a']--;

            while (!stack.isEmpty()) {
                char curr = stack.peek();
                if (hasSmaller(curr, freq)) break;
                sb.append(stack.pop());
            }
        }
        return sb.toString();
    }

    private boolean hasSmaller(char c, int[] freq) {
        for (int i = 0; i < (c - 'a'); ++i) if (freq[i] > 0) return true;
        return false;
    }


    /*
     * -----------------------------------------------------------------------
     * PROBLEM (LeetCode 739 Variant): Count Colder Days (Daily Temperatures)
     * -----------------------------------------------------------------------
     * Variant of Daily Temperatures: instead of returning the number of days
     * to wait per index, collect ALL wait-day counts into a list (in the order
     * each warmer day was discovered).
     *
     * Example:
     *   Input : temperatures = [73, 74, 75, 71, 72]
     *   Output list:            [1, 1, 2, 1]
     *
     *   - Day 0 (73) → Day 1 (74): wait = 1
     *   - Day 1 (74) → Day 2 (75): wait = 1
     *   - Day 3 (71) → Day 4 (72): wait = 1   (discovered at i=4, gap = 4-3=1)
     *   - Day 2 (75) → no warmer day discovered in this snippet (would remain)
     *
     * -----------------------------------------------------------------------
     * ALGORITHM (Monotonic Stack — same as Daily Temperatures, collect to list)
     * -----------------------------------------------------------------------
     * Maintain a monotonic stack of INDICES with unsatisfied "warmer day" queries.
     *
     * For each day i (left → right):
     *   1. While stack non-empty AND temperatures[stack.top] < temperatures[i]:
     *        - Pop idx → current day i is warmer than day idx
     *        - Add (i - idx) to the result list
     *   2. Push i onto the stack.
     *
     * Indices remaining in the stack at the end have no warmer future day
     * (they simply produce no entry in the output list).
     *
     * Step-by-step (temperatures = [73, 74, 75, 71, 72]):
     *
     *   i=0 (73): stack empty → push 0.                     stk=[0]
     *   i=1 (74): 74>73 → pop 0, list.add(1-0=1).  push 1. stk=[1]
     *   i=2 (75): 75>74 → pop 1, list.add(2-1=1).  push 2. stk=[2]
     *   i=3 (71): 71<75 → push 3.                           stk=[2,3]
     *   i=4 (72): 72>71 → pop 3, list.add(4-3=1).
     *             72<75 → push 4.                           stk=[2,4]
     *   End: idx 2 and 4 remain → no warmer day, not added to list.
     *
     *   Output list: [1, 1, 1] ✓
     *
     * TC = O(N)  — each index pushed and popped at most once
     * SC = O(N)  — stack and output list each hold at most N entries
     * -----------------------------------------------------------------------
     */
    private static List<Integer> countcolderDays1(int[] temperatures) {
        int n = temperatures.length;
        List<Integer> nextWarmerday = new ArrayList<>();

        Stack<Integer> stk = new Stack<>();// to store index of next warmer day in stack
        for (int i = 0; i < n; i++) {
            while (!stk.isEmpty() && temperatures[stk.peek()] < temperatures[i]) {
                int idx = stk.pop();
                nextWarmerday.add(i - idx);
            }
            stk.push(i);
        }
        return nextWarmerday;
    }

}

