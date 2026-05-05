package com.company;

import java.util.Collections;

import static java.util.Arrays.asList;

public class ListNode {
    public int val;
    public ListNode next;

    public ListNode() {
    }

    public ListNode(int val) {
        this.val = val;
    }

    public ListNode(int val, ListNode next) {
        this.val = val;
        this.next = next;
    }

    /*
     * PROBLEM: Rotate List (LeetCode 61)
     * Rotate a linked list to the right by k places.
     *
     * ALGORITHM: Two Pointers (find list length, compute effective rotation, split and reconnect)
     * TC: O(n) | SC: O(1)
     */
    public ListNode rotateRight(ListNode head, int k) {
        if (head == null) return null;
        if (k == 0) return head;
        ListNode fast = head;
        int l = 1;
        while (fast.next != null) {
            fast = fast.next;
            l++;
        }
        int shifts = k % l;

        if (l == k || shifts == 0) return head;

        ListNode current = head;
        int curr = 1;
        while (curr < l - shifts) {
            current = current.next;
            curr++;
        }
        ListNode temp = current.next;
        current.next = null;
        fast.next = head;
        return temp;
    }

    /*
     * PROBLEM: Get Linked List Size (Helper)
     * Return the number of nodes in the linked list rooted at root.
     *
     * ALGORITHM: Linear scan
     * TC: O(n) | SC: O(1)
     */
    // O(n)
// [1,2,3,4]
//5
    /*
    Dry run:-
    l = 4
    r = 4;
    q = 1
  k = 5
     */

    public int size(ListNode root) {
        ListNode curr = root;
        int size = 0;
        while (curr != null) {
            curr = curr.next;
            size++;
        }
        return size;
    }

    /*
     * PROBLEM: Split Linked List in Parts (LeetCode 725)
     * Split a linked list into k consecutive parts as evenly as possible (optimised version).
     *
     * ALGORITHM: Two Pointers (compute size, distribute div+1 nodes to the first mod parts)
     * TC: O(n) | SC: O(k)
     */
    public ListNode[] splitListToPartsO(ListNode root, int k) {
        ListNode[] res = new ListNode[k];
        int size = size(root);  // First you find out the size of linked list.
        ListNode curr = root;
        ListNode prev = root;
        int i = 0;
        int div = size / k;  // Get how many part it can really divided.
        int mod = size % k; // Get how many additional one node need to be added from start.
        while (curr != null) {
            res[i++] = curr;
            int temp = div;
            while (temp-- != 0) {
                prev = curr;
                curr = curr.next;
            }
            if (mod != 0) {    // Take one extra node till mod > 0 i.e. ensures that one extra node is added from starting
                prev = curr;
                curr = curr.next;
                mod--;
            }

            prev.next = null;
        }
        return res;
    }


    /*
     * PROBLEM: Split Linked List in Parts (LeetCode 725)
     * Split a linked list into k consecutive parts (alternative implementation).
     *
     * ALGORITHM: Linear scan (compute length, then walk to split boundaries)
     * TC: O(n) | SC: O(k)
     */
    public ListNode[] splitListToParts(ListNode root, int k) {
        ListNode[] list = new ListNode[k];

        if (root == null) return list;
        if (k == 0) {
            ListNode[] nodes = new ListNode[1];
            nodes[0] = root;
            return nodes;
        }
        ListNode fast = root;
        ListNode head = root;
        System.out.println(head.val);

        int l = 1;
        while (fast.next != null) {
            fast = fast.next;
            l++;
        }

        int r = l % k;
        int q = l / k;
        if (k < l) {
            boolean isFirstPart = true;

            while (l / k-- > 0) {
                while (k-- != 0) {
                    ListNode node = new ListNode();
                    int curr = 0;

                    if (isFirstPart) {
                        while (curr < r + q) {
                            root = root.next;
                            curr++;
                        }
                        node = root.next;
                        isFirstPart = false;
                    } else {
                        while (curr < q) {
                            root = root.next;
                            curr++;
                        }
                        node = root.next;
                    }
                    list[l - l / k] = node;
                }
            }
        } else if (k == l) list[0] = root;
        else {
            int count = 1;
            while (count == l) {
                list[count - 1] = new ListNode(root.val, null);
                root = root.next;
                count++;
            }
        }
        return list;
    }


    /*
     * PROBLEM: Odd Even Linked List (LeetCode 328)
     * Group all odd-indexed nodes together followed by even-indexed nodes in-place.
     *
     * ALGORITHM: Two Pointers (interleave odd and even pointer chains, then join them)
     * TC: O(n) | SC: O(1)
     */
    public ListNode oddEvenList(ListNode head) {

        int counter = 1;
        ListNode odd = head;
        ListNode even = head;

        while (head.next != null) {
            if (counter % 2 == 0) {
                even.next = head.next;
                System.out.println("Even= " + even.val);
            } else {
                odd.next = head.next;
                System.out.println("Even= " + odd.val);
            }
            head = head.next;
            counter++;
        }

        even = odd.next;
        head = odd.next;
        return head;
    }


    /*
"abcd"
"cdef"
3
s = "abcd", t = "acde", maxCost = 0


------------------------------------------------
s = "abcd", t = "bcdf", maxCost = 3



// start = 0, end = 0;

"krrgw"
"zjxss"
19
     */

    /*
    find the increasing sequence from the last having x breaks the sequence
find next greater element than x from last i.e z
swap z with x
reverse subarray from x + 1 to end

[1,2,3]
[1, 3,2]
index = 2, elem = 3;
temp = 3;
nums[2] = nums[3-1-2]
nums[2] = nums[0]
nums[2] = 1;
nums[0]= 3;
3,2,1,

tempswap = 1;
nums[2]=2;
nums[1]=1;

     3,1,2
     [3,2,1]




         */


    /*
     * PROBLEM: Next Permutation (LeetCode 31)
     * Rearrange nums into the lexicographically next greater permutation in-place.
     *
     * ALGORITHM: Two Pointers (find rightmost descending break, swap with next greater element,
     *            then reverse the suffix)
     * TC: O(n) | SC: O(1)
     */
    public void nextPermutation(int[] nums) {
        int elem = Integer.MIN_VALUE;
        int index = 0;
        int n = nums.length;
        for (int i = n - 1; i > 0; i--) {
            index++;
            if (nums[i] >= elem) {
                elem = nums[i];
                continue;
            } else {
                break;
            }
        }

        int temp = elem;
        nums[index] = nums[n - index];
        nums[n - index] = temp;


        if (index >= n - 1) {
            Collections.reverse(asList(nums));
            System.out.println(asList(nums));
        } else {
            // One by one reverse first
            // and last elements of a[0..k-1]
            for (int i = index + 1; i < n; i++) {
                int tempswap = nums[i];
                nums[i] = nums[n - i];
                nums[n - i] = tempswap;
            }
        }
        for (int num : nums) {
            System.out.println(num);
        }
    }


    /*
     * PROBLEM: Get Equal Substrings Within Budget — Optimal (LeetCode 1208)
     * Find the maximum length substring you can make equal within maxCost budget (sliding window).
     *
     * ALGORITHM: Sliding Window (expand right, shrink left when cost exceeds budget)
     * TC: O(n) | SC: O(1)
     */
    public int equalSubstringOptimal(String s, String t, int maxCost) {
        int start = 0, end = 0, currcost = 0, maxlength = 0;
        int n = s.length();
        while (end < n) {
            currcost += Math.abs(s.charAt(end) - t.charAt(end));
            if (currcost > maxCost) {
                currcost -= Math.abs(s.charAt(start) - t.charAt(start));
                start++;
            }

            maxlength = Math.max(maxlength, end - start + 1);
            end++;
        }

        return maxlength;
    }


    /*
     * PROBLEM: Get Equal Substrings Within Budget (LeetCode 1208)
     * Find the maximum length substring you can make equal within maxCost (DP approach).
     *
     * ALGORITHM: DP (build cost array, then linear scan with running cost)
     * TC: O(n) | SC: O(n)
     */
    public int equalSubstring(String s, String t, int maxCost) {
        if (s.length() == 0 || t.length() == 0) return 0;
        if (s.length() != t.length()) return 0;
        if (maxCost == 0) return 1;

        int[] dp = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            int curr = Math.abs((int) (s.charAt(i) - t.charAt(i)));
            int cost;
            if (i == 0) {
                cost = curr;
            } else {
                cost = Math.min(dp[i - 1], curr);
            }
            dp[i] = cost;
            System.out.println(dp[i]);
        }


        int index = 0;
        int maxLength = 0;
        int oriCost = 0;
        while (maxLength < maxCost && oriCost + dp[s.length() - 1 - index] <= maxCost) {
            maxLength += 1;
            oriCost += dp[s.length() - 1 - index];
            System.out.println(oriCost);
            index++;
        }
        return dp[s.length() - 1] > maxCost ? 1 : maxLength;
    }

}
