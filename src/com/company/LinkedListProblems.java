package com.company;

import java.util.*;

public class LinkedListProblems {
    //Author: Anand
    /*
    ## Intersection of tow linked list##
    The idea is to keep on traversing lists until intersection point is found
    Proof : We are bound to get intersection point beacause every time greater list ends it will swich
    to smaller list and vice-versa . As a result of which the difference of their lengths will keep
    on getting reduced and finally become zero. This is the intersection point.
    */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        ListNode tmp1 = headA;
        ListNode tmp2 = headB;

        while (tmp1 != tmp2) {

            if (tmp1 == null) tmp1 = headB;
            else tmp1 = tmp1.next;

            if (tmp2 == null) tmp2 = headA;
            else tmp2 = tmp2.next;

        }

        return tmp1;
    }

    // ## Reverse LinkedList ##

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode() {}
     * ListNode(int val) { this.val = val; }
     * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public ListNode reverseList(ListNode head) {
        if (head == null) return head;
        ListNode curr = head, prev = null, next;
        while (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        head = prev;
        return head;
    }

    // ## Reverse LinkedList in K Group ##

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode() {}
     * ListNode(int val) { this.val = val; }
     * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */
    public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return head;
        int len = getLength(head);
        return reverse(head, k, len);
    }

    private int getLength(ListNode curr) {
        int cnt = 0;
        while (curr != null) {
            cnt++;
            curr = curr.next;
        }
        return cnt;
    }

    // Recursive function to reverse in groups
    private ListNode reverse(ListNode head, int k, int len) {
        // base case
        if (len < k) return head;

        ListNode curr = head, prev = null, next = null;
        for (int i = 0; i < k; i++) {
            // reverse linked list code
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }

        ListNode nextNode = reverse(curr, k, len - k);
        head.next = nextNode;
        return prev;
    }

    public ListNode removeNodes(ListNode head) {

        TreeMap<Integer, Integer> tm = new TreeMap<>();

        List<Integer> nodes = new ArrayList<>();
        ListNode temp = head;
        while (temp != null) {
            nodes.add(temp.val);
            temp = temp.next;
        }

        Set<Integer> indexDeleted = new HashSet<>();

        for (int i = nodes.size() - 1; i >= 0; i--) {
            tm.put(nodes.get(i), tm.getOrDefault(nodes.get(i), 0) + 1);

            if (tm.higherKey(nodes.get(i)) != null && tm.higherKey(nodes.get(i)) > nodes.get(i)) {
                indexDeleted.add(i);
            }
        }


        ListNode ptr = new ListNode(-1);
        ptr.next = head;

        ListNode cur = head;
        ListNode prev = ptr;

        int curr = 0;

        while (cur != null) {
            if (indexDeleted.contains(curr++)) {
                prev.next = cur.next;
            } else {
                prev = cur;
            }
            cur = cur.next;
        }
        return ptr.next;

    }

    // Insert a node in linked list
    public ListNode insertGreatestCommonDivisors(ListNode head) {
        ListNode ptr = head;

        while (ptr != null && ptr.next != null) {
            ListNode nn = new ListNode(gcd(ptr.val, ptr.next.val));
            /* 4. Make next of new Node as next of prev_node */
            nn.next = ptr.next;

            /* 5. make next of prev_node as new_node */
            ptr.next = nn;
            ptr = ptr.next.next;
        }

        return head;
    }

    public int gcd(int a, int b) {
        if (a == 0)
            return b;

        return gcd(b % a, a);
    }

    /**
     * Definition for singly-linked list.
     * public class ListNode {
     * int val;
     * ListNode next;
     * ListNode() {}
     * ListNode(int val) { this.val = val; }
     * ListNode(int val, ListNode next) { this.val = val; this.next = next; }
     * }
     */

    private ListNode append(ListNode head_ref, int new_data) {
        /* 1. allocate node */
        ListNode new_node = new ListNode();

        ListNode last = head_ref; /* used in step 5*/

        /* 2. put in the data */
        new_node.val = new_data;

    /* 3. This new node is going to be
    the last node, so make next of
    it as null*/
        new_node.next = null;

    /* 4. If the Linked List is empty,
    then make the new node as head */
        if (head_ref == null) {
            head_ref = new_node;
            return head_ref;
        }

        /* 5. Else traverse till the last node */
        while (last.next != null) {
            last = last.next;
        }

        /* 6. Change the next of last node */
        last.next = new_node;
        return head_ref;
    }


    public ListNode doubleIt(ListNode head) {

        StringBuilder sb = new StringBuilder();

        while (head != null) {
            sb.append(head.val);
            head = head.next;
        }


        int carry = 0;
        StringBuilder nn = new StringBuilder();
        for (int i = sb.length() - 1; i >= 0; i--) {
            int c = sb.charAt(i) - '0';
            int d = (c * 2) + carry;
            int[] digits = new int[2];
            int ind = 0;
            while (d > 0) {
                digits[ind++] = d % 10;
                d /= 10;
            }

            nn.append(digits[0]);
            carry = digits[1];
        }


        if (carry != 0) nn.append(carry);
        List<Integer> nll = new ArrayList<>();

        if (nn.toString() == "0") {
            nll.add(0);
        } else {
            for (int i = 0; i < nn.length(); i++) {
                int c = nn.charAt(i) - '0';
                nll.add(c);
            }
        }

        Collections.reverse(nll);
        ListNode temp = new ListNode(nll.get(0));
        for (int i = 1; i < nll.size(); i++) {
            append(temp, nll.get(i));
        }

        return temp;
    }


    public int minAbsoluteDifference(List<Integer> nums, int x) {

        TreeMap<Integer, Integer> freq = new TreeMap<>();
        for (int i = x; i < nums.size(); i++) freq.put(nums.get(i), freq.getOrDefault(nums.get(i), 0) + 1);

        int md = Integer.MAX_VALUE;
        for (int i = 0; i < nums.size() - x; i++) {
            int e = nums.get(i);
            if (freq.floorKey(e) != null) md = Math.min(md, Math.abs(e - freq.floorKey(e)));
            if (freq.ceilingKey(e) != null) md = Math.min(md, Math.abs(e - freq.ceilingKey(e)));

            freq.put(nums.get(x + i), freq.getOrDefault(nums.get(x + i), 0) - 1);
            if (freq.get(nums.get(x + i)) <= 0) freq.remove(nums.get(x + i));
        }

        return md;
    }



}


