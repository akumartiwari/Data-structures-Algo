package com.company;

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
}