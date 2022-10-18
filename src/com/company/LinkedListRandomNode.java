package com.company;

import java.util.Random;

class LinkedListRandomNode {

    /**
     * @param head The linked list's head.
     * Note that the head is guaranteed to be not null, so it contains at least one node.
     */

    ListNode head;
    Random generator;

    public LinkedListRandomNode(ListNode head) {
        this.head = head;
        this.generator = new Random();
    }

    /**
     * Returns a random node's value.
     */
    public int getRandom() {
        int elem = -1, index = -1;
        ListNode temp = head;
        while (temp != null) {
            index++;
            if (index == 0) {
                elem = temp.val;
            } else {
                int random = generator.nextInt(index + 1) + 1;
                if (random == index) {
                    elem = temp.val;
                }
            }
            temp = temp.next;
        }
        return elem;
    }
}
