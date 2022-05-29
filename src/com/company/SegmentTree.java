package com.company;

import java.util.*;

// Implementation of segment tree
public class SegmentTree {
    int st[];// array to store segment tree nodes

    SegmentTree(int[] arr, int n) {
        // allocate memory for segment tree
        int ht = (int) Math.ceil(Math.log(n) / Math.log(2));
        // maximum size of seg tree
        int max_size = 2 * (int) Math.pow(2, n) - 1;

        st = new int[max_size];
        constructorUtil(arr, 0, n - 1, 0);
    }

    private int getSum(int n, int qs, int qe) {
        // Check for erroneous input values
        if (qs < 0 || qe > n - 1 || qs > qe) {
            System.out.println("Invalid Input");
            return -1;
        }
        return getSumUtil(0, n - 1, qs, qe, 0);
    }

    private int getSumUtil(int ss, int se, int qs, int qe, int si) {

        // If segment of this node is a part of given range, then return
        // the sum of the segment
        if (qs <= ss && qe >= se)
            return st[si];

        // If segment of this node is outside the given range
        if (se < qs || ss > qe)
            return 0;

        // If a part of this segment overlaps with the given range
        int mid = getMid(ss, se);
        return getSumUtil(ss, mid, qs, qe, 2 * si + 1) +
                getSumUtil(mid + 1, se, qs, qe, 2 * si + 2);

    }

    private int constructorUtil(int[] arr, int l, int r, int index) {
        if (l == r) {
            st[l] = arr[r];
            return arr[r];
        }

        int mid = getMid(l, r);
        st[index] = constructorUtil(arr, l, mid, 2 * index + 1) + // left subtree
                constructorUtil(arr, mid + 1, r, 2 * index + 2); // right subtree

        return st[index];
    }

    private int getMid(int l, int r) {
        return l + (r - l) / 2;
    }


    // The function to update a value in input array and segment tree.
    // It uses updateValueUtil() to update the value in segment tree
    void updateValue(int arr[], int n, int i, int new_val) {
        // Check for erroneous input index
        if (i < 0 || i > n - 1) {
            System.out.println("Invalid Input");
            return;
        }

        // Get the difference between new value and old value
        int diff = new_val - arr[i];

        // Update the value in array
        arr[i] = new_val;

        // Update the values of nodes in segment tree
        updateValueUtil(0, n - 1, i, diff, 0);
    }

    private void updateValueUtil(int ss, int se, int i, int diff, int si) {
        // Base Case: If the input index lies outside the range of
        // this segment
        if (i < ss || i > se)
            return;

        // If the input index is in range of this node, then update the
        // value of the node and its children
        st[si] = st[si] + diff;
        if (se != ss) {
            int mid = getMid(ss, se);
            updateValueUtil(ss, mid, i, diff, 2 * si + 1);
            updateValueUtil(mid + 1, se, i, diff, 2 * si + 2);
        }
    }


    /*
        Input
        ["BookMyShow", "gather", "gather", "scatter", "scatter"]
        [[2, 5], [4, 0], [2, 0], [5, 1], [5, 1]]
        Output
        [null, [0, 0], [], true, false]

        Explanation
        BookMyShow bms = new BookMyShow(2, 5); // There are 2 rows with 5 seats each
        bms.gather(4, 0); // return [0, 0]
                          // The group books seats [0, 3] of row 0.
        bms.gather(2, 0); // return []
                          // There is only 1 seat left in row 0,
                          // so it is not possible to book 2 consecutive seats.
        bms.scatter(5, 1); // return True
                           // The group books seat 4 of row 0 and seats [0, 3] of row 1.
        bms.scatter(5, 1); // return False
                           // There are only 2 seats left in the hall.
     */
    class BookMyShow {

        int n, m;
        List<List<int[]>> rowsOccupied;

        public BookMyShow(int n, int m) {
            this.n = n;
            this.m = m;
            rowsOccupied = new ArrayList<>();
            for (int k = 0; k < m; k++) {
                rowsOccupied.add(new ArrayList<>());
            }
        }

        public int[] gather(int k, int maxRow) {
            // Find out if such slot is available
            for (List<int[]> row : rowsOccupied) {
                for (int[] l : row) {


                }
            }
            return new int[]{};
        }

        public boolean scatter(int k, int maxRow) {
            return false;
        }
    }

    /**
     * Your BookMyShow object will be instantiated and called as such:
     * BookMyShow obj = new BookMyShow(n, m);
     * int[] param_1 = obj.gather(k,maxRow);
     * boolean param_2 = obj.scatter(k,maxRow);
     */

    public static class OperatorCombinationsResults {

        //Wrapper, Time O(Cn^2), Space O(Cn^2), Cn is Catalan number, n is array length
        public static int operatorCombinations(List<Integer> tokens) {
            List<Integer> res = helper(tokens, new HashMap<>());
            Collections.sort(res, Collections.reverseOrder());
            return res.get(0);

        }

        //DFS + memoziation, Time average O(Cn^2) when memoization takes places, Space O(Cn^2)
        private static List<Integer> helper(List<Integer> t, Map<List<Integer>, List<Integer>> map) {
            if (t.size() <= 1)
                return t;
            if (map.containsKey(t))
                return map.get(t);
            List<Integer> res = new ArrayList<>();
            for (int i = 1; i < t.size(); i++) {
                List<Integer> left = helper(t.subList(0, i), map);
                List<Integer> right = helper(t.subList(i, t.size()), map);
                for (int a : left) {
                    for (int b : right) {
                        res.add(a + b);
                        res.add(a - b);
                        res.add(a * b);
                        if (b != 0)
                            res.add(a / b);
                    }
                }
            }
            map.put(t, res); //map stores (token, partial result) pair
            return res;
        }

//        public static void main(String[] args) {
//            List<Integer> list = Arrays.asList(3,4,5,1);
//            System.out.println("Input: " + list);
//            System.out.print("Max possible value is: ");
//            System.out.println(operatorCombinations(list));
//        }
    }


    /*

    Generate random max index
    Given an array of integers, randomly return an index of the maximum value seen by far.
            e.g.
            Given [11,30,2,30,30,30,6,2,62, 62]

    Having iterated up to the at element index 5 (where the last 30 is), randomly give an index among [1, 3, 4, 5] which are indices of 30 - the max value by far. Each index should have a ¼ chance to get picked.

    Having iterated through the entire array, randomly give an index between 8 and 9 which are indices of the max value 62.
     */

    public void sampleIdx(int[] array) {
        if (array == null || array.length == 0) {
            return;
        }
        Random rnd = new Random();
        int res = 0, max = Integer.MIN_VALUE, count = 0;
        for (int i = 0; i < array.length; i++) {
            if (max < array[i]) {
                max = array[i];
                res = i;
                count = 1;
            } else if (max == array[i]) {
                count++;
                int idx = rnd.nextInt(count); //(0, k - 1)
                if (idx == 0) {

                    res = i;
                    System.out.print("A max value index up to the " + i + "th element is " + res);

                }
            }
        }
    }

    /*

    public ListNode convertToList(SolutionMax.TreeNode node) {
        if (node == null)
            return null;
        ListNode head = new ListNode(-1);
        ListNode tail = treeToList(node, head);
        head = head.next;
        head.prev = tail;
        tail.next = head;

        return head;
    }

    private static ListNode treeToList(SolutionMax.TreeNode n, ListNode tail) {
        if (n == null)
            return tail;

        tail = treeToList(n.left, tail);
        tail.next = new ListNode(n.data);
        tail.next.prev = tail;
        tail = tail.next;
        tail = treeToList(n.right, tail);

        return tail;
    }
     */
}
