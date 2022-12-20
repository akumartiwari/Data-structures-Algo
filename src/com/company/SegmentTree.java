package com.company;

import javafx.util.Pair;

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
        st[si] += diff;
        if (se != ss) {
            int mid = getMid(ss, se);
            updateValueUtil(ss, mid, i, diff, 2 * si + 1);
            updateValueUtil(mid + 1, se, i, diff, 2 * si + 2);
        }
    }

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

        class SegTree {
            int n, m; // n = no of rows, m = no of cols
            long[] segTree; // number of available seats in a range
            long[] sum; // sunm array to store sum of available seats in a range

            SegTree(int n, int m) {
                this.n = n;
                this.m = m;
                segTree = new long[4 * n];
                sum = new long[4 * n];

                buildSegTree(0, n - 1, 0, m);

            }

            private void buildSegTree(int lo, int hi, int ind, int maxAvalableSeats) {
                // base case
                if (lo == hi) {
                    segTree[ind] = maxAvalableSeats; // initialise with max seats
                    sum[ind] = maxAvalableSeats; // initialise with max sum of seats
                    return;
                }

                int mid = (lo + hi) / 2;

                buildSegTree(lo, mid, 2 * ind + 1, maxAvalableSeats);
                buildSegTree(mid + 1, hi, 2 * ind + 2, maxAvalableSeats);

                segTree[ind] = Math.max(segTree[2 * ind + 1], segTree[2 * ind + 2]);
                sum[ind] = sum[2 * ind + 1] + sum[2 * ind + 2];
            }


            public int gather(int lo, int hi, int k, int mR, int ind) {
                // base case
                /**
                 Method to check if seats are available in a single row
                 **/
                if (lo > mR || segTree[ind] < k) return -1;
                if (lo == hi) return lo;

                int mid = (lo + hi) / 2;

                int c = gather(lo, mid, k, mR, 2 * ind + 1);

                if (c == -1) {
                    c = gather(mid + 1, hi, k, mR, 2 * ind + 2);
                }
                return c;
            }

            public int gather(int k, int mR) {
                return gather(0, n - 1, k, mR, 0);
            }

            private void update(int lo, int hi, int pos, int val, int idx) {
                // base case
                /**
                 Method to update available in a single row
                 **/
                if (lo == hi) {
                    segTree[idx] = val;
                    sum[idx] = val;
                    return;
                }

                int mid = (lo + hi) / 2;
                // left seats are available in left half
                if (pos <= mid) {
                    update(lo, mid, pos, val, 2 * idx + 1);
                } else {
                    update(mid + 1, hi, pos, val, 2 * idx + 2);
                }
                segTree[idx] = Math.max(segTree[2 * idx + 1], segTree[2 * idx + 2]);
                sum[idx] = sum[2 * idx + 1] + sum[2 * idx + 2];
            }

            public void update(int pos, int val) {
                update(0, n - 1, pos, val, 0);
            }

            private long sumQuery(int lo, int hi, int k, int mR, int ind) {
                // base case '
                if (lo > mR || hi < k) return 0;
                if (lo >= k && hi <= mR) return sum[ind];
                int mid = (lo + hi) / 2;
                return sumQuery(lo, mid, k, mR, 2 * ind + 1) + sumQuery(mid + 1, hi, k, mR, 2 * ind + 2);
            }

            public long sumQuery(int k, int mR) {
                return sumQuery(0, n - 1, k, mR, 0);
            }
        }

        SegTree segTree;
        int[] rowSeats;

        public BookMyShow(int n, int m) {
            segTree = new SegTree(n, m);
            rowSeats = new int[n]; // rows to evaluate available seats
            Arrays.fill(rowSeats, m); // max available seats in a row
            return;
        }

        public int[] gather(int k, int maxRow) {
            int row = segTree.gather(k, maxRow);
            if (row == -1) return new int[]{};
            int col = segTree.m - rowSeats[row];
            rowSeats[row] -= k;  // reduce the seats
            segTree.update(row, rowSeats[row]);
            return new int[]{row, col};
        }

        public boolean scatter(int k, int maxRow) {
            long sum = segTree.sumQuery(k, maxRow);

            if (sum < k) return false;
            for (int i = 0; i <= maxRow && k != 0; i++) {
                if (rowSeats[i] > 0) {
                    long t = Math.min(rowSeats[i], k);
                    rowSeats[i] -= t;
                    k -= t;
                    segTree.update(i, rowSeats[i]);
                }
            }
            return true;
        }
    }

/**
 * Your BookMyShow object will be instantiated and called as such:
 * BookMyShow obj = new BookMyShow(n, m);
 * int[] param_1 = obj.gather(k,maxRow);
 * boolean param_2 = obj.scatter(k,maxRow);
 */
}

class NumArray {

    class SegTree {
        int n;
        int[] segTree;
        int[] sum;

        SegTree(int n) {
            this.n = n;
            segTree = new int[4 * n];
            sum = new int[4 * n];
            build(0, n - 1, 0, Integer.MAX_VALUE);
        }

        private void build(int lo, int hi, int idx, int val) {
            //base case
            if (lo == hi) {
                segTree[idx] = val;
                sum[idx] = val;
                return;
            }

            int mid = (lo + hi) / 2;
            build(lo, mid, 2 * idx + 1, val);
            build(mid + 1, hi, 2 * idx + 2, val);
            segTree[idx] = Math.max(segTree[2 * idx + 1], segTree[2 * idx + 2]);
            sum[idx] = sum[2 * idx + 1] + sum[2 * idx + 2];
        }

        private void update(int lo, int hi, int pos, int val, int idx) {
            // base case
            if (lo == hi) {
                segTree[idx] = val;
                sum[idx] = val;
                return;
            }

            int mid = (lo + hi) / 2;
            // left seats are available in left half
            if (pos <= mid) {
                update(lo, mid, pos, val, 2 * idx + 1);
            } else {
                update(mid + 1, hi, pos, val, 2 * idx + 2);
            }
            segTree[idx] = Math.max(segTree[2 * idx + 1], segTree[2 * idx + 2]);
            sum[idx] = sum[2 * idx + 1] + sum[2 * idx + 2];
        }

        private int sumQuery(int lo, int hi, int ind) {
            // base case '
            if (lo > ind || hi < ind) return 0;
            if (lo <= ind && hi >= ind) return sum[ind];
            int mid = (lo + hi) / 2;
            return sumQuery(lo, mid, 2 * ind + 1) + sumQuery(mid + 1, hi, 2 * ind + 2);
        }
    }

    SegTree segT;

    public NumArray(int[] nums) {
        segT = new SegTree(nums.length); // Initialise segTree with max values
        Arrays.fill(segT.segTree, Integer.MAX_VALUE);
    }

    public void update(int index, int val) {
        segT.update(0, segT.n - 1, index, val, 0);
    }

    public int sumRange(int left, int right) {
        return segT.sumQuery(left, right, 0);
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * obj.update(index,val);
 * int param_2 = obj.sumRange(left,right);
 */