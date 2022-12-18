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

//TODO- TBD
class ST {
    public int[] cycleLengthQueries(int n, int[][] queries) {
        long[] arr = new long[n];
        for (int i = 0; i < n; i++) arr[i] = i + 1;
        segmentTree st = new segmentTree(arr);
        List<Integer> ans = new ArrayList<>();
        for (int[] query : queries) {
            // Find cycle size of tree

            int start = query[0];
            int end = query[1];

            Map<Integer, List<Long>> graph = new HashMap<>();// node, adjacent nodes
            buildGraph(arr, graph);

            boolean[] visited = new boolean[n];
            Arrays.fill(visited, false);
            ans.add(bfs(start, end, graph, visited));
        }

        return ans.stream().mapToInt(x -> x).toArray();
    }


    private void buildGraph(long[] arr, Map<Integer, List<Long>> graph) {
        // base case
        for (long a : arr) {
            if (!graph.containsKey((int) a)) graph.put((int) a, new ArrayList<>());
            graph.get(a).add(2 * a);
            graph.get(a).add(2 * a + 1);
        }
    }

    private int bfs(int start, int end, Map<Integer, List<Long>> graph, boolean[] visited) {
        // base case
        PriorityQueue<Pair<Integer, Integer>> queue = new PriorityQueue<>((a, b) -> Integer.compare(a.getValue(), b.getValue())); // MAX PQ based on distance of node from start
        queue.add(new Pair<>(start, 0));
        visited[start] = true;
        while (!queue.isEmpty()) {
            Pair<Integer, Integer> pair = queue.poll();
            if (pair.getKey() == end) return pair.getValue();
            for (long e : graph.get(start)) {
                if (!visited[(int) e]) {
                    visited[(int) e] = true;
                    queue.add(new Pair<>((int) e, pair.getValue() + 1));
                }
            }
        }
        return -1;
    }

    public static class segmentTree {

        public long[] arr;
        public long[] tree;
        public long[] lazy;

        segmentTree(long[] array) {
            int n = array.length;
            arr = new long[n];
            System.arraycopy(array, 0, arr, 0, n);
            tree = new long[4 * n + 1];
            lazy = new long[4 * n + 1];
        }

        public void build(int[] arr, int s, int e, int[] tree, int index) {

            if (s == e) {
                tree[index] = arr[s];
                return;
            }

            //otherwise divide in two parts and fill both sides simply
            int mid = (s + e) / 2;
            build(arr, s, mid, tree, 2 * index);
            build(arr, mid + 1, e, tree, 2 * index + 1);

            //who will build the current position dude
            tree[index] = Math.min(tree[2 * index], tree[2 * index + 1]);
        }

        public int query(int sr, int er, int sc, int ec, int index, int[] tree) {

            if (lazy[index] != 0) {
                tree[index] += lazy[index];

                if (sc != ec) {
                    lazy[2 * index + 1] += lazy[index];
                    lazy[2 * index] += lazy[index];
                }

                lazy[index] = 0;
            }

            //no overlap
            if (sr > ec || sc > er) return Integer.MAX_VALUE;

            //found the index baby
            if (sr <= sc && ec <= er) return tree[index];

            //finding the index on both sides hehehehhe
            int mid = (sc + ec) / 2;
            int left = query(sr, er, sc, mid, 2 * index, tree);
            int right = query(sr, er, mid + 1, ec, 2 * index + 1, tree);

            return Integer.min(left, right);
        }

        //now we will do point update implementation
        //it should be simple then we expected for sure
        public void update(int index, int indexr, int increment, int[] tree, int s, int e) {

            if (lazy[index] != 0) {
                tree[index] += lazy[index];

                if (s != e) {
                    lazy[2 * index + 1] = lazy[index];
                    lazy[2 * index] = lazy[index];
                }

                lazy[index] = 0;
            }

            //no overlap
            if (indexr < s || indexr > e) return;

            //found the required index
            if (s == e) {
                tree[index] += increment;
                return;
            }

            //search for the index on both sides
            int mid = (s + e) / 2;
            update(2 * index, indexr, increment, tree, s, mid);
            update(2 * index + 1, indexr, increment, tree, mid + 1, e);

            //now update the current range simply
            tree[index] = Math.min(tree[2 * index + 1], tree[2 * index]);
        }

        public void rangeUpdate(int[] tree, int index, int s, int e, int sr, int er, int increment) {

            //if not at all in the same range
            if (e < sr || er < s) return;

            //complete then also move forward
            if (s == e) {
                tree[index] += increment;
                return;
            }

            //otherwise move in both subparts
            int mid = (s + e) / 2;
            rangeUpdate(tree, 2 * index, s, mid, sr, er, increment);
            rangeUpdate(tree, 2 * index + 1, mid + 1, e, sr, er, increment);

            //update current range too na
            //i always forget this step for some reasons hehehe, idiot
            tree[index] = Math.min(tree[2 * index], tree[2 * index + 1]);
        }

        public void rangeUpdateLazy(int[] tree, int index, int s, int e, int sr, int er, int increment) {

            //update lazy values
            //resolve lazy value before going down
            if (lazy[index] != 0) {
                tree[index] += lazy[index];

                if (s != e) {
                    lazy[2 * index + 1] += lazy[index];
                    lazy[2 * index] += lazy[index];
                }

                lazy[index] = 0;
            }

            //no overlap case
            if (sr > e || s > er) return;

            //complete overlap
            if (sr <= s && er >= e) {
                tree[index] += increment;

                if (s != e) {
                    lazy[2 * index + 1] += increment;
                    lazy[2 * index] += increment;
                }
                return;
            }

            //otherwise go on both left and right side and do your shit
            int mid = (s + e) / 2;
            rangeUpdateLazy(tree, 2 * index, s, mid, sr, er, increment);
            rangeUpdateLazy(tree, 2 * index + 1, mid + 1, e, sr, er, increment);

            tree[index] = Math.min(tree[2 * index + 1], tree[2 * index]);

        }
    }
}