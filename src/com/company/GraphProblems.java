package com.company;

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;

public class GraphProblems {

    /*
    For each edge (i, j) in edges,
    we find a neighbour ii of node i,
    we find a neighbour jj of node i,
    If ii, i, j,jj has no duplicate, then that's a valid sequence.

    Ad the intuition mentioned,
    we don't have to enumearte all neignbours,
    but only some nodes with big value.

    But how many is enough?
    I'll say 3.
    For example, we have ii, i, j now,
    we can enumerate 3 of node j biggest neighbour,
    there must be at least one node different node ii and node i.

    So we need to iterate all edges (i, j),
    for each node we keep at most 3 biggest neighbour, which this can be done in O(3) or O(log3).

     */
    // Author: Anand
    // TC = O(n+m); n = length of edges, m = max neighbours of a node
    // SC = O(n)
    public int maximumScore(int[] scores, int[][] edges) {

        int maxSum = -1;
        int n = scores.length;
        int maxscore = 0;
        Node[] list = new Node[n];

        for (int i = 0; i < list.length; i++) {
            list[i] = new Node(scores[i]);
            maxscore = Math.max(scores[i], maxscore);
        }
        for (int[] edge : edges) {
            Node start = list[edge[0]];
            Node end = list[edge[1]];
            start.add(end);
        }

        for (int[] edge : edges) {
            Node start = list[edge[0]];
            Node end = list[edge[1]];

            if (start.value + end.value + maxscore + maxscore <= maxSum) continue;
            Queue<Node> queue = new PriorityQueue<>((a, b) -> b.value - a.value);

            for (Node node : start.next) {
                if (node != end) queue.offer(node);
            }

            if (queue.size() == 0) continue;
            // pull 2 adjacent neighbours out of queue
            Node start1 = queue.poll();
            Node start2 = queue.poll();

            queue.clear();

            for (Node node : end.next) {
                if (node != start) queue.offer(node);
            }

            if (queue.size() == 0) continue;
            // pull 2 adjacent neighbours out of queue
            Node end1 = queue.poll();
            Node end2 = queue.poll();

            int sum = start.value + end.value + start1.value;

            if (start1 != end1) {
                sum += end1.value;
            } else if (start2 == null && end2 == null) {
                continue;
            } else if (start2 != null && end2 != null) {
                sum += Math.max(start2.value, end2.value);
            } else if (start2 == null) {
                sum += end2.value;
            } else {
                sum += start2.value;
            }
            maxSum = Math.max(sum, maxSum);

        }
        return maxSum;
    }

    static class Node {
        int value;
        Set<GraphProblems.Node> next = new HashSet<>();

        Node(int val) {
            this.value = val;
        }

        void add(GraphProblems.Node node) {
            next.add(node); // a ---> [b]
            node.next.add(this); // b --> [a]
        }
    }


}
