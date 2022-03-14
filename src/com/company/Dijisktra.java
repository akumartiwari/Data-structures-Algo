package com.company;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;

public class Dijisktra {
    // Author : Anand
    List<int[]>[] nextgraph, pregraph;

    public long minimumWeight(int n, int[][] edges, int src1, int src2, int dest) {
        buildGraph(n, edges);

        // To fetch shorstest path from all possible nodes
        long[] src1To = new long[n], src2To = new long[n], destTo = new long[n];
        Arrays.fill(src1To, -1);
        Arrays.fill(src2To, -1);
        Arrays.fill(destTo, -1);

        shortestPath(src1, src1To, nextgraph);
        shortestPath(src2, src2To, nextgraph);
        shortestPath(dest, destTo, pregraph);

        long res = -1;
        for (int i = 0; i < n; i++) {
            long d1 = src1To[i];
            long d2 = src2To[i];
            long d3 = destTo[i];

            if (d1 >= 0 && d2 >= 0 && d3 >= 0) {
                long d = d1 + d2 + d3;
                if (res == -1 || d < res) {
                    res = d;
                }
            }
        }
        return res;
    }

    // Dijisktra algorithm to find shortest distance b/w each node
    private void shortestPath(int src, long[] srcTo, List<int[]>[] graph) {
        // min PQ to find SD b/w src, dest
        PriorityQueue<long[]> queue = new PriorityQueue<>((a, b) -> Long.compare(a[1], b[1]));

        queue.add(new long[]{src, 0});

        while (!queue.isEmpty()) {
            long[] node = queue.poll();

            int to = (int) node[0];
            long dist = node[1];
            if (srcTo[to] != -1 && srcTo[to] <= dist) continue;
            srcTo[to] = dist;
            // For all adjacent nodes continue the process;
            for (int[] next : graph[to]) {
                queue.offer(new long[]{next[0], dist + next[1]});
            }
        }
    }

    private void buildGraph(int n, int[][] edges) {
        nextgraph = new ArrayList[n];
        pregraph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            nextgraph[i] = new ArrayList<int[]>();
            pregraph[i] = new ArrayList<int[]>();
        }

        for (int[] edge : edges) {
            int from = edge[0];
            int to = edge[1];
            int wt = edge[2];
            nextgraph[from].add(new int[]{to, wt});
            pregraph[to].add(new int[]{from, wt});
        }
    }

}
