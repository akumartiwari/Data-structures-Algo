package com.company;

import java.util.*;

public class MultisourceBFS {
    public static void main(String[] args) {

        int n = 4;
        List<Integer>[] graph = new ArrayList[n + 1];

        for (int i = 0; i <= n; i++) graph[i] = new ArrayList<>();

        // Adjacency list denoting road
        // between city u and v
        graph[1].add(2);
        graph[2].add(1);
        graph[2].add(3);
        graph[3].add(2);
        graph[3].add(4);
        graph[4].add(3);


        int[] v = new int[n + 1];
        // Array for storing whether ith
        // city is blocked or not
        v[1] = 0;
        v[2] = 1;
        v[3] = 1;
        v[4] = 0;


        System.out.print(multiBfs(n, v, graph));

    }

    private static int multiBfs(int n, int[] v, List<Integer>[] graph) {

        boolean[] vis = new boolean[n + 1];
        int max = 1;
        Arrays.fill(vis, false); // marked visited array as false
        // For each vertex apply BFS
        for (int i = 1; i <= n; i++) {
            if (!vis[i] && v[i] == 0) {
                max = Math.max(max, bfsUtil(graph, v, vis, i));
            }
        }
        return max;
    }

    private static int bfsUtil(List<Integer>[] graph, int[] v, boolean[] visited, int src) {
        int pathLen = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.add(src);
        visited[src] = true;
        while (!queue.isEmpty()) {
            int elem = queue.poll();
            for (int i = 0; i < graph[elem].size(); i++) {
                //  Not visited and path open
                if (!visited[graph[elem].get(i)] && v[graph[elem].get(i)] == 0) {
                    visited[graph[elem].get(i)] = true;
                    queue.add(graph[elem].get(i));
                    pathLen++;
                } else if (!visited[graph[elem].get(i)] && v[graph[elem].get(i)] == 1) // Not visited and path blocked
                    pathLen++;
            }
        }
        return pathLen + 1; // First node need to added as well
    }

}
