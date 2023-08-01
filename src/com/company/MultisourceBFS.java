package com.company;

import javafx.util.Pair;

import java.util.HashMap;import java.util.*;
import java.util.HashMap;

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

    class Solution {
        public long minimumFuelCost(int[][] roads, int seats) {
            int n = roads.length;
            Map<Integer, List<Integer>> graph = new HashMap<>();

            for (int[] road : roads) {
                if (!graph.containsKey(road[0])) graph.put(road[0], new ArrayList<>());
                graph.get(road[0]).add(road[1]);
                if (!graph.containsKey(road[1])) graph.put(road[1], new ArrayList<>());
                graph.get(road[1]).add(road[0]);
            }

            boolean[] vis = new boolean[n + 1];
            Arrays.fill(vis, false); // marked visited array as false
            List<Integer> ends = new ArrayList<>();

            for (int key : graph.keySet()) {
                if (graph.get(key).size() == 1) ends.add(key);
            }

            int petrol = 0;
            if (n == 0) return 0;

            System.out.println(graph);
            System.out.println(ends);


            if (ends.size() == n) return n;
            for (int extreme : ends) {
                if (!vis[extreme]) {
                    petrol += bfsUtil(graph, vis, extreme, seats - 1);
                }
            }
            return 1 + petrol;
        }


        // Dijkstra algorithm to find the shortest distance b/w each node
        private void shortestPath(int src, long[] srcTo, List<List<Integer>> graph) {
            // min PQ to find SD b/w src, dest
            PriorityQueue<long[]> queue = new PriorityQueue<>((a, b) -> Long.compare(a[1], b[1]));

            queue.add(new long[]{src, 0});

            while (!queue.isEmpty()) {
                long[] node = queue.poll();
                int to = (int) node[0];
                long dist = node[1];
                if (srcTo[to] != Long.MAX_VALUE && srcTo[to] <= dist) continue;
                srcTo[to] = dist;
                // For all adjacent nodes continue the process;
                for (int next : graph.get(to)) {
                    queue.offer(new long[]{next, dist + 1});
                }
            }
        }


        private int bfsUtil(Map<Integer, List<Integer>> graph, boolean[] visited, int src, int seats) {
            int pathLen = 0;
            Queue<Pair<Integer, Integer>> queue = new PriorityQueue<>();
            queue.add(new Pair<>(src, seats));
            visited[src] = true;

            while (!queue.isEmpty()) {
                Pair<Integer, Integer> elem = queue.poll();

                if (elem.getKey() == 0) return pathLen;

                for (int i = 0; i < graph.get(elem.getKey()).size(); i++) {
                    //  Not visited and seats are available
                    if (!visited[graph.get(elem.getKey()).get(i)] && elem.getValue() > 0) {
                        visited[graph.get(elem.getKey()).get(i)] = true;
                        queue.add(new Pair<>(graph.get(elem.getKey()).get(i), seats - 1));
                        pathLen++;
                    }

                    // Not visited and seats are full
                    else if (!visited[graph.get(elem.getKey()).get(i)] && elem.getValue() <= 0) {
                        visited[graph.get(elem.getKey()).get(i)] = true;
                        queue.add(new Pair<>(graph.get(elem.getKey()).get(i), seats - 1));
                        pathLen += 2;
                    }
                }
            }
            return pathLen;
        }

    }
}