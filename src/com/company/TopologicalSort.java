package com.company;

import java.util.*;

public class TopologicalSort {

    // TOPOLOGICAL SORT
    // Use toplogical sort for indegree and pq to minisnmise the time taken to complete the course
    // TC = O(V+E) // As Simple DFS, SC = O(V) {Stack space}

    public int minimumTime(int n, int[][] relations, int[] time) {
        // create adjancy list of graph
        List<List<Integer>> graph = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
        }
        // create an indegree array for each node
        int[] indegree = new int[n];
        for (int[] e : relations) {
            // create the graph
            graph.get(e[0] - 1).add(e[1] - 1);
            indegree[e[1] - 1] += 1;
        }

        int maxTime = 0;
        // create a MIN-PQ to minimise time taken to complete all courses

        PriorityQueue<int[]> pq = new PriorityQueue<>(10, (t1, t2) -> t1[1] - t2[1]);

        //Insert all dead end nodes ie. nodes with 0 indegree
        for (int i = 0; i < n; i++) {
            if (indegree[i] == 0) pq.offer(new int[]{
                    i, time[i]
            });
        }

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int currCourse = curr[0];
            int currTime = curr[1];

            maxTime = Math.max(maxTime, currTime);

            // Visit all adjance vertex of curr node and update the time taken to complete the courses
            for (int next : graph.get(currCourse)) {
                // reduce indegreee by 1 as node as visited
                indegree[next] -= 1;

                // if indegree=0 means all its adajancent node has been visisted , Now move on to next  parent node
                if (indegree[next] == 0) {
                    pq.offer(new int[]{
                            next, currTime + time[next]
                    });
                }
            }
        }

        // return minTime Taken to  complete all courses
        return maxTime;
    }


    /*
      Steps:
        Build the graph using rowConditions
        Find topological sorting order for this graph
        Build one more graph using colConditions
        Find topological sorting order for this graph
        fill the matrix using the sorting order given by topological sort.
     */
    // TODO: Do it again
    private boolean buildGraph(int[][] conditions, int n, List<Integer> data) {
        List<Integer>[] adj = new ArrayList[n + 1];
        Arrays.fill(adj, new ArrayList<>());
        int[] indeg = new int[n + 1];

        for (int[] condition : conditions) {
            adj[condition[0]].add(condition[1]);
            indeg[condition[1]]++;
        }

        Queue<Integer> queue = new LinkedList<>();

        for (int i = 1; i <= n; i++) if (indeg[i] == 0) queue.offer(i);

        while (!queue.isEmpty()) {
            int node = queue.poll();
            data.add(node);
            for (int e : adj[node]) if (--indeg[e] == 0) queue.offer(e);
        }

        return data.size() == n;
    }

    public int[][] buildMatrix(int k, int[][] rowConditions, int[][] colConditions) {

        List<Integer> topRow = new ArrayList<>(), topCol = new ArrayList<>();

        if (!buildGraph(rowConditions, k, topRow) || !buildGraph(colConditions, k, topCol)) return new int[][]{};


        int[] row = new int[k + 1], col = new int[k + 1];

        for (int i = 0; i < k; i++) {
            row[topRow.get(i)] = i;
            col[topCol.get(i)] = i;
        }


        System.out.println(topRow);
        System.out.println(topCol);


        int[][] arr = new int[k][k];

        for (int i = 1; i <= k; i++) arr[row[i]][col[i]] = i;

        return arr;
    }
}
