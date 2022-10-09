package com.company;

import java.util.*;

// Topological ordering of a directed graph is a linear ordering of its vertices such that
// for every directed edge uv from vertex u to vertex v, u comes before v in the ordering.
public class TopologicalSort {

    // TOPOLOGICAL SORT
    // Use toplogical sort for indegree and pq to minisnmise the time taken to complete the course
    // TC = O(V+E) // As Simple DFS, SC = O(V) {Stack space}


    // TOPO Sort algorithm using DFS
    // The idea is to do dfs for all nodes after marking them visited,
    // after returning from recursion calls add them to stack

    public int[] topoSort(int N, List<List<Integer>> graph) {
        Stack<Integer> stk = new Stack<>();
        int[] vis = new int[N];

        for (int i = 0; i < N; i++) {
            if (vis[i] == 0) {
                findTopoSort(i, vis, graph, stk);
            }
        }

        // Pop elements out of stack to get final result
        int[] ans = new int[N];
        int ind = 0;
        while (!stk.empty()) {
            ans[ind++] = stk.pop();
        }

        return ans;
    }


    //TC = O(N+E), SC = O(N), ASS = O(N)
    private void findTopoSort(int node, int[] vis, List<List<Integer>> graph, Stack<Integer> stk) {
        vis[node] = 1;

        for (int e : graph.get(node)) {
            if (vis[e] == 0)
                findTopoSort(e, vis, graph, stk);
        }

        stk.push(node);
    }

    /*
      Steps:
        Build the graph using rowConditions
        Find topological sorting order for this graph
        Build one more graph using colConditions
        Find topological sorting order for this graph
        fill the matrix using the sorting order given by topological sort.
     */

    //TC  = O(N+E), SC = O(N) + O(N)
    // TOPO Sort using BFS algorithm
    public List<Integer> generateTopoSort(int[][] conditions, int k) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < k; i++) graph.add(new ArrayList<>());

        int[] indegree = new int[k];
        List<Integer> ans = new ArrayList<>();


        for (int[] condition : conditions) {
            graph.get(condition[0] - 1).add(condition[1] - 1);
            indegree[condition[1]]++;
        }

        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < k; i++) {
            if (indegree[i] == 0) queue.offer(i);
        }

        while (!queue.isEmpty()) {
            int element = queue.poll();

            ans.add(element + 1);
            for (int e : graph.get(element)) {
                if (--indegree[e] == 0) queue.add(e);
            }
        }
        return ans;
    }

    public int[][] buildMatrix(int k, int[][] rowConditions, int[][] colConditions) {
        List<Integer> order1 = generateTopoSort(rowConditions, k);
        List<Integer> order2 = generateTopoSort(colConditions, k);


        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < k; i++) {
            map.put(order1.get(i), i);
        }

        int[][] ans = new int[k][k];
        for (int i = 0; i < k; i++) {
            ans[i][map.get(order1.get(i))] = order1.get(i);
        }

        return ans;
    }

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
        for (int i = 0; i < n; i++)
            if (indegree[i] == 0) pq.offer(new int[]{i, time[i]});

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int currCourse = curr[0];
            int currTime = curr[1];

            maxTime = Math.max(maxTime, currTime);

            // Visit all adjance vertex of curr node and update the time taken to complete the courses
            for (int next : graph.get(currCourse)) {
                // reduce indegreee by 1 as node as visited
                indegree[next] -= 1;

                // if indegree=0 means all its adjancent node has been visited , Now move on to next  parent node
                if (indegree[next] == 0) {
                    pq.offer(new int[]{next, currTime + time[next]});
                }
            }
        }

        // return minTime Taken to  complete all courses
        return maxTime;
    }

}

