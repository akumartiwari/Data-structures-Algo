package Graph;

import java.util.List;

public class CycleDetectionDAG {

    // Cycle Detection in DAG using DFS

    // ALGO-
    // Make dfs call for each unvisited node and check if its in same path
    // else for each backtrack(pathVis[node] = 0)
    // to make sure it can be visited again via different starting node


    // TC = O(N+E), SC = O(2N), ASC = O(N)
    public boolean isCycle(int N, List<List<Integer>> graph) {

        int[] vis = new int[N];
        int[] pathVis = new int[N];

        for (int i = 0; i < N; i++) {
            if (vis[i] == 0)
                if (checkCycle(i, vis, graph, pathVis)) return true;
        }


        // Logic for  cycle detection in directed acyclic graph
        return false;
    }

    private boolean checkCycle(int node, int[] vis, List<List<Integer>> graph, int[] pathVis) {
        vis[node] = 1;
        pathVis[node] = 1;

        for (int n : graph.get(node)) {
            if (vis[n] == 0) if (checkCycle(n, vis, graph, pathVis)) return true;
            else if (pathVis[n] == 1) return true;
        }

        pathVis[node] = 0;
        return false;
    }

   // Cycle Detection in DAG using BFS


}
