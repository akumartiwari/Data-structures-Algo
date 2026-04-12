package Graph;

import java.util.List;

/*
 * ============================================================
 * Problem: Course Schedule / Cycle Detection in Directed Graph
 *          (LC 207 variant) — Medium
 * ============================================================
 * DESCRIPTION:
 *   Given a directed graph of N nodes and edges, detect if a
 *   cycle exists. A cycle means you can return to the same
 *   node by following directed edges — making topological
 *   ordering impossible (e.g., circular course prerequisites).
 *
 * EXAMPLE:
 *   N=4, graph: 0→1, 1→2, 2→3, 3→1
 *   Node 1→2→3→1 forms a cycle  → return true
 *
 *   N=3, graph: 0→1, 1→2
 *   No back edge found           → return false
 *
 * ALGORITHM (DFS + Path Visited tracking):
 *   1. Maintain two arrays:
 *      - vis[]     → whether node has ever been visited
 *      - pathVis[] → whether node is on the CURRENT DFS path
 *   2. For each unvisited node, launch DFS:
 *      - Mark vis[node]=1 and pathVis[node]=1
 *      - For each neighbour:
 *          · If unvisited → recurse; cycle found? → return true
 *          · If already on current path (pathVis=1) → CYCLE found
 *      - On backtrack → reset pathVis[node]=0
 *        (allows node to be visited from a different start)
 *   3. Return false if no cycle found in any component.
 *
 * KEY INSIGHT: pathVis (not just vis) detects back edges.
 *   A visited node that is NOT on the current path is safe —
 *   it was already fully explored with no cycle found.
 *
 * TC: O(N + E)   — N = nodes, E = edges (each node/edge visited once)
 * SC: O(2N)      — vis + pathVis arrays; O(N) recursion stack
 * ============================================================
 */
public class CycleDetectionDAG {

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
