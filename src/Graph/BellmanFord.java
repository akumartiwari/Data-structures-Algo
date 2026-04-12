package Graph;

import java.util.Arrays;

/*
 * ============================================================
 * Algorithm: Bellman-Ford — Single Source Shortest Path
 * ============================================================
 * DESCRIPTION:
 *   Finds shortest distances from a source vertex to ALL other
 *   vertices in a weighted directed graph. Unlike Dijkstra,
 *   works with NEGATIVE edge weights and detects negative cycles.
 *
 * EXAMPLE:
 *   V=5, edges: 0→1(-1), 0→2(4), 1→2(3), 1→3(2), 1→4(2),
 *               3→2(5), 3→1(1), 4→3(-3)
 *   From src=0: dist = [0, -1, 2, -2, 1]
 *
 * ALGORITHM:
 *   1. Init dist[src]=0, all others = INF.
 *   2. Relax ALL edges V-1 times:
 *      if dist[src] + w < dist[dest] → update dist[dest]
 *   3. Run one more pass — if any edge still relaxes,
 *      a negative-weight cycle exists.
 *
 * TC: O(V * E)   — V = vertices, E = edges
 * SC: O(V)       — distance array
 * ============================================================
 */
public class BellmanFord {
    int V;
    int E;
    Edge[] edge;

    static class Edge {
        int src, dest, weight;

        Edge() {
            src = dest = weight = 0;
        }
    }

    BellmanFord(int v, int e) {
        this.V = v;
        this.E = e;
        edge = new Edge[e];
        for (int i = 0; i < e; i++) {
            edge[i] = new Edge();
        }
    }

    // Bellman-ford algorithm
    public static void main(String[] args) {
        int V = 5; // Number of vertices in graph
        int E = 8; // Number of edges in graph

        BellmanFord graph = new BellmanFord(V, E);

        // add edge 0-1 (or A-B in above figure)
        graph.edge[0].src = 0;
        graph.edge[0].dest = 1;
        graph.edge[0].weight = -1;

        // add edge 0-2 (or A-C in above figure)
        graph.edge[1].src = 0;
        graph.edge[1].dest = 2;
        graph.edge[1].weight = 4;

        // add edge 1-2 (or B-C in above figure)
        graph.edge[2].src = 1;
        graph.edge[2].dest = 2;
        graph.edge[2].weight = 3;

        // add edge 1-3 (or B-D in above figure)
        graph.edge[3].src = 1;
        graph.edge[3].dest = 3;
        graph.edge[3].weight = 2;

        // add edge 1-4 (or B-E in above figure)
        graph.edge[4].src = 1;
        graph.edge[4].dest = 4;
        graph.edge[4].weight = 2;

        // add edge 3-2 (or D-C in above figure)
        graph.edge[5].src = 3;
        graph.edge[5].dest = 2;
        graph.edge[5].weight = 5;

        // add edge 3-1 (or D-B in above figure)
        graph.edge[6].src = 3;
        graph.edge[6].dest = 1;
        graph.edge[6].weight = 1;

        // add edge 4-3 (or E-D in above figure)
        graph.edge[7].src = 4;
        graph.edge[7].dest = 3;
        graph.edge[7].weight = -3;

        graph.BF(graph, 0);
    }

    private void BF(BellmanFord graph, int src) {

        int V = graph.V;
        int E = graph.E;
        int[] dist = new int[V];

        // Step 1: Initialize distances from src to all other
        // vertices as INFINITE

        for (int i = 0; i < V; i++) dist[i] = Integer.MAX_VALUE;
        dist[src] = 0;

        // Relax edges |v|-1 times
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < E; j++) {

                int s = graph.edge[j].src;
                int d = graph.edge[j].dest;
                int w = graph.edge[j].weight;
                if (dist[s] != Integer.MAX_VALUE && dist[s] + w < dist[d])
                    dist[d] = dist[s] + w;
            }
        }


        // Step 3: check for negative-weight cycles. The above
        // step guarantees shortest distances if graph doesn't
        // contain negative weight cycle. If we get a shorter
        // path, then there is a cycle.

        for (int j = 0; j < E; ++j) {

            int s = graph.edge[j].src;
            int d = graph.edge[j].dest;
            int w = graph.edge[j].weight;
            if (dist[s] != Integer.MAX_VALUE && dist[s] + w < dist[d]) {
                System.out.println("Graph contains negative weight");
                return;
            }
        }

        printArr(dist, V);
    }

    private void printArr(int[] dist, int v) {
        System.out.println("vertex distance from src");

        for (int i = 0; i < v; i++)
            System.out.println(i + "\t\t" + dist[i]);
    }

    /*
     * ============================================================
     * Problem: Network Delay Time (LC 743) — Medium
     * ============================================================
     * DESCRIPTION:
     *   A signal is sent from node k in a directed weighted graph of n nodes.
     *   Find the time it takes for ALL nodes to receive the signal.
     *   Return -1 if it's impossible (some node unreachable).
     *
     * EXAMPLE:
     *   times=[[2,1,1],[2,3,1],[3,4,1]], n=4, k=2
     *   Signal: 2→1(1ms), 2→3(1ms), 3→4(1ms)
     *   All nodes reached at t=2  → Output: 2
     *
     * ALGORITHM (Bellman-Ford):
     *   1. Init dist[k-1]=0, all others = INF.
     *   2. Relax all edges n-1 times.
     *   3. If any dist[i] == INF → unreachable → return -1.
     *   4. Return max(dist) — the last node to receive the signal.
     *
     * TC: O(V * E)   — V = n nodes, E = times.length
     * SC: O(V)       — distance array
     * ============================================================
     */
    // Bellman ford
    public int networkDelayTime(int[][] times, int n, int k) {
        int[] dist = new int[n];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[k - 1] = 0;

        for (int i = 0; i < n; i++) {
            for (int[] time : times) {
                int src = time[0] - 1;
                int dest = time[1] - 1;
                int wt = time[2];
                if (dist[src] != Integer.MAX_VALUE && dist[src] + wt < dist[dest])
                    dist[dest] = dist[src] + wt;
            }
        }

        if (Arrays.stream(dist).anyMatch(x -> x == Integer.MAX_VALUE)) return -1;
        return Arrays.stream(dist).max().orElse(-1);
    }
}
