package Graph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

public class ShortestPathAlgo {
    // TC = O(E + V)
    public static void main(String[] args) {

        // No of vertices
        int v = 8;
        ArrayList<ArrayList<Integer>> adj = new ArrayList<ArrayList<Integer>>(v);

        for (int i = 0; i < v; i++) adj.add(new ArrayList<Integer>());
        addEdge(adj, 0, 1);
        addEdge(adj, 0, 3);
        addEdge(adj, 1, 2);
        addEdge(adj, 3, 4);
        addEdge(adj, 3, 7);
        addEdge(adj, 4, 5);
        addEdge(adj, 4, 6);
        addEdge(adj, 4, 7);
        addEdge(adj, 5, 6);
        addEdge(adj, 6, 7);
        int source = 0, dest = 7;
        printShortestDistance(adj, source, dest, v);
    }

    private static void printShortestDistance(ArrayList<ArrayList<Integer>> adj, int source, int dest, int v) {
        // predesessor array --> used to trace path in bfs
        int[] pred = new int[v];
        int[] dist = new int[v]; // to store min dist from source
        if (!BFS(adj, source, dest, pred, dist, v)) {
            System.out.println("Source and destination are not connected");
            return;
        }


        // Fetch-path
        // crwal in reverse direction
        LinkedList<Integer> path = new LinkedList<>();
        int crawl = dest;
        while (pred[crawl] != -1) {
            path.add(pred[crawl]);
            crawl = pred[crawl];
        }

        // shortest distance
        System.out.println(dist[dest]);
        // print path
        for (int i = path.size() - 1; i >= 0; i--) System.out.print(path.get(i) + " ");
    }

    private static boolean BFS(ArrayList<ArrayList<Integer>> adj, int source, int dest, int[] pred, int[] dist, int v) {
        Arrays.fill(pred, -1);
        Arrays.fill(dist, Integer.MAX_VALUE);
        boolean[] visited = new boolean[v];
        Queue<Integer> queue = new LinkedList<>();
        queue.add(source);
        visited[source] = true;
        dist[source] = 0;

        while (!queue.isEmpty()) {
            int u = queue.poll();
            for (int i = 0; i < adj.get(u).size(); i++) {
                if (visited[adj.get(u).get(i)]) continue;
                // if not visited then mark it visted and process it in queue
                visited[adj.get(u).get(i)] = true;
                queue.offer(adj.get(u).get(i));
                dist[adj.get(u).get(i)] = 1 + dist[u];
                pred[adj.get(u).get(i)] = u;
                if (adj.get(u).get(i) == dest) return true;
            }
        }
        return false;
    }

    private static void addEdge(ArrayList<ArrayList<Integer>> adj, int vertex1, int vertex2) {
        adj.get(vertex1).add(vertex2);
        adj.get(vertex2).add(vertex1);
    }
}
