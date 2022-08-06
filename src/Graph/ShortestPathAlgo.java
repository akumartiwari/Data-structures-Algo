package Graph;

import java.util.*;
import java.util.stream.Collectors;

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
                // if not visited then mark it visited and process it in queue
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
    /*
    Input: n = 5, roads = [[0,1],[1,2],[2,3],[0,2],[1,3],[2,4]]
    Output: 43
    Explanation: The figure above shows the country and the assigned values of [2,4,5,3,1].
    - The road (0,1) has an importance of 2 + 4 = 6.
    - The road (1,2) has an importance of 4 + 5 = 9.
    - The road (2,3) has an importance of 5 + 3 = 8.
    - The road (0,2) has an importance of 2 + 5 = 7.
    - The road (1,3) has an importance of 4 + 3 = 7.
    - The road (2,4) has an importance of 5 + 1 = 6.
    The total importance of all roads is 6 + 9 + 8 + 7 + 7 + 6 = 43.
    It can be shown that we cannot obtain a greater total importance than 43.=
    */
    //Author: Anand
    // TC = O(nlogn)
    public long maximumImportance(int n, int[][] roads) {
        Map<Integer, Integer> nodesConnectedCount = new HashMap<>();
        for (int[] r : roads) {
            nodesConnectedCount.put(r[0], nodesConnectedCount.getOrDefault(r[0], 0) + 1);
            nodesConnectedCount.put(r[1], nodesConnectedCount.getOrDefault(r[1], 0) + 1);
        }


        nodesConnectedCount = sortByValueInteger(nodesConnectedCount);

        Map<Integer, Integer> assign = new HashMap<>();
        int max = n;
        for (Map.Entry<Integer, Integer> entry : nodesConnectedCount.entrySet()) assign.put((int) entry.getKey(), n--);

        long ans = 0L;
        for (int[] r : roads) ans += assign.get(r[0]) + assign.get(r[1]);

        return ans;

    }

    // function to sort hashmap by values
    public static HashMap<Integer, Integer> sortByValueInteger(Map<Integer, Integer> hm) {

        return hm.entrySet()
        .stream()
        .sorted((i1, i2) -> i2.getValue().compareTo(i1.getValue()))
        .collect(Collectors.toMap(
                Map.Entry::getKey,
                Map.Entry::getValue,
                (e1, e2) -> e1, LinkedHashMap::new));
    }
}
