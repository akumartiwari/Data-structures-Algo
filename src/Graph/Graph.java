package Graph;

import java.util.*;


// Adjacency list representation of graph
public class Graph {
    private final int V;
    private final List<Integer>[] adj;
    boolean[] marked; // marked[v]=true, if vertex 'v' is visited
    int[] edgeTo; // edgeTo[v] -> previous vertex on path from s to v
    int s;

    public Graph(int v) {
        this.V = v;
        adj = new List[v];

        for (int i = 0; i < v; i++) {
            adj[i] = new ArrayList<>();
        }
    }

    public void addEdge(int v, int w) {
        adj[v].add(w);
        adj[w].add(v);
    }

    public Iterable<Integer> adj(int v) {
        return adj[v];
    }

    private void DepthFirstSearchPaths(Graph G, int s) {
        // Initialise all constructors
        this.marked = new boolean[G.V]; // marked[v]=true, if vertex 'v' is visited
        this.edgeTo = new int[V]; // edgeTo[v] -> previous vertex on path from s to v
        this.s = s;
        dfs(G, s);
    }

    /*
    Algo:-
    - Mark vertex  as visited.
    - Recursively visit all unmarked vertices adjacent to .
     */

    private void dfs(Graph g, int v) {
        marked[v] = true;
        for (int w : g.adj(v)) {
            if (!marked[w]) {
                dfs(g, w);
                edgeTo[w] = v; // previous vertex on path from v to w
            }
        }
    }

    // Prints path from vertex v to s
    public Iterable<Integer> pathTo(int v) {
        if (!hasPath(v)) return null;
        // Use A Stack to store path
        Stack<Integer> path = new Stack<>();
        for (int x = v; x != s; x = edgeTo[x]) path.push(x);
        path.push(s);
        return path;
    }

    private boolean hasPath(int v) {
        return marked[v];
    }

    /*
    Algorithm
    Repeat until queue is empty
    Remove vertex  from queue.
    Add to queue all unmarked vertices adjacent to  and mark them.
 */
    class BreadthFirstPaths {

        private boolean[] marked;
        private int[] edge;

        private void bfs(Graph G, int s) {

            Queue<Integer> q = new LinkedList<>();
            q.offer(s);
            marked[s] = true;
            while (!q.isEmpty()) {
                int v = q.poll();

                for (int w : G.adj(v)) {
                    if (marked[w]) continue;
                    q.offer(w);
                    marked[w] = true;
                    edgeTo[w] = v;
                }
            }
        }
    }


//    Connected Components
//    A connected component is a maximal set of connected vertices.

    // Count all connected components of a graph
    class CC {
        private final boolean[] marked;
        private final int[] id;
        private int count;

        public CC(Graph G) {
            marked = new boolean[G.V];
            id = new int[G.V];

            for (int i = 0; i < G.V; i++) {
                if (!marked[i]) {
                    visitConnectedCells(G, i);
                    count++;
                }
            }

            System.out.println(count());
        }

        // check id of component in which vertex v exists
        public int id(int v) {
            return id[v];
        }

        // count total number of components
        private int count() {
            return count;
        }

        private void visitConnectedCells(Graph g, int v) {
            marked[v] = true;
            id[v] = count;
            for (int w : g.adj(v)) {
                if (!marked[w]) visitConnectedCells(g, w);
            }
        }
    }

    //Directed Graphs
    //Quick Find - Eager Approach
    class QuickFindUFEager {
        private int[] id;

        QuickFindUFEager(int N) {
            id = new int[N];
            for (int i = 0; i < N; i++) id[i] = i;
        }


        // check whether p and q are in the same component (2 array accesses)
        private boolean isConnected(int p, int q) {
            return id[p] == id[q];
        }


        // change all entries with id[p] to id[q] (at most 2N + 2 array accesses)
        private void union(int p, int q) {
            int pid = id[p];
            int qid = id[q];
            for (int i = 0; i < id.length; i++)
                if (id[i] == pid) id[i] = qid;
        }
    }

    // Quick Find - Lazy Approach
    class QuickFindUFLazy {
        private int[] id;

        QuickFindUFLazy(int N) {
            id = new int[N];
            for (int i = 0; i < N; i++) id[i] = i;
        }

        // check if p and q have same root (depth of p and q array accesses)
        private boolean isConnected(int p, int q) {
            return id[p] == id[q];
        }

        // chase parent pointers until reach root (depth of i array accesses)
        private int root(int p) {
            while (id[p] != p) p = id[p];
            return p;
        }

        // change root of p to point to root of q (depth of p and q array accesses)
        private void union(int p, int q) {
            int i = root(p);
            int j = root(q);
            id[i] = j;
        }
    }


    // Weighted Quick Union
    class WeightQuickUnionUF {

        int[] id, sz;

        WeightQuickUnionUF(int N) {
            id = new int[N];
            sz = new int[N];

            for (int i = 0; i < N; i++) {
                id[i] = i;
                sz[i] = i;
            }
        }

        // check if p and q have same root (depth of p and q array accesses)
        private boolean isConnected(int p, int q) {
            return id[p] == id[q];
        }

        // chase parent pointers until reach root (depth of i array accesses)
        private int root(int p) {
            while (id[p] != p) p = id[p];
            return p;
        }


        // change root of p to point to root of q (depth of p and q array accesses)
        private void union(int p, int q) {
            int i = root(p);
            int j = root(q);
            if (i == j) return;
            if (sz[i] < sz[j]) {
                id[i] = j; // connect node smaller parent wt with larger parent wt
                sz[j] += sz[i];
            } else {
                id[j] = i; // connect node smaller parent wt with larger parent wt
                sz[i] += sz[j];
            }
        }
    }

    // Quick Union + path comparison
    class QuickUnionPathComparisonUF {

        int[] id;

        QuickUnionPathComparisonUF(int N) {
            id = new int[N];
            for (int i = 0; i < N; i++) id[i] = i;
        }

        // check if p and q have same root (depth of p and q array accesses)
        private boolean isConnected(int p, int q) {
            return root(p) == root(q);
        }

        // chase parent pointers until reach root (depth of i array accesses)
        private int root(int i) {
            while (id[i] != i) {
                id[i] = id[id[i]];
                i = id[i];
            }

            return i;
        }


        // change root of p to point to root of q (depth of p and q array accesses)
        private void union(int p, int q) {
            int i = root(p);
            int j = root(q);
            id[i] = j;
        }
    }

        /*
    Input: n = 7, edges = [[0,2],[0,5],[2,4],[1,6],[5,4]]
    Output: 14
    Explanation: There are 14 pairs of nodes that are unreachable from each other:
    [[0,1],[0,3],[0,6],[1,2],[1,3],[1,4],[1,5],[2,3],[2,6],[3,4],[3,5],[3,6],[4,6],[5,6]].
    Therefore, we return 14.
     */

    //Author: Anand
    private int count;

    public long countPairs(int n, int[][] edges) {
        Map<Integer, List<Integer>> graph = new HashMap<>();

        for (int[] edge : edges) {
            if (graph.containsKey(edge[0])) {
                List<Integer> exist = graph.get(edge[0]);
                exist.add(edge[1]);
                graph.put(edge[0], exist);
            } else graph.put(edge[0], new ArrayList<>(Arrays.asList(edge[1])));

            if (graph.containsKey(edge[1])) {
                List<Integer> exist = graph.get(edge[1]);
                exist.add(edge[0]);
                graph.put(edge[1], exist);
            } else graph.put(edge[1], new ArrayList<>(Arrays.asList(edge[0])));
        }

        List<Integer> components = new ArrayList<>();

        int[] visited = new int[n];
        for (int i = 0; i < n; ++i) {
            if (visited[i] == 0) {
                count = 0;
                DFS(i, graph, visited);
                components.add(count);
            }
        }

        long ans = 0;
        long sum = 0;

        for (int c : components) {
            sum += c;
            ans += (long) c * (n - sum);
        }

        return ans;
    }

    private void DFS(int start, Map<Integer, List<Integer>> graph, int[] visited) {
        visited[start] = 1;
        count++;
        for (int i = 0; i < (graph.containsKey(start) ? graph.get(start).size() : 0); ++i) {
            if (visited[(graph.get(start).get(i))] == 0)
                DFS(graph.get(start).get(i), graph, visited);
        }
    }
}



