package Graph;

import javafx.util.Pair;

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
        private final int[] id;

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
        private final int[] id;

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
            } else graph.put(edge[0], new ArrayList<>(Collections.singletonList(edge[1])));

            if (graph.containsKey(edge[1])) {
                List<Integer> exist = graph.get(edge[1]);
                exist.add(edge[0]);
                graph.put(edge[1], exist);
            } else graph.put(edge[1], new ArrayList<>(Collections.singletonList(edge[0])));
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

    /*
    Input: nums = [1,5,5,4,11], edges = [[0,1],[1,2],[1,3],[3,4]]
    Output: 9
    Explanation: The diagram above shows a way to make a pair of removals.
    - The 1st component has nodes [1,3,4] with values [5,4,11]. Its XOR value is 5 ^ 4 ^ 11 = 10.
    - The 2nd component has node [0] with value [1]. Its XOR value is 1 = 1.
    - The 3rd component has node [2] with value [5]. Its XOR value is 5 = 5.
    The score is the difference between the largest and smallest XOR value which is 10 - 1 = 9.
    It can be shown that no other pair of removals will obtain a smaller score than 9.
    Author: Anand

    ### Solution:-
    Property ->  a^a=0, 0^a=a

    The idea is to use this property via below algorithm
    - Do dfs from root and store xor for each node's subtree
    - Store ancestors for each node
    - Now for each pair of edges there are couple of ways to select them
        1. both edges can be part of same side
            like :-   / -> edge1
                     /  -> edge2

        2. edges are part of diffent side
              like :-    |
                (edge1) / \ (edge2)

    - For the above 2 cases calculate XOR of each of 3 segmemnt via XOR property
    and minimise ans.

    - Return minimum ans.
     */
    class Solution {
        int[] nums;
        Map<Integer, List<Integer>> adj;
        int[] xor;
        Set<Integer>[] ancestors;

        public int minimumScore(int[] nums, int[][] edges) {

            this.nums = nums;
            adj = new HashMap<>();
            xor = new int[nums.length];
            ancestors = new Set[nums.length];


            int ans = Integer.MAX_VALUE;
            for (int[] edge : edges) {
                if (adj.containsKey(edge[0])) {
                    List<Integer> exist = adj.get(edge[0]);
                    exist.add(edge[1]);
                    adj.put(edge[0], exist);
                } else adj.put(edge[0], new ArrayList<>(Collections.singletonList(edge[1])));

                if (adj.containsKey(edge[1])) {
                    List<Integer> exist = adj.get(edge[1]);
                    exist.add(edge[0]);
                    adj.put(edge[1], exist);
                } else adj.put(edge[1], new ArrayList<>(Collections.singletonList(edge[0])));
            }


            dfs(0, -1, new ArrayList<>());


            for (int i = 0; i < edges.length; i++) {
                for (int j = i + 1; j < edges.length; j++) {
                    int subNode1 = getSubRoot(edges[i]), subNode2 = getSubRoot(edges[j]);
                    int xc = xor[0], xa = xor[subNode1], xb = xor[subNode2];
                    // if both child subTree lies under same side
                    if (ancestors[subNode2].contains(subNode1)) {
                        xc ^= xa;
                        xa ^= xb;
                    } else if (ancestors[subNode1].contains(subNode2)) {
                        xc ^= xb;
                        xb ^= xa;
                    }
                    // They lies under different subtree
                    else {
                        xc ^= xa;
                        xc ^= xb;
                    }

                    int min = Math.min(xc, Math.min(xa, xb));
                    int max = Math.max(xc, Math.max(xa, xb));
                    ans = Math.min(ans, max - min);
                }
            }

            return ans;
        }

        private int dfs(int i, int parent, List<Integer> path) {
            int ans = nums[i];
            ancestors[i] = new HashSet<>();
            ancestors[i].addAll(path);
            path.add(i);

            for (int child : adj.get(i)) {
                if (child != parent) {
                    ans ^= dfs(child, i, path);
                }
            }

            path.remove(path.size() - 1);
            return xor[i] = ans;

        }

        private int getSubRoot(int[] edge) {
            int i = edge[0];
            int j = edge[1];
            if (ancestors[i].contains(j)) return i;
            return j;
        }
    }

    public int maxStarSum(int[] vals, int[][] edges, int k) {
        Map<Integer, List<Integer>> adj = new HashMap<>();
        if (edges.length == 0) {
            if (vals.length >= 1) {
                return Arrays.stream(vals).max().getAsInt();
            }
            return 0;
        }

        int ans = Integer.MIN_VALUE;
        for (int[] edge : edges) {
            if (!adj.containsKey(edge[0])) adj.put(edge[0], new ArrayList<>());
            adj.get(edge[0]).add(edge[1]);

            if (!adj.containsKey(edge[1])) adj.put(edge[1], new ArrayList<>());
            adj.get(edge[1]).add(edge[0]);
        }
        System.out.println(adj);

        for (Map.Entry<Integer, List<Integer>> entry : adj.entrySet()) {
            List<Integer> nodes = entry.getValue();
            nodes.add(entry.getKey());

            System.out.println(entry.getKey() + ":" + Arrays.toString(nodes.toArray()));

            ans = Math.max(ans, mss(nodes.stream().mapToInt(x -> x).toArray(), vals, k));
            System.out.println(ans);
        }

        return ans;
    }

    private int mss(int[] arr, int[] vals, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>(Collections.reverseOrder());
        for (int a : arr) pq.offer(vals[a]);


        if (pq.size() == 1) return pq.poll();
        int len = k, sum = 0;
        while (len-- >= 0 & !pq.isEmpty()) {
            int element = pq.poll();
            if (element <= 0) return sum;
            sum += element;
        }

        return sum;
    }
    // The left node has the value 2 * val, and
    // The right node has the value 2 * val + 1.
    // The concept is to find LCA for each node in query
    // and then find the distance of node with its LCA
    // the total length of cycle is l1+l2+1
    public int[] cycleLengthQueries(int n, int[][] queries) {
        int[] ans = new int[queries.length];
        int ind = 0;
        for (int[] query : queries) {
            int x = query[0], y = query[1];
            while (x != y) {
                if (x > y) x /= 2;
                else y /= 2;
                ans[ind]++;
            }
            ans[ind++]++;
        }
        return ans;
    }
}



