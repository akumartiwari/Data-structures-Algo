package com.company;

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
}



