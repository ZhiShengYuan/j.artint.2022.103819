/****************************************************
 * DeepOpt-MWDS
 * A C++ Implementation for the MWDS problem
 *
 * This code implements:
 *   1) Data structures for a vertex-weighted graph
 *   2) Five reduction rules
 *   3) ConstructDS to build an initial solution
 *   4) CC2V3+ configuration checking
 *   5) Main local search with frequency-based scoring
 *   6) DeepOpt-based perturbations
 *
 * Author: Zenkexi
 * Date: 2025/3/15
 * License: MIT
 * comment written by: ChatGpt O1-preview
 ****************************************************/
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <fstream>
#include <limits>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <cassert>

 //----------------------------------------------------
 // PART 1. Data Structures
 //----------------------------------------------------

struct Graph {
    int n;  // number of vertices
    int m;  // number of edges
    std::vector<long long> weight; // weight[v]
    std::vector<std::vector<int>> adj; // adjacency list

    // For convenience, store degree if needed
    std::vector<int> deg;

    Graph(int n_ = 0) : n(n_), m(0) {
        adj.resize(n_);
        weight.resize(n_, 1LL);
        deg.resize(n_, 0);
    }
};

//
// A simple container to hold the current solution data
// including configuration arrays and frequency arrays
//
struct MWDSContext {
    // The candidate solution
    // If D[v] == true, vertex v is in the solution
    std::vector<bool> inSolution;

    // For each vertex, how often it remains non-dominated
    // or how frequently it was needed
    std::vector<long long> freq;

    // Configuration-check array for CC2V3+
    // conf[v] in {0,1,2}
    std::vector<int> conf;

    // Age of each vertex (used to break ties by oldest)
    std::vector<long long> age;

    // For quick checking if the graph is dominated
    // or for counting # undominated vertices
    // undominatedCount[v] = how many solution vertices currently dominate v
    // If undominatedCount[v] == 0, vertex v is not dominated
    std::vector<int> coverCount;
};

// A struct used to revert solutions changed by certain reduction rules
// M[v] = -1 if not changed, otherwise M[v] = index of a vertex that caused the weight transform
struct RevertMap {
    std::vector<int> mapping;
    RevertMap(int n) : mapping(n, -1) {}
};

// Random engine used throughout
static std::mt19937_64 rng(std::chrono::steady_clock::now().time_since_epoch().count());

//----------------------------------------------------
// PART 2. Reduction Rules
//----------------------------------------------------
/*
   Weighted-Degree-0:
     If deg(v) = 0, we must fix v into solution.
   Weighted-Degree-1 (two variants):
     If v is a pendant of u, compare w(v) and w(u).
   Weighted-Degree-2 (two variants):
     If (v1, u) and (v2, u) are edges, check weight relations.

   We'll store the "fixed" vertices in a vector fixedSet
   and remove them from the graph, adjusting or rewriting
   adjacency as needed.
*/
static bool applyReductionRules(Graph& g, std::vector<bool>& fixed, RevertMap& revMap) {
    // We'll iterate over all vertices in ascending order
    // to see if they can be reduced. If so, we fix some set
    // of vertices and remove them from the graph.
    //
    // Return true if at least one rule was applied.
    bool changed = false;

    // We'll keep track of which vertices to remove
    std::vector<int> toRemove;
    toRemove.reserve(g.n);

    for (int v = 0; v < g.n; v++) {
        // Already removed or fixed
        if (g.deg[v] < 0) continue;
        if (fixed[v]) continue;

        if (g.deg[v] == 0) {
            // Weighted-Degree-0: fix v
            fixed[v] = true;
            toRemove.push_back(v);
            changed = true;
        }
        else if (g.deg[v] == 1) {
            // Weighted-Degree-1
            // Let u be the unique neighbor
            if (!g.adj[v].empty()) {
                int u = g.adj[v].front();
                // If w[v] >= w[u], fix u
                if (g.weight[v] >= g.weight[u]) {
                    // Weighted-Degree-1 Rule 2 from the paper
                    fixed[u] = true;
                    toRemove.push_back(u);
                    toRemove.push_back(v);
                    changed = true;
                }
                else {
                    // Weighted-Degree-1 Rule 3
                    // w'(u) = w(u) - w(v)
                    g.weight[u] -= g.weight[v];
                    // Mark v as fixed
                    fixed[v] = true;
                    toRemove.push_back(v);
                    // For revert: revMap.mapping[u] = v
                    // Means: "u's weight was reduced because of v's rule."
                    revMap.mapping[u] = v;
                    changed = true;
                }
            }
        }
        else if (g.deg[v] == 2) {
            // Weighted-Degree-2
            // Suppose N(v) = {u, w}
            if (g.adj[v].size() == 2) {
                int u = g.adj[v][0];
                int w = g.adj[v][1];
                // We only trigger if v and w also share adjacency with u?
                // Actually the paper's condition was:
                // "If G includes v1, v2, and u s.t. N(v1) = {v2,u}, N(v2)={v1,u}..."
                // We look for that pattern:
                // v is analogous to v1, let's see if its neighbor w is analogous to v2.
                // check if deg[u] > 0, etc.
                // We'll see if w is also deg=2 and shares the same neighbor u
                // but we must be sure the adjacency sets match exactly for that pattern.
                if (g.deg[u] >= 0 && g.deg[w] >= 0) {
                    // We check if w is also connected to u and w's adjacency is {v, u}
                    bool pattern = false;
                    if (g.deg[w] == 2) {
                        // Check adjacency of w. Must contain v and u
                        // Also check adjacency of u must contain v and w for the symmetrical condition
                        bool wHasV = false;
                        bool wHasU = false;
                        for (auto& nx : g.adj[w]) {
                            if (nx == v) wHasV = true;
                            if (nx == u) wHasU = true;
                        }
                        if (wHasV && wHasU) {
                            // Now we are in the pattern Weighted-Degree-2
                            pattern = true;
                        }
                    }
                    if (pattern) {
                        // Now compare w[u] with min(w[v], w[w])
                        long long wm = std::min(g.weight[v], g.weight[w]);
                        if (g.weight[u] <= wm) {
                            // Weighted-Degree-2 Rule 4
                            // fix u
                            fixed[u] = true;
                            toRemove.push_back(u);
                            toRemove.push_back(v);
                            toRemove.push_back(w);
                            changed = true;
                        }
                        else {
                            // Weighted-Degree-2 Rule 5
                            // w'(u) = w(u) - min{w(v), w(w)}
                            g.weight[u] -= wm;
                            // Suppose w(v) <= w(w), fix v else fix w
                            if (g.weight[v] <= g.weight[w]) {
                                fixed[v] = true;
                                toRemove.push_back(v);
                            }
                            else {
                                fixed[w] = true;
                                toRemove.push_back(w);
                            }
                            // toRemove both v and w from the graph
                            toRemove.push_back((g.weight[v] <= g.weight[w]) ? w : v);

                            // Also revert mapping, if needed
                            revMap.mapping[u] = (g.weight[v] <= g.weight[w]) ? v : w;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    // Now remove them
    if (changed) {
        // Mark deg[v] = -1 for removed v
        // Also remove them from neighbors
        std::unordered_set<int> srem;
        srem.reserve(toRemove.size());
        for (auto& rv : toRemove) {
            if (rv < 0 || rv >= g.n) continue;
            if (g.deg[rv] < 0) continue;
            srem.insert(rv);
        }
        for (auto& rv : srem) {
            g.deg[rv] = -1;
            for (auto& nx : g.adj[rv]) {
                // remove rv from adjacency of nx
                if (g.deg[nx] >= 0) {
                    // physically remove
                    // we can do a small "erase" trick
                    auto& adjnx = g.adj[nx];
                    auto it = std::remove(adjnx.begin(), adjnx.end(), rv);
                    if (it != adjnx.end()) {
                        adjnx.erase(it, adjnx.end());
                        g.deg[nx]--;
                        g.m--;
                    }
                }
            }
            g.adj[rv].clear();
        }
    }

    return changed;
}

// Helper to revert a solution from the reduced graph if needed
// After we have a solution in the reduced graph, we "unfix" certain
// vertices if their weight was reduced by a neighbor in the original graph.
static void revertSolution(const Graph& g, MWDSContext& ctx, const RevertMap& revMap) {
    // For each vertex u, if revMap.mapping[u] != -1, that means
    // "u's weight was reduced by some vertex v", so we might remove u from solution
    // if we ended up including it, or see that v must be in solution, etc.
    //
    // The paper's Proposition 1 basically states that
    // if u is not in the solution of the reduced problem, we can add v
    // or if u is in the solution, then it's consistent.
    // Implementation-wise, the simplest approach is:
    //   - For each u with mapping[u] != -1, if we ended up not including u,
    //     we must add the mapped vertex. If we included u, we do nothing more.
    //
    // You can carefully track or just do the simpler method: if "u is out => v in solution."
    // The paper’s logic ensures the same cost is retained.

    int n = g.n;
    for (int u = 0; u < n; u++) {
        int mapv = revMap.mapping[u];
        if (mapv >= 0) {
            // If we do not include u, we must include mapv
            if (!ctx.inSolution[u]) {
                ctx.inSolution[mapv] = true;
            }
        }
    }
}

//----------------------------------------------------
// PART 3. ConstructDS
//----------------------------------------------------
//
// 1) Keep applying reduction rules until none applies
// 2) Mark all "fixed" vertices into the candidate solution
//    or store them so they are never removed
// 3) Greedily add vertices until feasible
// 4) Remove redundant vertices
//
static void updateCoverCount(const Graph& g, MWDSContext& ctx) {
    // Reset coverCount to zero
    std::fill(ctx.coverCount.begin(), ctx.coverCount.end(), 0);

    // For each v in solution, mark its neighborhood as covered
    for (int v = 0; v < g.n; v++) {
        if (!ctx.inSolution[v]) continue;
        if (g.deg[v] < 0) continue; // removed in reduction
        // covers v itself
        ctx.coverCount[v]++;
        // covers neighbors
        for (auto& nx : g.adj[v]) {
            ctx.coverCount[nx]++;
        }
    }
}

// Returns true if the entire graph is covered
static bool isFeasible(const Graph& g, MWDSContext& ctx) {
    for (int v = 0; v < g.n; v++) {
        if (g.deg[v] < 0) continue; // removed
        if (ctx.coverCount[v] == 0) {
            return false;
        }
    }
    return true;
}

// Frequency-based score function
// score_f(v) = sum of freq[] of newly dominated vertices / w(v)  if v not in solution
//            = negative sum of freq[] of lost coverage if v is in the solution
static double computeScoreF(const Graph& g, MWDSContext& ctx, int v) {
    if (g.deg[v] < 0) return -1e9; // invalid
    long long sumFreq = 0LL;

    if (!ctx.inSolution[v]) {
        // Potentially adding v
        // Look at all vertices that become covered if we add v
        // i.e. those with coverCount=0 in N[v] plus v itself if coverCount[v]==0
        if (ctx.coverCount[v] == 0) {
            sumFreq += ctx.freq[v];
        }
        for (auto& nx : g.adj[v]) {
            if (ctx.coverCount[nx] == 0) {
                sumFreq += ctx.freq[nx];
            }
        }
        if (g.weight[v] == 0) return 1e9; // if weight is zero after some rule, huge
        return double(sumFreq) / double(g.weight[v]);
    }
    else {
        // Potentially removing v
        // sum frequencies of all vertices that v is the unique dominator for
        // i.e. those with coverCount=1
        for (auto& nx : g.adj[v]) {
            if (ctx.coverCount[nx] == 1) {
                sumFreq += ctx.freq[nx];
            }
        }
        if (ctx.coverCount[v] == 1) {
            sumFreq += ctx.freq[v];
        }
        // negative score
        if (g.weight[v] == 0) {
            // If weight is zero, removing it would cost coverage for free -> be careful
            // We'll treat it as negative but small magnitude to discourage removing it if it is crucial
            return -double(sumFreq) / 1.0;
        }
        return -double(sumFreq) / double(g.weight[v]);
    }
}

// The main constructing procedure
static std::vector<bool> ConstructDS(Graph& g) {
    int n = g.n;
    std::vector<bool> fixed(n, false);
    RevertMap revMap(n);

    // 1) Keep applying reduction until no longer possible
    bool changed = true;
    while (changed) {
        changed = applyReductionRules(g, fixed, revMap);
    }

    // Build an initial feasible solution on the reduced graph
    MWDSContext ctx;
    ctx.inSolution.assign(n, false);
    ctx.freq.assign(n, 1LL);
    ctx.conf.assign(n, 1);
    ctx.age.assign(n, 0LL);
    ctx.coverCount.assign(n, 0);

    // Mark fixed vertices in the solution
    for (int v = 0; v < n; v++) {
        if (fixed[v]) {
            if (g.deg[v] >= 0) {
                ctx.inSolution[v] = true;
            }
        }
    }

    // 2) Add vertices until feasible
    updateCoverCount(g, ctx);
    while (!isFeasible(g, ctx)) {
        // Among all non-dominated vertices, pick the candidate that maximizes score_f
        int bestV = -1;
        double bestScore = -1e9;

        // We only need to consider vertices that are not in the solution
        // and that can cover at least one uncovered vertex
        for (int v = 0; v < n; v++) {
            if (g.deg[v] < 0) continue;
            if (ctx.inSolution[v]) continue;

            // Check if it covers something that is not covered
            // quickly skip if it doesn't help
            if (ctx.coverCount[v] == 0) {
                double sc = computeScoreF(g, ctx, v);
                if (sc > bestScore) { bestScore = sc; bestV = v; }
            }
            else {
                // We also check neighbors
                bool needed = false;
                for (auto& nx : g.adj[v]) {
                    if (ctx.coverCount[nx] == 0) { needed = true; break; }
                }
                if (needed) {
                    double sc = computeScoreF(g, ctx, v);
                    if (sc > bestScore) { bestScore = sc; bestV = v; }
                }
            }
        }
        if (bestV == -1) {
            // if we cannot find any that helps, break or do something
            // but ideally this won't happen if the graph is actually coverable
            break;
        }
        // Add bestV
        ctx.inSolution[bestV] = true;
        // update coverage
        ctx.coverCount[bestV]++;
        for (auto& nx : g.adj[bestV]) {
            ctx.coverCount[nx]++;
        }
    }

    // 3) Shrink (remove redundant)
    // repeatedly remove any vertex from the solution if score_f(v)==0
    bool removedSomething = true;
    while (removedSomething) {
        removedSomething = false;
        // BMS sample size t = 100
        int t = 100;
        std::vector<int> cands;
        cands.reserve(n);

        for (int v = 0; v < n; v++) {
            if (g.deg[v] < 0) continue;
            if (!ctx.inSolution[v]) continue;
            double sc = computeScoreF(g, ctx, v);
            if (std::abs(sc) < 1e-12) {
                cands.push_back(v);
            }
        }
        if (!cands.empty()) {
            // randomly select up to t of them, pick largest weight
            std::shuffle(cands.begin(), cands.end(), rng);
            int pick = -1;
            long long bestW = -1;
            for (int i = 0; i < (int)std::min((size_t)t, cands.size()); i++) {
                int vv = cands[i];
                if (g.weight[vv] > bestW) {
                    bestW = g.weight[vv];
                    pick = vv;
                }
            }
            if (pick >= 0) {
                // remove it
                ctx.inSolution[pick] = false;
                // update coverage
                ctx.coverCount[pick]--;
                for (auto& nx : g.adj[pick]) {
                    ctx.coverCount[nx]--;
                }
                removedSomething = true;
            }
        }
    }

    // Return the final solution for the reduced graph
    revertSolution(g, ctx, revMap);
    return ctx.inSolution;
}

//----------------------------------------------------
// PART 4. CC2V3+ (Updating Rules + Using Rule)
//----------------------------------------------------
static void initConfiguration(MWDSContext& ctx, int n) {
    for (int i = 0; i < n; i++) {
        ctx.conf[i] = 1; // all allowed initially
        ctx.age[i] = 0;
    }
}

static void onAddVertex(const Graph& g, MWDSContext& ctx, int v) {
    // If we add v, for each vertex in N1(v):
    //   if conf[u1] in {0,2}, set conf[u1]=1
    // For each in N2(v) with conf=0, set conf=1
    for (auto& u1 : g.adj[v]) {
        if (ctx.conf[u1] == 0 || ctx.conf[u1] == 2) {
            ctx.conf[u1] = 1;
        }
        // For neighbors of u1, if conf[x]==0, set conf[x]=1 only if x is in N2(v)?
        // More directly: We'll do a second-level pass for them, but we can do it more systematically:
    }
    // For second-level neighbors
    for (auto& u1 : g.adj[v]) {
        for (auto& u2 : g.adj[u1]) {
            if (ctx.conf[u2] == 0) {
                ctx.conf[u2] = 1;
            }
        }
    }
}

static void onRemoveVertex(const Graph& g, MWDSContext& ctx, int v, const std::vector<int>& set1) {
    // set1 contains those that become non-dominated because of removing v
    // for each x in set1, for each y in N1[x], conf[y]=2
    for (auto& x : set1) {
        for (auto& y : g.adj[x]) {
            if (!ctx.inSolution[y]) {
                ctx.conf[y] = 2;
            }
        }
    }
    // for vertices in N2(v) \ union(N1[x]), if conf[u]==0, set conf[u]=1
    // We'll gather N2(v)
    std::unordered_set<int> n2;
    n2.insert(v);
    for (auto& x : g.adj[v]) {
        n2.insert(x);
    }
    // expand to neighbors
    std::vector<int> tmp;
    for (auto& x : g.adj[v]) {
        for (auto& nx : g.adj[x]) {
            n2.insert(nx);
        }
    }
    // union of N1[x] for x in set1
    std::unordered_set<int> skip;
    for (auto& x : set1) {
        for (auto& nx : g.adj[x]) {
            skip.insert(nx);
        }
    }
    // Now for each u in n2 - skip
    for (auto& u : n2) {
        if (skip.count(u) == 0) {
            if (ctx.conf[u] == 0) {
                ctx.conf[u] = 1;
            }
        }
    }
    // finally, conf[v] = 0
    ctx.conf[v] = 0;
}

//----------------------------------------------------
// PART 5. Local Search + Helper Routines
//----------------------------------------------------

// update coverage count when we add a vertex v
static void addToSolution(const Graph& g, MWDSContext& ctx, int v) {
    ctx.inSolution[v] = true;
    ctx.coverCount[v]++;
    for (auto& nx : g.adj[v]) {
        ctx.coverCount[nx]++;
    }
}

// update coverage count when we remove v
// return the set of vertices that become non-dominated
static std::vector<int> removeFromSolution(const Graph& g, MWDSContext& ctx, int v) {
    ctx.inSolution[v] = false;
    // track who gets coverCount=0 after removal
    std::vector<int> lost;
    // first coverCount[v]--
    ctx.coverCount[v]--;
    if (ctx.coverCount[v] == 0) {
        lost.push_back(v);
    }
    // then neighbors
    for (auto& nx : g.adj[v]) {
        ctx.coverCount[nx]--;
        if (ctx.coverCount[nx] == 0) {
            lost.push_back(nx);
        }
    }
    return lost;
}

// check if entire graph is dominated
static bool isCompleteCover(const Graph& g, MWDSContext& ctx) {
    for (int v = 0; v < g.n; v++) {
        if (g.deg[v] < 0) continue;
        if (ctx.coverCount[v] == 0) return false;
    }
    return true;
}

// frequency update: after each iteration, freq[w]++
static void updateFrequency(const Graph& g, MWDSContext& ctx) {
    for (int v = 0; v < g.n; v++) {
        if (g.deg[v] < 0) continue;
        if (ctx.coverCount[v] == 0) {
            ctx.freq[v]++;
        }
    }
}

// BMS selection of a vertex among “candCount” with largest “scoreF”
static int pickBestAmongSample(const Graph& g, MWDSContext& ctx, const std::vector<int>& cands, int sampleSize) {
    if (cands.empty()) return -1;
    if ((int)cands.size() <= sampleSize) {
        // pick best in entire cands
        double bestScore = -1e9;
        int pick = -1;
        for (auto& v : cands) {
            double sc = computeScoreF(g, ctx, v);
            if (sc > bestScore) {
                bestScore = sc;
                pick = v;
            }
            else if (std::fabs(sc - bestScore) < 1e-14) {
                // tie break on configuration or age
                // prefer higher conf => prefer older
                if (ctx.conf[v] > ctx.conf[pick]) {
                    pick = v;
                }
                else if (ctx.conf[v] == ctx.conf[pick]) {
                    // older
                    if (ctx.age[v] > ctx.age[pick]) {
                        pick = v;
                    }
                }
            }
        }
        return pick;
    }
    else {
        // randomly pick "sampleSize" of them
        // then find best among those
        std::vector<int> sample = cands;
        std::shuffle(sample.begin(), sample.end(), rng);
        sample.resize(sampleSize);
        double bestScore = -1e9;
        int pick = -1;
        for (auto& v : sample) {
            double sc = computeScoreF(g, ctx, v);
            if (sc > bestScore) {
                bestScore = sc;
                pick = v;
            }
            else if (std::fabs(sc - bestScore) < 1e-14) {
                if (ctx.conf[v] > ctx.conf[pick]) {
                    pick = v;
                }
                else if (ctx.conf[v] == ctx.conf[pick]) {
                    if (ctx.age[v] > ctx.age[pick]) {
                        pick = v;
                    }
                }
            }
        }
        return pick;
    }
}

// A small utility to pick a random undominated vertex
static int pickRandomUndominated(const Graph& g, MWDSContext& ctx) {
    // We'll gather all v where coverCount[v]==0
    std::vector<int> und;
    und.reserve(g.n);
    for (int v = 0; v < g.n; v++) {
        if (g.deg[v] < 0) continue;
        if (ctx.coverCount[v] == 0) {
            und.push_back(v);
        }
    }
    if (und.empty()) return -1;
    // pick random
    std::uniform_int_distribution<int> dist(0, (int)und.size() - 1);
    return und[dist(rng)];
}

// “Adding rule” from CC2V3+ with frequency-based tie-break
//   Among neighbors of the chosen uncovered vertex v, pick the one that’s
//   feasible to add (conf[w] != 0) with highest scoreF, break ties on conf and age.
static int pickVertexToAdd(const Graph& g, MWDSContext& ctx, int uncovered) {
    // Among N(uncovered), pick best
    int pick = -1;
    double bestScore = -1e9;
    for (auto& nx : g.adj[uncovered]) {
        if (ctx.inSolution[nx]) continue; // skip
        if (ctx.conf[nx] == 0) {
            // The special case: if there's only 1 uncovered vertex in entire graph, we ignore conf=0
            // as described in the paper
            // We can do that check:
            //   if there’s exactly 1 undominated, break conf=0
            continue;
        }
        double sc = computeScoreF(g, ctx, nx);
        if (sc > bestScore) {
            bestScore = sc;
            pick = nx;
        }
        else if (std::fabs(sc - bestScore) < 1e-14) {
            // tie break on conf => age
            if (ctx.conf[nx] > ctx.conf[pick]) {
                pick = nx;
            }
            else if (ctx.conf[nx] == ctx.conf[pick]) {
                if (ctx.age[nx] > ctx.age[pick]) {
                    pick = nx;
                }
            }
        }
    }
    // If pick==-1 and there's exactly 1 uncovered vertex in entire graph => we can pick a neighbor ignoring conf=0
    if (pick == -1) {
        // check how many total uncovered
        int countUnd = 0;
        for (int i = 0; i < g.n; i++) {
            if (g.deg[i] < 0) continue;
            if (ctx.coverCount[i] == 0) countUnd++;
            if (countUnd > 1) break;
        }
        if (countUnd == 1) {
            // pick among neighbors ignoring conf
            for (auto& nx : g.adj[uncovered]) {
                if (!ctx.inSolution[nx]) {
                    double sc = computeScoreF(g, ctx, nx);
                    if (sc > bestScore) {
                        bestScore = sc;
                        pick = nx;
                    }
                }
            }
        }
    }
    return pick;
}

// compute the total cost
static long long computeSolutionCost(const Graph& g, MWDSContext& ctx) {
    long long cost = 0;
    for (int v = 0; v < g.n; v++) {
        if (g.deg[v] < 0) continue;
        if (ctx.inSolution[v]) {
            cost += g.weight[v];
        }
    }
    return cost;
}

//----------------------------------------------------
// PART 6. DeepOpt (Perturbation) for MWDS
//----------------------------------------------------
static void deepOptMWDS(const Graph& g, MWDSContext& ctx,
    long long& bestCost,
    std::vector<bool>& bestSol,
    int Rcount, int InnerDepth)
{
    // 1) Removing Phase
    //   remove Rcount times: pick a random in-solution vertex v, remove all in N2[v] ∩ solution
    //   create locked set Dlocked => those remain in solution
    //   create an "unlocked" set => empty initially
    //
    // 2) Search Phase (InnerDepth steps):
    //   repeatedly do:
    //     - if not feasible, add needed vertices
    //     - remove 2 vertices from "unlocked"
    //   only vertices in "unlocked" can be removed, while "locked" ones remain always
    //
    // 3) Unlock them
    //   final new solution is Dlocked ∪ unlocked
    //   done

    int n = g.n;
    std::vector<bool> locked(n, false);
    // removing phase
    std::uniform_int_distribution<int> distV(0, n - 1);
    int times = 1 + (int)(distV(rng) % Rcount); // random in [1, Rcount]

    for (int i = 0; i < times; i++) {
        // pick a random in-solution vertex
        std::vector<int> solVertices;
        solVertices.reserve(n);
        for (int v = 0; v < n; v++) {
            if (g.deg[v] < 0) continue;
            if (ctx.inSolution[v]) {
                solVertices.push_back(v);
            }
        }
        if (solVertices.empty()) break;
        std::shuffle(solVertices.begin(), solVertices.end(), rng);
        int chosen = solVertices.front();
        // remove N2[chosen] ∩ solution
        // gather N2[chosen]
        std::vector<int> n2;
        n2.push_back(chosen);
        for (auto& nx : g.adj[chosen]) {
            n2.push_back(nx);
        }
        for (auto& nx : g.adj[chosen]) {
            for (auto& nx2 : g.adj[nx]) {
                n2.push_back(nx2);
            }
        }
        // remove them
        std::unordered_set<int> sN2(n2.begin(), n2.end());
        for (auto& vx : sN2) {
            if (vx >= 0 && vx < n && ctx.inSolution[vx]) {
                // remove
                auto lost = removeFromSolution(g, ctx, vx);
                onRemoveVertex(g, ctx, vx, lost);
            }
        }
    }
    // lock the rest
    for (int v = 0; v < n; v++) {
        if (g.deg[v] < 0) continue;
        if (ctx.inSolution[v]) {
            locked[v] = true;
        }
    }

    // 2) search phase
    int in_nonimpr = 0;
    long long currCost = computeSolutionCost(g, ctx);
    while (in_nonimpr < InnerDepth) {
        // ensure feasibility
        while (!isCompleteCover(g, ctx)) {
            int uv = pickRandomUndominated(g, ctx);
            if (uv < 0) break; // should not happen if still feasible
            // pick a neighbor to add
            int addv = pickVertexToAdd(g, ctx, uv);
            if (addv < 0) break;
            // add it
            addToSolution(g, ctx, addv);
            // freq update for coverage
            updateFrequency(g, ctx);

            // remove redundant in the unlocked set
            // i.e. any vertex inSolution but not locked and scoreF==0
            bool removedFlag = true;
            while (removedFlag) {
                removedFlag = false;
                for (int v = 0; v < n; v++) {
                    if (g.deg[v] < 0) continue;
                    if (!ctx.inSolution[v]) continue;
                    if (locked[v]) continue;
                    double sc = computeScoreF(g, ctx, v);
                    if (std::fabs(sc) < 1e-14) {
                        auto lost = removeFromSolution(g, ctx, v);
                        onRemoveVertex(g, ctx, v, lost);
                        removedFlag = true;
                        break;
                    }
                }
            }
        }
        // check improvement
        currCost = computeSolutionCost(g, ctx);
        if (currCost < bestCost) {
            bestCost = currCost;
            bestSol = ctx.inSolution;
            in_nonimpr = 0;
        }
        else {
            in_nonimpr++;
        }

        // random remove in unlocked
        // pick best to remove
        // pick random to remove
        std::vector<int> unlockedSol;
        for (int v = 0; v < n; v++) {
            if (g.deg[v] < 0) continue;
            if (ctx.inSolution[v] && !locked[v]) {
                unlockedSol.push_back(v);
            }
        }
        if (unlockedSol.size() < 2) continue;

        // pick best from unlocked
        int t = 100;
        std::shuffle(unlockedSol.begin(), unlockedSol.end(), rng);
        if ((int)unlockedSol.size() > t) unlockedSol.resize(t);
        double bestSc = -1e9;
        int bestV = -1;
        for (auto& vx : unlockedSol) {
            double sc = computeScoreF(g, ctx, vx);
            if (sc > bestSc) { bestSc = sc; bestV = vx; }
        }
        if (bestV >= 0) {
            auto lost = removeFromSolution(g, ctx, bestV);
            onRemoveVertex(g, ctx, bestV, lost);
        }
        // pick random from unlocked again
        unlockedSol.clear();
        for (int v = 0; v < n; v++) {
            if (g.deg[v] < 0) continue;
            if (ctx.inSolution[v] && !locked[v]) {
                unlockedSol.push_back(v);
            }
        }
        if (!unlockedSol.empty()) {
            std::shuffle(unlockedSol.begin(), unlockedSol.end(), rng);
            int rpick = unlockedSol.front();
            auto lost2 = removeFromSolution(g, ctx, rpick);
            onRemoveVertex(g, ctx, rpick, lost2);
        }
    }

    // unlock all
    for (int v = 0; v < n; v++) {
        locked[v] = false;
    }
}

//----------------------------------------------------
// PART 7. The Main Local Search (DeepOpt-MWDS)
//----------------------------------------------------
static std::vector<bool> deepOptMWDS_solve(Graph& g, double cutoffTime) {
    // 1) Construct initial solution
    std::vector<bool> initSol = ConstructDS(g);

    // create context
    MWDSContext ctx;
    ctx.inSolution = initSol;
    ctx.freq.assign(g.n, 1LL);
    ctx.conf.assign(g.n, 1);
    ctx.age.assign(g.n, 0LL);
    ctx.coverCount.assign(g.n, 0);

    // Setup coverage
    updateCoverCount(g, ctx);

    // best solution
    std::vector<bool> bestSol = ctx.inSolution;
    long long bestCost = 0LL;
    for (int v = 0; v < g.n; v++) {
        if (ctx.inSolution[v] && g.deg[v] >= 0) bestCost += g.weight[v];
    }

    // time measurement
    auto start = std::chrono::steady_clock::now();

    // We define an outer loop that tries removing or adding
    // The paper uses "out_nonimpr" to track how many steps without improvement
    // Then calls deepOptMWDS if that hits OutterDepth
    long long out_nonimpr = 0;
    long long OutterDepth = 5000; // for example

    // We define a "gap" line
    // if the density is large, we set gap to average weight; else to max weight
    double density = (double)g.m / (double)std::max(1, g.n);
    long long sumWeightSol = 0LL;
    int solCount = 0;
    for (int v = 0; v < g.n; v++) {
        if (ctx.inSolution[v] && g.deg[v] >= 0) { sumWeightSol += g.weight[v]; solCount++; }
    }
    long long gap = (solCount > 0) ? sumWeightSol / solCount : 1;
    if (density < 0.07) {
        // use max weight
        gap = 0;
        for (int v = 0; v < g.n; v++) {
            if (ctx.inSolution[v] && g.deg[v] >= 0) {
                gap = std::max(gap, g.weight[v]);
            }
        }
    }

    int iteration = 0;
    const int t = 100;  // BMS sample size

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsedSec = std::chrono::duration<double>(now - start).count();
        if (elapsedSec >= cutoffTime) break; // stop

        iteration++;

        // 1) If D is feasible and we find vertices with score==0, remove them
        if (isCompleteCover(g, ctx)) {
            // remove all with scoreF=0 by BMS picking largest weight
            bool removedFlag = true;
            while (removedFlag) {
                removedFlag = false;
                std::vector<int> zeroCands;
                for (int v = 0; v < g.n; v++) {
                    if (g.deg[v] < 0) continue;
                    if (ctx.inSolution[v]) {
                        double sc = computeScoreF(g, ctx, v);
                        if (std::fabs(sc) < 1e-14) {
                            zeroCands.push_back(v);
                        }
                    }
                }
                if (!zeroCands.empty()) {
                    int pick = -1;
                    long long bestW = -1LL;
                    std::shuffle(zeroCands.begin(), zeroCands.end(), rng);
                    int range = std::min(t, (int)zeroCands.size());
                    for (int i = 0; i < range; i++) {
                        int vv = zeroCands[i];
                        if (g.weight[vv] > bestW) {
                            bestW = g.weight[vv];
                            pick = vv;
                        }
                    }
                    if (pick >= 0) {
                        auto lost = removeFromSolution(g, ctx, pick);
                        onRemoveVertex(g, ctx, pick, lost);
                        removedFlag = true;
                    }
                }
            }

            // check best
            long long currCost = computeSolutionCost(g, ctx);
            if (currCost < bestCost) {
                bestCost = currCost;
                bestSol = ctx.inSolution;
                out_nonimpr = 0;
            }
        }

        // 2) If w(D)+gap >= w(D*), remove from D until w(D)+gap < w(D*)
        {
            long long currCost = computeSolutionCost(g, ctx);
            while (currCost + gap >= bestCost) {
                // remove best among the solution by BMS
                std::vector<int> cands;
                for (int v = 0; v < g.n; v++) {
                    if (ctx.inSolution[v] && g.deg[v] >= 0) {
                        cands.push_back(v);
                    }
                }
                if (cands.empty()) break;
                int pick = pickBestAmongSample(g, ctx, cands, t);
                if (pick < 0) break;
                auto lost = removeFromSolution(g, ctx, pick);
                onRemoveVertex(g, ctx, pick, lost);
                currCost = computeSolutionCost(g, ctx);
            }
        }

        // 3) Add steps to dominate newly uncovered
        if (!isCompleteCover(g, ctx)) {
            int uv = pickRandomUndominated(g, ctx);
            if (uv < 0) continue;
            int av = pickVertexToAdd(g, ctx, uv);
            if (av < 0) continue;

            long long cCost = computeSolutionCost(g, ctx);
            if (cCost + g.weight[av] + gap <= bestCost) {
                // direct add
                addToSolution(g, ctx, av);
            }
            else {
                // try exchanging with BMS
                // pick an in-solution vertex with biggest score
                std::vector<int> ins;
                for (int v = 0; v < g.n; v++) {
                    if (ctx.inSolution[v] && g.deg[v] >= 0) {
                        ins.push_back(v);
                    }
                }
                int pick = pickBestAmongSample(g, ctx, ins, t);
                if (pick < 0) {
                    // random accept av with probability 1/|UndomdSet(D)|
                    int cUnd = 0;
                    for (int v = 0; v < g.n; v++) {
                        if (g.deg[v] < 0) continue;
                        if (ctx.coverCount[v] == 0) cUnd++;
                    }
                    if (cUnd <= 1) {
                        addToSolution(g, ctx, av);
                    }
                    else {
                        std::uniform_real_distribution<double> d01(0.0, 1.0);
                        double p = d01(rng);
                        if (p <= 1.0 / std::max(1, cUnd)) {
                            addToSolution(g, ctx, av);
                        }
                    }
                }
                else {
                    // see if exchanging helps
                    double scAv = computeScoreF(g, ctx, av);
                    double scPick = computeScoreF(g, ctx, pick);
                    if (scAv + scPick > 0.0) {
                        // remove pick, add av
                        auto lost = removeFromSolution(g, ctx, pick);
                        onRemoveVertex(g, ctx, pick, lost);
                        addToSolution(g, ctx, av);
                    }
                    else {
                        // random accept av with probability 1/|UndomdSet(D)|
                        int cUnd = 0;
                        for (int v = 0; v < g.n; v++) {
                            if (g.deg[v] < 0) continue;
                            if (ctx.coverCount[v] == 0) cUnd++;
                        }
                        if (cUnd <= 1) {
                            addToSolution(g, ctx, av);
                        }
                        else {
                            std::uniform_real_distribution<double> d01(0.0, 1.0);
                            double p = d01(rng);
                            if (p <= 1.0 / std::max(1, cUnd)) {
                                addToSolution(g, ctx, av);
                            }
                        }
                    }
                }
            }
        }

        // freq update
        updateFrequency(g, ctx);

        // improvement check
        long long currCost = computeSolutionCost(g, ctx);
        if (currCost < bestCost) {
            bestCost = currCost;
            bestSol = ctx.inSolution;
            out_nonimpr = 0;
        }
        else {
            out_nonimpr++;
        }

        // 4) If out_nonimpr hits OutterDepth, do deepOptMWDS
        if (out_nonimpr >= OutterDepth) {
            deepOptMWDS(g, ctx, bestCost, bestSol, /*Rcount=*/5, /*InnerDepth=*/1000);
            out_nonimpr = 0;
        }

        // (Optional) re-check gap
        // e.g. if solution changed significantly. We skip for brevity
    }

    return bestSol;
}

//----------------------------------------------------
// PART 8. Example Main
//----------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_file> <cutoff>\n";
        return 1;
    }
    std::string graphFile = argv[1];
    double cutoff = atof(argv[2]);

    // 1) Read graph
    // Suppose the input format is:
    // n m
    // w(0)
    // w(1)
    // ...
    // w(n-1)
    // then m lines of edges: (u v)
    // 0-based indexing
    //
    // Adjust this reading logic as needed.
    std::ifstream fin(graphFile);
    if (!fin) {
        std::cerr << "Cannot open file: " << graphFile << "\n";
        return 1;
    }
    int n, m;
    fin >> n >> m;
    Graph g(n);
    g.m = m;
    for (int i = 0; i < n; i++) {
        fin >> g.weight[i];
    }
    for (int i = 0; i < m; i++) {
        int u, v;
        fin >> u >> v;
        if (u < 0 || u >= n || v < 0 || v >= n) continue; // skip invalid
        g.adj[u].push_back(v);
        g.adj[v].push_back(u);
        g.deg[u]++;
        g.deg[v]++;
    }
    fin.close();

    // 2) Solve
    std::vector<bool> bestSol = deepOptMWDS_solve(g, cutoff);

    // 3) Output results
    // compute final cost
    long long cost = 0LL;
    for (int v = 0; v < n; v++) {
        if (v < (int)bestSol.size() && bestSol[v]) {
            cost += g.weight[v];
        }
    }
    std::cout << "Best solution cost = " << cost << "\n";
    // Optionally list which vertices are chosen
    for (int v = 0; v < n; v++) {
        if (bestSol[v]) std::cout << v << " ";
    }
    std::cout << "\n";
    return 0;
}
