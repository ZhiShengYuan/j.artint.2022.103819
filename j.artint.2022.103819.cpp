// main-cpp-gemini-env-debug.cpp
/****************************************************
 * DeepOpt-MWDS
 * MWDS 问题的一个 C++ 实现，基于论文:
 * "Improved local search for the minimum weight dominating
 * set problem in massive graphs by using a deep optimization
 * mechanism" (Artificial Intelligence 314 (2023) 103819)
 *
 * 添加了基于环境变量 MWDS_DEBUG 的调试输出控制。
 * 设置 MWDS_DEBUG=1 来启用调试输出。
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
 #include <thread> // For std::thread
 #include <mutex>  // For std::mutex
 #include <atomic> // For std::atomic
 #include <future> // For std::async, std::future
 #include <string>
 #include <sstream>
 #include <stdexcept> // For std::runtime_error
 #include <iomanip> // For std::setprecision
 #include <cstdlib> // For std::getenv
 #include <cstring> // For std::strcmp
 #include <numeric> // For std::iota
 #include <filesystem>
 #include <locale>   // Add for tolower
 #include <iomanip> 
 #include <ctime> 
 static bool isDebugEnabled = false;

 //----------------------------------------------------
 // PART 1. 数据结构
 //----------------------------------------------------

struct Graph {
     int n;
     long long m;
     std::vector<long long> weight; // Current weights (can be modified by reduction)
     std::vector<std::vector<int>> adj; // Adjacency list for N(v)
     std::vector<std::vector<int>> N_v_closed; // Closed neighborhood N[v] = N(v) U {v}
     std::vector<long long> original_weight;
     std::vector<int> deg; // Current degree (after reduction)
     std::vector<int> original_deg;
     std::vector<int> revert_map; // revert_map[u] = v means weight of u was reduced due to fixing v

     Graph(int n_ = 0) : n(n_), m(0) {
         adj.resize(n_);
         N_v_closed.resize(n_);
         weight.resize(n_, 1LL);
         original_weight.resize(n_, 1LL);
         deg.resize(n_, 0);
         original_deg.resize(n_, 0);
         revert_map.assign(n_, -1);
     }

     void addEdge(int u, int v) {
         if (u < 0 || u >= n || v < 0 || v >= n || u == v) {
             if (isDebugEnabled) { std::cerr << "警告: 跳过无效边 (" << u << "," << v << ") during addEdge\n"; }
             return;
         }
         adj[u].push_back(v);
         adj[v].push_back(u);
     }

     void finalize() {
         long long actual_edge_count = 0;
         for (int i = 0; i < n; ++i) {
             if (!adj[i].empty()) {
                 std::sort(adj[i].begin(), adj[i].end());
                 adj[i].erase(std::unique(adj[i].begin(), adj[i].end()), adj[i].end());
             }
             deg[i] = adj[i].size();
             original_deg[i] = deg[i];
             actual_edge_count += deg[i];

             // Populate N_v_closed
             N_v_closed[i] = adj[i];
             N_v_closed[i].push_back(i); // Add self
             std::sort(N_v_closed[i].begin(), N_v_closed[i].end());
             N_v_closed[i].erase(std::unique(N_v_closed[i].begin(), N_v_closed[i].end()), N_v_closed[i].end());
         }
         m = actual_edge_count / 2;
     }
 };


 struct MWDSContext {
     std::vector<bool> inSolution;
     std::vector<long long> current_vertex_scores;
     std::vector<long long> freq_or_penalty_weight;
     std::vector<int> conf;
     std::vector<long long> age;
     std::vector<int> coverCount;

     std::vector<int> undominated_vertices_list;
     std::vector<bool> is_undominated;
     std::vector<int> redundant_vertices_list;
     std::vector<bool> is_redundant;

     std::mt19937_64 local_rng;

     MWDSContext(int n, unsigned int seed) :
         inSolution(n, false),
         current_vertex_scores(n, 0LL),
         freq_or_penalty_weight(n, 1LL),
         conf(n, 1),
         age(n, 0LL),
         coverCount(n, 0),
         is_undominated(n, false),
         is_redundant(n, false),
         local_rng(seed) {
             undominated_vertices_list.reserve(n);
             redundant_vertices_list.reserve(n);
         }
     MWDSContext() = default;
 };


// --- MODIFICATION: Declare global_rng unseeded ---
// static std::mt19937_64 global_rng(std::chrono::steady_clock::now().time_since_epoch().count()); // OLD
static std::mt19937_64 global_rng; // NEW

static std::mutex best_sol_mutex;
static std::vector<bool> globalBestSol;
static std::atomic<long long> globalBestCost(std::numeric_limits<long long>::max());


// Enum for Graph Format
 enum class GraphFormat {
     AUTO_DETECT,
     WEIGHTED_LIST, 
     DIMACS_CLQ     
 };


 //----------------------------------------------------
 // PART 0: MAXHEAP IMPLEMENTATION (Adapted from author's main.cpp)
 //----------------------------------------------------
 static const Graph* graph_ptr_for_heap_ = nullptr;
 static const MWDSContext* context_ptr_for_heap_ = nullptr;
std::string escape_json_string(const std::string& s) {
    std::stringstream ss;
    for (char c : s) {
        switch (c) {
            case '"':  ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b";  break;
            case '\f': ss << "\\f";  break;
            case '\n': ss << "\\n";  break;
            case '\r': ss << "\\r";  break;
            case '\t': ss << "\\t";  break;
            default:
                if ('\x00' <= c && c <= '\x1f') {
                    ss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    ss << c;
                }
        }
    }
    return ss.str();
}
 // Helper to get score/weight ratio for PQ or comparisons (used by MAXHEAP)
 static double getScorePerWeight_for_heap(int v) {
    if (!graph_ptr_for_heap_ || !context_ptr_for_heap_ ||
        v < 0 || v >= graph_ptr_for_heap_->n || 
        v >= graph_ptr_for_heap_->weight.size() || graph_ptr_for_heap_->weight[v] <= 0 ||
        v >= context_ptr_for_heap_->current_vertex_scores.size()) {
        return -std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(context_ptr_for_heap_->current_vertex_scores[v]) / static_cast<double>(graph_ptr_for_heap_->weight[v]);
 }


 class MAXHEAP
 {
 public:
     MAXHEAP(int capacity) : capacity_(capacity), size_(0) {
         arr_ = new int[capacity_]; // Stores vertex indices
         pos_ = new int[capacity_]; // pos_[vertex_idx] = heap_array_idx
                                    // Assumes vertex indices are 0 to capacity-1
         std::fill(pos_, pos_ + capacity_, -1);
     }
     ~MAXHEAP() {
         delete[] arr_;
         delete[] pos_;
     }

     void swap(int heap_idx1, int heap_idx2) {
         int temp_v = arr_[heap_idx1];
         arr_[heap_idx1] = arr_[heap_idx2];
         pos_[arr_[heap_idx1]] = heap_idx1;
         arr_[heap_idx2] = temp_v;
         pos_[arr_[heap_idx2]] = heap_idx2;
     }

     bool isLeaf(int heap_idx) {
         return (heap_idx >= size_ / 2) && (heap_idx < size_);
     }

     int getLeftChild(int heap_idx) {
         return (2 * heap_idx + 1);
     }

     int getRightChild(int heap_idx) {
         return (2 * heap_idx + 2);
     }

     int getParent(int heap_idx) {
         return (heap_idx - 1) / 2;
     }

     // Compares v1 and v2 based on their score/weight ratio
     // Returns > 0 if v1 is better, < 0 if v2 is better, 0 if equal
     int compare_vertices(int v1, int v2) {
         double r1 = getScorePerWeight_for_heap(v1);
         double r2 = getScorePerWeight_for_heap(v2);
         if (r1 > r2) return 1;
         if (r1 < r2) return -1;
         // Tie-breaking: smaller original weight is better (consistent with some greedy approaches)
         // Or prefer older vertices (larger age) as in some parts of the user's logic
         if (v1 < graph_ptr_for_heap_->original_weight.size() && v2 < graph_ptr_for_heap_->original_weight.size() &&
             graph_ptr_for_heap_->original_weight[v1] != graph_ptr_for_heap_->original_weight[v2]) {
             return (graph_ptr_for_heap_->original_weight[v1] < graph_ptr_for_heap_->original_weight[v2]) ? 1 : -1;
         }
         if (v1 < context_ptr_for_heap_->age.size() && v2 < context_ptr_for_heap_->age.size() &&
             context_ptr_for_heap_->age[v1] != context_ptr_for_heap_->age[v2]) {
             return (context_ptr_for_heap_->age[v1] > context_ptr_for_heap_->age[v2]) ? 1 : -1;
         }
         return 0; // Default to no preference or could use vertex index
     }

     void adjust(int heap_idx) { // Sift-down
         int largest_child_idx;
         while (!isLeaf(heap_idx)) {
             largest_child_idx = getLeftChild(heap_idx);
             int right_child_idx = getRightChild(heap_idx);

             if ((right_child_idx < size_) && 
                 compare_vertices(arr_[right_child_idx], arr_[largest_child_idx]) > 0) {
                 largest_child_idx = right_child_idx;
             }

             if (compare_vertices(arr_[heap_idx], arr_[largest_child_idx]) >= 0) { // Parent is already better or equal
                 return;
             }
             swap(heap_idx, largest_child_idx);
             heap_idx = largest_child_idx; // Move down to the swapped child
         }
     }

     void insert(int vertex_idx) {
         if (pos_[vertex_idx] != -1) { // Already in heap
             if (isDebugEnabled && vertex_idx < context_ptr_for_heap_->current_vertex_scores.size() && context_ptr_for_heap_->current_vertex_scores[vertex_idx] <=0 ) {
                 // This might happen if we call changeVal on a vertex that should be removed
                 // For direct insert, we expect score > 0.
             }
            // If insert is called on an existing item, it's like a changeVal.
            // For safety, one might remove then re-insert, or just call adjust up/down.
            // The changeVal function handles this correctly.
            // Here, we assume insert is for new items.
            // If it's a bug, it means it should have been changeVal or remove first.
             return;
         }
         if (size_ >= capacity_) {
             if (isDebugEnabled) std::cerr << "Error: MAXHEAP capacity reached during insert." << std::endl;
             return; // Or throw exception
         }

         int current_heap_idx = size_++;
         arr_[current_heap_idx] = vertex_idx;
         pos_[vertex_idx] = current_heap_idx;

         while (current_heap_idx != 0 && 
                compare_vertices(arr_[current_heap_idx], arr_[getParent(current_heap_idx)]) > 0) {
             swap(current_heap_idx, getParent(current_heap_idx));
             current_heap_idx = getParent(current_heap_idx);
         }
     }

     int removeRoot() {
         if (size_ == 0) {
             if (isDebugEnabled) std::cerr << "Error: MAXHEAP empty during removeRoot." << std::endl;
             return -1; // Or throw
         }
         int root_vertex = arr_[0];
         pos_[root_vertex] = -1; // Mark as removed

         arr_[0] = arr_[--size_];
         if (size_ > 0) { // If heap is not empty after removal
             pos_[arr_[0]] = 0;
             adjust(0);
         }
         return root_vertex;
     }
     
     // Removes the vertex 'vertex_idx' from the heap.
     // This is different from removeAtHeapIdx.
     void removeVertex(int vertex_idx) {
        if (vertex_idx < 0 || vertex_idx >= capacity_ || pos_[vertex_idx] == -1) return; // Not in heap or invalid
        
        int heap_idx_to_remove = pos_[vertex_idx];
        pos_[vertex_idx] = -1; // Mark as removed from pos_ immediately

        if (heap_idx_to_remove == size_ -1) { // If it's the last element
            size_--;
            return;
        }

        arr_[heap_idx_to_remove] = arr_[--size_]; // Replace with last element
        if (size_ > 0 && heap_idx_to_remove < size_) { // Check bounds if heap_idx_to_remove was valid before size change
             pos_[arr_[heap_idx_to_remove]] = heap_idx_to_remove;

            // Try to sift up first
            int current_idx = heap_idx_to_remove;
            while(current_idx != 0 && compare_vertices(arr_[current_idx], arr_[getParent(current_idx)]) > 0) {
                swap(current_idx, getParent(current_idx));
                current_idx = getParent(current_idx);
            }
            // If sift up didn't move it, it might need to sift down
            if (current_idx == heap_idx_to_remove) {
                 adjust(heap_idx_to_remove);
            }
        }
     }


     int getRoot() {
         if (size_ == 0) return -1;
         return arr_[0];
     }

     int getSize() {
         return size_;
     }
     
     bool isInHeap(int vertex_idx) const {
         if (vertex_idx < 0 || vertex_idx >= capacity_) return false;
         return pos_[vertex_idx] != -1;
     }

     void changeVal(int vertex_idx) {
         if (vertex_idx < 0 || vertex_idx >= capacity_ ) return;

         if (pos_[vertex_idx] == -1) { // Not in heap, try to insert if score is positive
             if (vertex_idx < context_ptr_for_heap_->current_vertex_scores.size() && context_ptr_for_heap_->current_vertex_scores[vertex_idx] > 0) {
                insert(vertex_idx);
             }
             return;
         }
         
         // If score becomes non-positive, it should be removed
         if (vertex_idx < context_ptr_for_heap_->current_vertex_scores.size() && context_ptr_for_heap_->current_vertex_scores[vertex_idx] <= 0) {
             removeVertex(vertex_idx);
             return;
         }

         // Otherwise, its score changed, re-heapify at its position
         int heap_idx = pos_[vertex_idx];
         // Try sifting up first
         int current_idx = heap_idx;
         while(current_idx != 0 && compare_vertices(arr_[current_idx], arr_[getParent(current_idx)]) > 0) {
             swap(current_idx, getParent(current_idx));
             current_idx = getParent(current_idx);
         }
         // If sift up didn't move it, it might need to sift down (its score might have decreased)
         if (current_idx == heap_idx) {
              adjust(heap_idx);
         }
     }

     void clear() {
        size_ = 0;
        // For pos_, it needs to be reset for all potential vertex indices if they were ever in heap
        // However, if capacity_ is the max vertex_id + 1, this is fine.
        std::fill(pos_, pos_ + capacity_, -1);
     }


 private:
     int* arr_;    // Heap array storing vertex indices
     int* pos_;    // pos_[vertex_idx] stores its index in arr_
     int size_;
     int capacity_; // Max number of elements (typically g.n)
 };


 //----------------------------------------------------
 // PART 1.5: Data Loader 
 //----------------------------------------------------
 Graph loadGraph(const std::string& filename, GraphFormat format = GraphFormat::AUTO_DETECT) {
     std::ifstream fin(filename);
     if (!fin) {
         throw std::runtime_error("错误: 无法打开图文件: " + filename);
     }

     std::string line;
     int n = 0;
     long long declared_m = 0; 
     GraphFormat detected_format = format;
     Graph g;
     bool p_line_found = false;
     bool header_read = false; 
     bool reading_weights = false;
     int weights_read = 0;
     long long edges_read_count = 0; 


     // --- Auto-Detection Phase (if needed) ---
     if (detected_format == GraphFormat::AUTO_DETECT) {
         std::streampos initial_pos = fin.tellg();
         int lines_scanned = 0;
         const int MAX_SCAN_LINES = 100; 

         while (lines_scanned < MAX_SCAN_LINES && std::getline(fin, line)) {
             lines_scanned++;
              line.erase(0, line.find_first_not_of(" \t\r\n")); 
             line.erase(line.find_last_not_of(" \t\r\n") + 1); 
             if (line.empty() || line[0] == 'c' || line[0] == '%') continue; 

             std::stringstream ss(line);
             char type;
             if (line[0] == 'p') { 
                 std::string format_str;
                 int n_detect;
                 long long e_detect;
                 ss >> type >> format_str >> n_detect >> e_detect;
                 if (!ss.fail() && (format_str == "col" || format_str == "edge" || format_str == "max" || format_str == "clique" || format_str == "sp")) {
                     detected_format = GraphFormat::DIMACS_CLQ;
                      if (isDebugEnabled) {std::cout << "DEBUG: Detected DIMACS format." << std::endl;}
                     break;
                 }
             } else {
                  std::stringstream ss_orig(line);
                  int n_orig;
                  long long m_orig;
                  if (ss_orig >> n_orig >> m_orig && n_orig > 0 && ss_orig.eof()) {
                      detected_format = GraphFormat::WEIGHTED_LIST;
                        if (isDebugEnabled) {std::cout << "DEBUG: Detected WEIGHTED_LIST format." << std::endl;}
                      break;
                  }
             }
         }
          fin.clear(); 
          fin.seekg(initial_pos); 
          if (detected_format == GraphFormat::AUTO_DETECT){
               detected_format = GraphFormat::DIMACS_CLQ; 
               std::cerr << "警告: 无法自动检测图格式，假设为 WEIGHTED_LIST。\n";
          }
     }

     // --- Loading Phase ---
     if (detected_format == GraphFormat::DIMACS_CLQ) {
         //std::cout << "Info: Loading graph as DIMACS format." << std::endl;
         while (std::getline(fin, line)) {
             if (line.empty()) continue;
              line.erase(0, line.find_first_not_of(" \t\r\n"));
              if (line.empty()) continue; 

             std::stringstream ss(line);
             char type = line[0];

             if (type == 'c' || type == '%') { 
                 continue; 
  } else if (type == 'p') {
                 if (p_line_found) {
                     std::cerr << "警告: 找到多个 'p' 行，忽略后续行。\n";
                     continue;
                 }
                 std::string format_str;
                 ss >> type >> format_str >> n >> declared_m;
                 if (ss.fail() || n <= 0) {
                     throw std::runtime_error("错误: 无法解析 DIMACS 'p' 行或顶点数无效: " + line);
                 }
                 g = Graph(n);
                 for (int i = 0; i < n; ++i) {
                     long long assigned_weight = ( (i + 1) % 200 ) + 1;
                     if (assigned_weight == 0) {
                     }
                     g.weight[i] =  assigned_weight;
                     g.original_weight[i] = assigned_weight;
                 }
                 // --- MODIFIED SECTION END ---
                 p_line_found = true;
                   if (isDebugEnabled) {std::cout << "DEBUG: Parsed 'p' line: n=" << n << ", declared_m=" << declared_m << std::endl;}
             } else if (type == 'e') {
                 if (!p_line_found) {
                     throw std::runtime_error("错误: 在 'p' 行之前找到 DIMACS 'e' 行。");
                 }
                 int u, v;
                 ss >> type >> u >> v; 
                 if (ss.fail()) {
                     std::cerr << "警告: 无法解析 DIMACS 'e' 行: " << line << std::endl;
                     continue;
                 }
                 g.addEdge(u - 1, v - 1);
                 edges_read_count++;
             } else {
                  std::stringstream ss_edge(line); 
                  int u_no_e, v_no_e;
                  if(p_line_found && (ss_edge >> u_no_e >> v_no_e) && ss_edge.eof()){
                        g.addEdge(u_no_e - 1, v_no_e - 1);
                        edges_read_count++;
                  } else {
                       std::cerr << "警告: 忽略无法识别的行类型 '" << type << "' 或格式错误的行，在 DIMACS 文件中: " << line << std::endl;
                  }
             }
         }
         if (!p_line_found){
              throw std::runtime_error("错误: DIMACS 文件中未找到 'p' 行。");
         }


     } else if (detected_format == GraphFormat::WEIGHTED_LIST) {
          std::cout << "Info: Loading graph as WEIGHTED_LIST format." << std::endl;
         while (std::getline(fin, line)) {
              line.erase(0, line.find_first_not_of(" \t\r\n")); 
              line.erase(line.find_last_not_of(" \t\r\n") + 1); 
             if (line.empty() || line[0] == 'c' || line[0] == '%') continue;

             std::stringstream ss(line);
             if (!header_read) {
                 ss >> n >> declared_m;
                 if (ss.fail() || n <= 0 || !ss.eof()) { 
                     throw std::runtime_error("错误: 无法解析 WEIGHTED_LIST 标题行 'N M' 或 N 无效: " + line);
                 }
                 g = Graph(n);
                 header_read = true;
                 reading_weights = true;
                  if (isDebugEnabled) {std::cout << "DEBUG: Parsed header: n=" << n << ", declared_m=" << declared_m << std::endl;}
                 continue;
             }

             if (reading_weights) {
                 long long w;
                 if (!(ss >> w)) {
                      throw std::runtime_error("错误: 读取顶点 " + std::to_string(weights_read) + " 的权重时出错: " + line);
                 }
                  if (!ss.eof()){ 
                       std::cerr << "警告: 权重行上有额外数据 " << weights_read << ": " << line << std::endl;
                  }
                 if (w <= 0) {
                     std::cerr << "警告: 顶点 " << weights_read << " 的权重 " << w << " 非正。设置为 1。\n";
                     w = 1;
                 }
                 g.weight[weights_read] = w;
                 g.original_weight[weights_read] = w;
                 weights_read++;
                 if (weights_read == n) {
                     reading_weights = false; 
                 }
             } else { 
                 int u, v;
                 if (!(ss >> u >> v) || !ss.eof()) { 
                     std::cerr << "警告: 无法解析 WEIGHTED_LIST 边行: " << line << std::endl;
                     continue;
                 }
                 g.addEdge(u, v);
                 edges_read_count++;
             }
         }
          if (!header_read){
               throw std::runtime_error("错误: WEIGHTED_LIST 文件中未找到标题行 'N M'。");
          }
         if (weights_read != n) {
              throw std::runtime_error("错误: 读取的权重数 (" + std::to_string(weights_read) + ") 与预期的顶点数 (" + std::to_string(n) + ") 不匹配。");
         }

     } else {
         throw std::runtime_error("错误: 未知的图格式。");
     }

     g.finalize(); 

       if (isDebugEnabled) {
            std::cout << "DEBUG: Graph loaded. n=" << g.n << ", final m=" << g.m << std::endl;
            if (g.n > 0 && g.N_v_closed.size() > 0 && !g.N_v_closed.empty() && !g.N_v_closed[0].empty()) {
                 std::cout << "DEBUG: Vertex 0: weight=" << g.weight[0] << ", original_weight=" << g.original_weight[0] << ", degree=" << g.deg[0] << ", N[0]=";
                 for(size_t i=0; i < std::min((size_t)10, g.N_v_closed[0].size()); ++i) std::cout << g.N_v_closed[0][i] << " ";
                 if (g.N_v_closed[0].size() > 10) std::cout << "...";
                 std::cout << std::endl;
            }
       }
      if ((declared_m > 0 || detected_format == GraphFormat::DIMACS_CLQ) && declared_m != g.m) {
            std::cerr << "警告: 文件中声明的边数 (" << declared_m
                      << ") 与加载后唯一的边数 (" << g.m << ") 不匹配。\n";
      }
     return g;
 }


 //----------------------------------------------------
 // PART 2. 规约规则 
 //----------------------------------------------------
 static bool applyReductionRules(Graph &g, std::vector<bool> &fixed) {
    bool changed_in_pass = false;
    std::vector<int> vertices_to_process; 
    vertices_to_process.reserve(g.n);
    for (int v = 0; v < g.n; ++v) {
        if (v < g.deg.size() && g.deg[v] >= 0 && v < fixed.size() && !fixed[v]) {
            vertices_to_process.push_back(v);
        }
    }

    std::vector<int> vertices_to_delete; 
    std::vector<int> vertices_to_fix; 
    std::vector<std::pair<int, long long>> weight_updates; 
    std::vector<std::pair<int, int>> revert_map_updates; 

    for (int v : vertices_to_process) {
         if (v < 0 || v >= g.n || v >= g.deg.size() || g.deg[v] < 0 || v >= fixed.size() || fixed[v]) continue;

        if (g.deg[v] == 0) {
             if (isDebugEnabled) {std::cout << "DEBUG: Reduction Rule 1 on vertex " << v << std::endl;}
            vertices_to_fix.push_back(v);
            vertices_to_delete.push_back(v); 
            changed_in_pass = true;
            continue; 
        }

        if (g.deg[v] == 1) {
            if (v >= g.adj.size() || g.adj[v].empty()) continue; 
            int u = g.adj[v].front(); 
             if (u < 0 || u >= g.n || u >= g.deg.size() || g.deg[u] < 0) continue; 

             if (v >= g.weight.size() || u >= g.weight.size()) continue;

            if (g.weight[v] >= g.weight[u]) {
                  if (isDebugEnabled) {std::cout << "DEBUG: Reduction Rule 2 on pendant " << v << " (neighbor " << u << ")" << std::endl;}
                vertices_to_fix.push_back(u); 
                vertices_to_delete.push_back(v); 
                vertices_to_delete.push_back(u); 
                changed_in_pass = true;
            }
            else {
                 if (isDebugEnabled) {std::cout << "DEBUG: Reduction Rule 3 on pendant " << v << " (neighbor " << u << ")" << std::endl;}
                long long new_weight_u = g.weight[u] - g.weight[v];
                 if (new_weight_u <= 0) { 
                      if (isDebugEnabled) {std::cerr << "DEBUG Warning: Reduction Rule 3 resulted in non-positive weight for " << u << ". Fixing u instead." << std::endl;}
                      vertices_to_fix.push_back(u);
                      vertices_to_delete.push_back(v);
                      vertices_to_delete.push_back(u);
                 } else {
                     weight_updates.push_back({u, new_weight_u});
                     vertices_to_fix.push_back(v);      
                     vertices_to_delete.push_back(v);   
                     revert_map_updates.push_back({u, v}); 
                 }
                changed_in_pass = true;
            }
            continue; 
        }

         if (g.deg[v] == 2) {
            if (v >= g.adj.size() || g.adj[v].size() != 2) continue; 
            int u1 = g.adj[v][0];
            int u2 = g.adj[v][1];
              if (u1 < 0 || u1 >= g.n || u1 >= g.deg.size() || g.deg[u1] < 0 || u1 >= g.adj.size()) continue;
              if (u2 < 0 || u2 >= g.n || u2 >= g.deg.size() || g.deg[u2] < 0 || u2 >= g.adj.size()) continue;

             bool triangle_found = false;
             bool u1_shorter = g.adj[u1].size() < g.adj[u2].size();
             int node_to_check = u1_shorter ? u2 : u1;
             const auto& list_to_search = u1_shorter ? g.adj[u1] : g.adj[u2];
             triangle_found = std::binary_search(list_to_search.begin(), list_to_search.end(), node_to_check);


             if (triangle_found) {
                  if (g.deg[v] == -2) continue; // Already processed as part of another triangle rule application

                 if (v >= g.weight.size() || u1 >= g.weight.size() || u2 >= g.weight.size()) continue;
                 
                 if (g.deg[u1] == 2) { 
                     long long min_v_u1_weight = std::min(g.weight[v], g.weight[u1]);
                     if (g.weight[u2] <= min_v_u1_weight) { 
                          if (isDebugEnabled) {std::cout << "DEBUG: Reduction Rule 4 on triangle (" << v << "," << u1 << "," << u2 << ")" << std::endl;}
                         vertices_to_fix.push_back(u2);
                         vertices_to_delete.push_back(v); vertices_to_delete.push_back(u1); vertices_to_delete.push_back(u2);
                         changed_in_pass = true;
                     }
                     else { 
                          if (isDebugEnabled) {std::cout << "DEBUG: Reduction Rule 5 on triangle (" << v << "," << u1 << "," << u2 << ")" << std::endl;}
                         long long new_weight_u2 = g.weight[u2] - min_v_u1_weight;
                          if (new_weight_u2 <= 0) {
                               if (isDebugEnabled) {std::cerr << "DEBUG Warning: Reduction Rule 5 resulted in non-positive weight for " << u2 << ". Fixing u2 instead." << std::endl;}
                               vertices_to_fix.push_back(u2);
                               vertices_to_delete.push_back(v); vertices_to_delete.push_back(u1); vertices_to_delete.push_back(u2);
                          } else {
                              weight_updates.push_back({u2, new_weight_u2});
                              int node_to_fix = (g.weight[v] <= g.weight[u1]) ? v : u1;
                              vertices_to_fix.push_back(node_to_fix);
                              vertices_to_delete.push_back(v); vertices_to_delete.push_back(u1); 
                              revert_map_updates.push_back({u2, node_to_fix}); 
                          }
                         changed_in_pass = true;
                     }
                      g.deg[v] = -2; g.deg[u1] = -2; 
                 }
                 else if (g.deg[u2] == 2) { 
                      long long min_v_u2_weight = std::min(g.weight[v], g.weight[u2]);
                     if (g.weight[u1] <= min_v_u2_weight) { 
                         if (isDebugEnabled) {std::cout << "DEBUG: Reduction Rule 4 on triangle (" << v << "," << u2 << "," << u1 << ")" << std::endl;}
                         vertices_to_fix.push_back(u1);
                         vertices_to_delete.push_back(v); vertices_to_delete.push_back(u2); vertices_to_delete.push_back(u1);
                         changed_in_pass = true;
                     }
                     else { 
                          if (isDebugEnabled) {std::cout << "DEBUG: Reduction Rule 5 on triangle (" << v << "," << u2 << "," << u1 << ")" << std::endl;}
                          long long new_weight_u1 = g.weight[u1] - min_v_u2_weight;
                          if (new_weight_u1 <= 0) {
                              if (isDebugEnabled) {std::cerr << "DEBUG Warning: Reduction Rule 5 resulted in non-positive weight for " << u1 << ". Fixing u1 instead." << std::endl;}
                               vertices_to_fix.push_back(u1);
                               vertices_to_delete.push_back(v); vertices_to_delete.push_back(u2); vertices_to_delete.push_back(u1);
                          } else {
                              weight_updates.push_back({u1, new_weight_u1});
                              int node_to_fix = (g.weight[v] <= g.weight[u2]) ? v : u2;
                              vertices_to_fix.push_back(node_to_fix);
                              vertices_to_delete.push_back(v); vertices_to_delete.push_back(u2); 
                              revert_map_updates.push_back({u1, node_to_fix}); 
                          }
                         changed_in_pass = true;
                     }
                      g.deg[v] = -2; g.deg[u2] = -2; 
                 }
             } 
         } 
    } 

    if(changed_in_pass) {
         for(const auto& update : weight_updates) {
             if (update.first >= 0 && update.first < g.n && update.first < g.weight.size()) { 
                  g.weight[update.first] = update.second;
                   if (isDebugEnabled) {std::cout << "DEBUG: Updated weight of " << update.first << " to " << update.second << std::endl;}
              }
         }

         for(int fv : vertices_to_fix) {
             if (fv >= 0 && fv < g.n && fv < fixed.size() && !fixed[fv]) { 
                 fixed[fv] = true;
                  if (isDebugEnabled) {std::cout << "DEBUG: Fixed vertex " << fv << std::endl;}
             }
         }

         for(const auto& update : revert_map_updates) {
              if (update.first >= 0 && update.first < g.n && update.first < g.revert_map.size()) { 
                  g.revert_map[update.first] = update.second; 
                    if (isDebugEnabled) {std::cout << "DEBUG: Set g.revert_map[" << update.first << "] = " << update.second << std::endl;}
              }
         }

        std::unordered_set<int> final_deletion_set;
        for(int rv : vertices_to_delete) {
             if (rv >= 0 && rv < g.n && rv < g.deg.size() && g.deg[rv] != -1) { 
                 final_deletion_set.insert(rv);
             }
        }
        for(int v_mark=0; v_mark<g.n; ++v_mark) {
             if (v_mark < g.deg.size() && g.deg[v_mark] == -2) { 
                final_deletion_set.insert(v_mark);
                g.deg[v_mark] = -1; 
            }
        }


        for (int rv : final_deletion_set) {
             if (rv < 0 || rv >= g.n || rv >= g.deg.size() || g.deg[rv] == -1) continue; 

               if (isDebugEnabled) {std::cout << "DEBUG: Deleting vertex " << rv << std::endl;}

             std::vector<int> neighbors_copy;
             if (rv < g.adj.size()) { 
                 neighbors_copy = g.adj[rv];
                 g.adj[rv].clear(); 
                 // g.adj[rv].shrink_to_fit(); // Can be slow, consider removing if not critical
             }
             g.deg[rv] = -1; 

             for (int neighbor : neighbors_copy) {
                 if (neighbor >= 0 && neighbor < g.n && neighbor < g.deg.size() && g.deg[neighbor] >= 0 &&
                     final_deletion_set.find(neighbor) == final_deletion_set.end())
                 {
                      if (neighbor < g.adj.size()) { 
                           auto& neighbor_adj = g.adj[neighbor];
                           auto it = std::remove(neighbor_adj.begin(), neighbor_adj.end(), rv);
                           if (it != neighbor_adj.end()) { 
                               neighbor_adj.erase(it, neighbor_adj.end());
                               g.deg[neighbor]--; 
                               if (g.deg[neighbor] < 0) { // Should not happen if logic is correct
                                    g.deg[neighbor] = 0; 
                               }
                           }
                      }
                 }
             }
        }

         long long total_degree = 0;
         for(int i=0; i<g.n; ++i) {
              if(i < g.deg.size() && g.deg[i] > 0) { 
                 total_degree += g.deg[i];
             }
         }
         g.m = total_degree / 2;
    } 

    return changed_in_pass;
 }

 static void revertSolution(const Graph &g, std::vector<bool> &solution_vec) {
     int n = g.n;
     if (isDebugEnabled) {std::cout << "DEBUG: Starting revertSolution..." << std::endl;}
     if (solution_vec.size() != (size_t)n) { 
         std::cerr << "ERROR: revertSolution called with mismatched solution vector size! sol_size=" << solution_vec.size() << " n=" << n << std::endl;
         return;
     }

     for (int u = 0; u < n; ++u) {
         if (u < g.revert_map.size() && g.revert_map[u] != -1) {
             int v_fixed = g.revert_map[u]; 
             if (u < solution_vec.size() && solution_vec[u]) {
                 if (v_fixed >= 0 && v_fixed < solution_vec.size()) {
                     if (solution_vec[v_fixed]) { 
                         if (isDebugEnabled) {std::cout << "DEBUG: Reverting: Removing fixed vertex " << v_fixed << " because reduced vertex " << u << " IS in solution." << std::endl;}
                         solution_vec[v_fixed] = false;
                     }
                 }
             }
         }
     }
     if (isDebugEnabled) {std::cout << "DEBUG: Finished revertSolution." << std::endl;}
 }


 //----------------------------------------------------
 // PART 3. ConstructDS & Helpers 
 //----------------------------------------------------
 static void updateUndominatedAndRedundantLists(const Graph &g, MWDSContext &ctx) {
    ctx.undominated_vertices_list.clear();
    ctx.redundant_vertices_list.clear();
    std::fill(ctx.is_undominated.begin(), ctx.is_undominated.end(), false);
    std::fill(ctx.is_redundant.begin(), ctx.is_redundant.end(), false);

    for (int v = 0; v < g.n; ++v) {
        if (v < g.deg.size() && g.deg[v] < 0) continue; 

        if (v < ctx.coverCount.size() && ctx.coverCount[v] == 0) {
            ctx.undominated_vertices_list.push_back(v);
            if (v < ctx.is_undominated.size()) ctx.is_undominated[v] = true;
        }

        if (v < ctx.inSolution.size() && ctx.inSolution[v]) {
            // A vertex is redundant if its score (benefit of removal) is 0,
            // meaning it doesn't uniquely cover any vertex.
            // The score for a vertex in solution is - (sum of weights of uniquely covered nodes).
            // So if score is 0, it's redundant.
            if (v < ctx.current_vertex_scores.size() && std::abs(ctx.current_vertex_scores[v]) < 1e-9 ) { // score is 0
                 ctx.redundant_vertices_list.push_back(v);
                 if (v < ctx.is_redundant.size()) ctx.is_redundant[v] = true;
            }
        }
    }
 }


 static void updateCoverCountForVertexAndNeighbors(const Graph &g, MWDSContext &ctx, int v, int delta) {
    if (v < 0 || v >= g.n || v >= g.deg.size() || g.deg[v] < 0) return;

    if (v < ctx.coverCount.size()) {
        ctx.coverCount[v] += delta;
         if (ctx.coverCount[v] < 0) ctx.coverCount[v] = 0; 
    }
    if (v < g.adj.size()) { 
        for (int neighbor : g.adj[v]) {
            if (neighbor >= 0 && neighbor < g.n && neighbor < g.deg.size() && g.deg[neighbor] >= 0 && neighbor < ctx.coverCount.size()) {
                ctx.coverCount[neighbor] += delta;
                if (ctx.coverCount[neighbor] < 0) ctx.coverCount[neighbor] = 0; 
            }
        }
    }
}

 static void calculateAllVertexScores(const Graph &g, MWDSContext &ctx) {
    for (int v_score_idx = 0; v_score_idx < g.n; ++v_score_idx) {
        if (v_score_idx >= g.deg.size() || g.deg[v_score_idx] < 0) { 
            if (v_score_idx < ctx.current_vertex_scores.size()) ctx.current_vertex_scores[v_score_idx] = -std::numeric_limits<long long>::max(); 
            continue;
        }

        long long score_val = 0;
        if (v_score_idx < ctx.inSolution.size() && !ctx.inSolution[v_score_idx]) { 
            if (v_score_idx < g.N_v_closed.size()) {
                for (int u_neighbor_or_self : g.N_v_closed[v_score_idx]) { 
                    if (u_neighbor_or_self >= g.deg.size() || g.deg[u_neighbor_or_self] < 0) continue; 
                    if (u_neighbor_or_self < ctx.coverCount.size() && ctx.coverCount[u_neighbor_or_self] == 0) {
                        if (u_neighbor_or_self < ctx.freq_or_penalty_weight.size()) {
                            score_val += ctx.freq_or_penalty_weight[u_neighbor_or_self];
                        }
                    }
                }
            }
        } else if (v_score_idx < ctx.inSolution.size() && ctx.inSolution[v_score_idx]) { 
             if (v_score_idx < g.N_v_closed.size()) {
                for (int u_neighbor_or_self : g.N_v_closed[v_score_idx]) {
                    if (u_neighbor_or_self >= g.deg.size() || g.deg[u_neighbor_or_self] < 0) continue;
                    
                    // Check if u_neighbor_or_self would become uncovered if v_score_idx is removed.
                    // This means u_neighbor_or_self is currently covered, and ALL its coverers in the solution
                    // are ONLY v_score_idx OR u_neighbor_or_self itself (if it's in solution).
                    // More simply: if coverCount[u_neighbor_or_self] == 1, and v_score_idx is in N[u_neighbor_or_self]
                    // (which it is, as u_neighbor_or_self is in N[v_score_idx]), then v_score_idx is critical for u_neighbor_or_self.
                    if (u_neighbor_or_self < ctx.coverCount.size() && ctx.coverCount[u_neighbor_or_self] == 1) {
                        // If u_neighbor_or_self is covered by exactly one vertex, and v_score_idx is in its neighborhood
                        // (which is true by definition of iterating N_v_closed[v_score_idx]),
                        // then removing v_score_idx makes u_neighbor_or_self uncovered (unless u_neighbor_or_self == v_score_idx).
                        // The critical part is that v_score_idx must be the *sole* reason (among others in solution) that u_neighbor_or_self is covered.
                        // The logic for coverCount[u_neighbor_or_self] == 1 correctly identifies this.
                        // If v_score_idx is u_neighbor_or_self, and coverCount[v_score_idx]==1, it means only v_score_idx covers itself.
                        if (u_neighbor_or_self < ctx.freq_or_penalty_weight.size()) {
                           score_val += ctx.freq_or_penalty_weight[u_neighbor_or_self];
                        }
                    }
                }
             }
            score_val = -score_val; 
        }
        if (v_score_idx < ctx.current_vertex_scores.size()) ctx.current_vertex_scores[v_score_idx] = score_val;
    }
}


 static double getScorePerWeight(const Graph &g, const MWDSContext &ctx, int v) {
    if (v < 0 || v >= g.n || v >= g.weight.size() || g.weight[v] <= 0 || v >= ctx.current_vertex_scores.size()) {
        return -std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(ctx.current_vertex_scores[v]) / static_cast<double>(g.weight[v]);
 }


 static bool isCompleteCover(const Graph &g, const MWDSContext &ctx) {
    for(int v_check=0; v_check<g.n; v_check++) {
        if(v_check < g.deg.size() && g.deg[v_check] >= 0) { 
            if(v_check >= ctx.coverCount.size() || ctx.coverCount[v_check] == 0) { 
                 if (isDebugEnabled) { std::cout << "DEBUG: isCompleteCover failed. Vertex " << v_check << " has coverCount " << (v_check < ctx.coverCount.size() ? std::to_string(ctx.coverCount[v_check]) : "out_of_bounds") << std::endl;}
                return false; 
            }
        }
    }
    return true;
 }


 static std::vector<bool> ConstructDS(Graph &g, MWDSContext& ctx) {
     int n = g.n;
     std::vector<bool> fixed(n, false); 

     if (isDebugEnabled) {std::cout << "DEBUG: Starting reduction phase..." << std::endl;}
     int reduction_passes = 0;
     while(applyReductionRules(g, fixed)) { 
          reduction_passes++;
     }
     if (isDebugEnabled) {std::cout << "DEBUG: Reduction phase finished after " << reduction_passes << " passes." << std::endl;}

     ctx.inSolution.assign(n, false);
     ctx.freq_or_penalty_weight.assign(n, 1LL);
     ctx.conf.assign(n, 1);
     ctx.age.assign(n, 0LL);
     ctx.coverCount.assign(n, 0);
     std::fill(ctx.is_undominated.begin(), ctx.is_undominated.end(), false);
     std::fill(ctx.is_redundant.begin(), ctx.is_redundant.end(), false);
     ctx.undominated_vertices_list.clear();
     ctx.redundant_vertices_list.clear();


     for(int v_fixed_idx=0; v_fixed_idx<n; v_fixed_idx++) {
          if(v_fixed_idx < fixed.size() && fixed[v_fixed_idx]) { 
             if (v_fixed_idx < ctx.inSolution.size()) { 
                  ctx.inSolution[v_fixed_idx] = true;
                  updateCoverCountForVertexAndNeighbors(g, ctx, v_fixed_idx, 1);
                  if (isDebugEnabled) {std::cout << "DEBUG: Adding fixed vertex " << v_fixed_idx << " to initial solution." << std::endl;}
             }
         }
     }
     
    // Setup for MAXHEAP
    graph_ptr_for_heap_ = &g;
    context_ptr_for_heap_ = &ctx;
    MAXHEAP max_heap(n); // n is capacity, assumes vertex ids 0 to n-1

    calculateAllVertexScores(g, ctx); // Initial scores for all vertices
    updateUndominatedAndRedundantLists(g, ctx); // Populate undominated list

    // Initial population of the MAXHEAP
    for (int v_idx = 0; v_idx < n; ++v_idx) {
        if (v_idx < g.deg.size() && g.deg[v_idx] >= 0 && 
            v_idx < ctx.inSolution.size() && !ctx.inSolution[v_idx] &&
            v_idx < ctx.current_vertex_scores.size() && ctx.current_vertex_scores[v_idx] > 0) {
            max_heap.insert(v_idx);
        }
    }

     if (isDebugEnabled) {std::cout << "DEBUG: Starting constructing phase (using MAXHEAP)..." << std::endl;}
     
     while(!isCompleteCover(g, ctx)) { 
          if (max_heap.getSize() == 0) { // Changed from pq.empty()
               std::cerr << "警告: ConstructDS 无法找到顶点来提高可行性 (MAXHEAP 为空)。图中可能存在问题。\n";
               int fallback_v = -1;
               if (!ctx.undominated_vertices_list.empty()) {
                   int target_uncovered = ctx.undominated_vertices_list.front();
                   long long min_w = std::numeric_limits<long long>::max();
                   if (target_uncovered < g.N_v_closed.size()){
                       for (int potential_adder : g.N_v_closed[target_uncovered]) {
                           if (potential_adder < g.deg.size() && g.deg[potential_adder] >=0 &&
                               potential_adder < ctx.inSolution.size() && !ctx.inSolution[potential_adder] &&
                               potential_adder < g.weight.size() && g.weight[potential_adder] < min_w) {
                               min_w = g.weight[potential_adder];
                               fallback_v = potential_adder;
                           }
                       }
                       if (fallback_v == -1 && !g.N_v_closed[target_uncovered].empty()) { 
                            for(int pa : g.N_v_closed[target_uncovered]){
                                if(pa < g.deg.size() && g.deg[pa]>=0 && pa < ctx.inSolution.size() && !ctx.inSolution[pa]){
                                    if (fallback_v == -1 || (pa < g.weight.size() && fallback_v < g.weight.size() && g.weight[pa] < g.weight[fallback_v])) {
                                         fallback_v = pa;
                                    }
                                }
                            }
                       }
                   }
               }

               if (fallback_v != -1) {
                    if (isDebugEnabled) {std::cout << "DEBUG: ConstructDS MAXHEAP empty, using fallback vertex " << fallback_v << std::endl;}
                    if (fallback_v < ctx.inSolution.size()) ctx.inSolution[fallback_v] = true;
                    updateCoverCountForVertexAndNeighbors(g, ctx, fallback_v, 1);
                    
                    calculateAllVertexScores(g, ctx); 
                    updateUndominatedAndRedundantLists(g, ctx);
                    // Update heap after fallback addition
                    for (int v_idx = 0; v_idx < n; ++v_idx) {
                        if (v_idx < g.deg.size() && g.deg[v_idx] >= 0 && v_idx < ctx.inSolution.size() && !ctx.inSolution[v_idx]) {
                            max_heap.changeVal(v_idx); // Will insert if score>0 and not present, or update/remove
                        } else if (max_heap.isInHeap(v_idx)) { // If it was a candidate but now in solution or deleted
                            max_heap.removeVertex(v_idx);
                        }
                    }
                    continue; 
               } else {
                    std::cerr << "错误: ConstructDS 无法使解可行 (MAXHEAP and Fallback failed)。图可能未连接或存在问题。\n";
                    break; 
               }
          } 

         int bestV = max_heap.removeRoot();

         if (bestV < 0 || bestV >= n || bestV >= g.deg.size() || g.deg[bestV] < 0 || bestV >= ctx.inSolution.size() || ctx.inSolution[bestV]) {
             continue; 
         }
         // No stale check needed as MAXHEAP should be up-to-date if changeVal is used correctly.
         // However, calculateAllVertexScores is global, so we update heap based on new global scores.
         
         if (isDebugEnabled) {std::cout << "DEBUG: ConstructDS adding vertex " << bestV << std::endl;}
         if (bestV < ctx.inSolution.size()) ctx.inSolution[bestV] = true; 
         updateCoverCountForVertexAndNeighbors(g, ctx, bestV, 1);
         
         calculateAllVertexScores(g, ctx); 
         updateUndominatedAndRedundantLists(g, ctx); 

         // Update MAXHEAP based on new scores
         // Vertices whose scores changed need to be updated in the heap.
         // bestV is already removed.
         for (int v_idx = 0; v_idx < n; ++v_idx) {
             if (v_idx == bestV) continue; // Already processed (removed from heap, added to solution)

             if (v_idx < g.deg.size() && g.deg[v_idx] >= 0 && v_idx < ctx.inSolution.size() && !ctx.inSolution[v_idx]) {
                 // This vertex is a candidate. Update its status in the heap.
                 max_heap.changeVal(v_idx); // changeVal will insert if new, update if existing, or remove if score <= 0
             } else {
                 // This vertex is not a candidate (e.g. in solution, or deleted from graph)
                 // If it was in the heap, it should be removed. changeVal on a non-candidate (score likely min_inf) would remove it.
                 // Or more explicitly:
                 if (max_heap.isInHeap(v_idx)) {
                     max_heap.removeVertex(v_idx);
                 }
             }
         }
     } 
      if (isDebugEnabled) {std::cout << "DEBUG: Constructing phase finished." << std::endl;}

      if (isDebugEnabled) {std::cout << "DEBUG: Starting shrinking phase..." << std::endl;}
     bool removedSomething = true;
     while(removedSomething) {
         removedSomething = false;
         calculateAllVertexScores(g, ctx); 
         updateUndominatedAndRedundantLists(g, ctx); 

         if(!ctx.redundant_vertices_list.empty()) {
             int vertex_to_remove = -1;
             long long max_weight_val = -1; 

             const int bms_t = 100;
             std::vector<int> sampled_redundant;
             if (ctx.redundant_vertices_list.size() <= (size_t)bms_t) {
                 sampled_redundant = ctx.redundant_vertices_list;
             } else {
                 std::sample(ctx.redundant_vertices_list.begin(), ctx.redundant_vertices_list.end(),
                             std::back_inserter(sampled_redundant),
                             bms_t, ctx.local_rng);
             }
            
             for (int current_vertex : sampled_redundant) { 
                 if (current_vertex < g.weight.size() && g.weight[current_vertex] > max_weight_val) { 
                     max_weight_val = g.weight[current_vertex];
                     vertex_to_remove = current_vertex;
                 }
             }
             
             if(vertex_to_remove != -1) {
                   if (isDebugEnabled) {std::cout << "DEBUG: Shrinking: Removing redundant vertex " << vertex_to_remove << " with weight " << max_weight_val << std::endl;}
                 if (vertex_to_remove < ctx.inSolution.size()) ctx.inSolution[vertex_to_remove] = false; 
                 updateCoverCountForVertexAndNeighbors(g, ctx, vertex_to_remove, -1);
                 removedSomething = true;
                 // Scores and lists will be updated at the start of the next iteration or after loop
             }
         }
     } 
      if (isDebugEnabled) {std::cout << "DEBUG: Shrinking phase finished." << std::endl;}
      
      calculateAllVertexScores(g, ctx); 
      updateUndominatedAndRedundantLists(g, ctx);

      // Clear static pointers for heap after use in this function
      graph_ptr_for_heap_ = nullptr;
      context_ptr_for_heap_ = nullptr;


       if (isDebugEnabled) {
            MWDSContext verify_ctx = ctx; 
            std::fill(verify_ctx.coverCount.begin(), verify_ctx.coverCount.end(), 0);
            for(int v_sol_idx = 0; v_sol_idx < n; ++v_sol_idx) {
                if (v_sol_idx < verify_ctx.inSolution.size() && verify_ctx.inSolution[v_sol_idx]) {
                    updateCoverCountForVertexAndNeighbors(g, verify_ctx, v_sol_idx, 1);
                }
            }
            if (!isCompleteCover(g, verify_ctx)) { 
                 std::cerr << "错误: ConstructDS 产生的最终解在缩减图上似乎不可行！\n";
            } else {
                 std::cout << "DEBUG: ConstructDS final solution appears feasible on reduced graph g." << std::endl;
            }
            long long constructed_cost = 0;
            int constructed_size = 0;
            std::vector<bool> solution_before_revert = ctx.inSolution; 
            revertSolution(g, solution_before_revert);

            for(int v_cost_idx=0; v_cost_idx<n; ++v_cost_idx) {
                 if(v_cost_idx < solution_before_revert.size() && solution_before_revert[v_cost_idx]) { 
                      constructed_size++;
                      if (v_cost_idx < g.original_weight.size()) { 
                          constructed_cost += g.original_weight[v_cost_idx];
                      }
                 }
            }
            std::cout << "DEBUG: ConstructDS finished. Initial solution size (after revert) = " << constructed_size << ", Initial cost (original weights) = " << constructed_cost << std::endl;
       }

     return ctx.inSolution; 
 }


 //----------------------------------------------------
 // PART 4. CC2V3+ 
 //----------------------------------------------------
 static void initConfiguration(MWDSContext &ctx, int n) {
     ctx.conf.assign(n, 1); 
     ctx.age.assign(n, 0LL);
 }

 static void onAddVertex_UpdateConf(const Graph &g, MWDSContext &ctx, int v_added) {
     if (v_added < 0 || v_added >= g.n || v_added >= g.deg.size() || g.deg[v_added] < 0) return;

     if (v_added < g.N_v_closed.size()) {
        for (int v_n_covered_by_v_added : g.N_v_closed[v_added]) { 
            if (v_n_covered_by_v_added >= g.deg.size() || g.deg[v_n_covered_by_v_added] < 0) continue;

            if (v_n_covered_by_v_added < g.N_v_closed.size()) {
                for (int u_potential_candidate : g.N_v_closed[v_n_covered_by_v_added]) {
                    if (u_potential_candidate >= g.deg.size() || g.deg[u_potential_candidate] < 0) continue;
                    if (u_potential_candidate < ctx.conf.size() && ctx.conf[u_potential_candidate] == 0) {
                        ctx.conf[u_potential_candidate] = 1; 
                        if (u_potential_candidate < ctx.age.size()) ctx.age[u_potential_candidate] = 0; 
                    }
                }
            }
        }
     }
     if (v_added < ctx.age.size()) ctx.age[v_added] = 0; 
 }

 static void onRemoveVertex_UpdateConf(const Graph &g, MWDSContext &ctx, int v_removed, const std::vector<int> &became_uncovered_nodes) {
     if (v_removed < 0 || v_removed >= g.n) return;

     for(int x_uncovered : became_uncovered_nodes) {
          if (x_uncovered < 0 || x_uncovered >=g.n || x_uncovered >= g.deg.size() || g.deg[x_uncovered] <0) continue;
          if (x_uncovered < g.N_v_closed.size()) {
            for(int y_encouraged : g.N_v_closed[x_uncovered]) { 
                if (y_encouraged < 0 || y_encouraged >= g.n || y_encouraged >= g.deg.size() || g.deg[y_encouraged] < 0) continue;
                if (y_encouraged < ctx.conf.size()) ctx.conf[y_encouraged] = 2; 
                if (y_encouraged < ctx.age.size()) ctx.age[y_encouraged] = 0;    
            }
          }
     }
     
     if (v_removed < ctx.conf.size()) ctx.conf[v_removed] = 0;
     if (v_removed < ctx.age.size()) ctx.age[v_removed] = 0; 
 }


 //----------------------------------------------------
 // PART 5. 局部搜索 + 辅助程序 
 //----------------------------------------------------

 static void addToSolution(const Graph &g, MWDSContext &ctx, int v) {
      if (v<0 || v>= g.n || v >= g.deg.size() || g.deg[v] < 0 || v >= ctx.inSolution.size() || ctx.inSolution[v]) return; 
      if (isDebugEnabled) {std::cout << "DEBUG: Adding vertex " << v << " to solution." << std::endl;}
      
      ctx.inSolution[v] = true;
      if (v < ctx.is_redundant.size()) ctx.is_redundant[v] = false; 

      updateCoverCountForVertexAndNeighbors(g, ctx, v, 1);
      onAddVertex_UpdateConf(g, ctx, v); 
      
      for(int i=0; i < g.n; ++i) { if (i < ctx.age.size() && i !=v ) ctx.age[i]++; }
      if (v < ctx.age.size()) ctx.age[v] = 0; 
 }

 static std::vector<int> removeFromSolution(const Graph &g, MWDSContext &ctx, int v) {
      if (v<0 || v>= g.n || v >= g.deg.size() || g.deg[v] < 0 || v >= ctx.inSolution.size() || !ctx.inSolution[v]) return {}; 
      if (isDebugEnabled) {std::cout << "DEBUG: Removing vertex " << v << " from solution." << std::endl;}
      
      ctx.inSolution[v] = false;
      std::vector<int> became_uncovered_locally; 

      std::vector<int> old_cover_counts_N_v;
      std::vector<int> N_v_nodes_for_check;
      if (v < g.N_v_closed.size()) {
          for(int node_in_Nv : g.N_v_closed[v]){
              if (node_in_Nv < g.deg.size() && g.deg[node_in_Nv] >=0 && node_in_Nv < ctx.coverCount.size()){
                  N_v_nodes_for_check.push_back(node_in_Nv);
                  old_cover_counts_N_v.push_back(ctx.coverCount[node_in_Nv]);
              }
          }
      }

      updateCoverCountForVertexAndNeighbors(g, ctx, v, -1);

      for(size_t i=0; i < N_v_nodes_for_check.size(); ++i) {
          int u_affected = N_v_nodes_for_check[i];
          if (u_affected < ctx.coverCount.size() && ctx.coverCount[u_affected] == 0 && old_cover_counts_N_v[i] > 0) {
              became_uncovered_locally.push_back(u_affected);
          }
      }
      
      onRemoveVertex_UpdateConf(g, ctx, v, became_uncovered_locally);
      
      for(int i=0; i < g.n; ++i) { if (i < ctx.age.size() && i !=v ) ctx.age[i]++; }
      if (v < ctx.age.size()) ctx.age[v] = 0;

      return became_uncovered_locally;
 }


 static void updateFreqAndAge(const Graph &g, MWDSContext &ctx) { 
     int n = g.n;
     for(int v_freq_idx=0; v_freq_idx<n; v_freq_idx++){
         if(v_freq_idx < g.deg.size() && g.deg[v_freq_idx]>=0 && 
            v_freq_idx < ctx.coverCount.size() && ctx.coverCount[v_freq_idx] == 0) { 
             if (v_freq_idx < ctx.freq_or_penalty_weight.size() && ctx.freq_or_penalty_weight[v_freq_idx] < std::numeric_limits<long long>::max() / 2) { 
                 ctx.freq_or_penalty_weight[v_freq_idx]++;
             }
         }
     }
 }

 static int pickBestAmongSample(const Graph &g, MWDSContext &ctx, const std::vector<int> &candidates, int sampleSize, bool for_removal) {
     if(candidates.empty()) return -1;
     double bestScoreRatio = for_removal ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(); 
     int bestVertex = -1;
     std::vector<int> actual_sample;

     std::mt19937_64 sample_rng(ctx.local_rng());

     if (candidates.size() <= (size_t)sampleSize) {
         actual_sample = candidates;
     } else {
         std::sample(candidates.begin(), candidates.end(),
                     std::back_inserter(actual_sample),
                     sampleSize, sample_rng);
     }

     for(int v_sample : actual_sample) {
          if (v_sample < 0 || v_sample >= g.n || v_sample >= g.deg.size() || g.deg[v_sample] < 0) continue; 
          
          double currentScoreRatio = getScorePerWeight(g, ctx, v_sample);
          bool better = false;

          if (for_removal) { 
              if (currentScoreRatio < bestScoreRatio - 1e-9) { better = true;}
              else if (std::abs(currentScoreRatio - bestScoreRatio) < 1e-9) { 
                  if (bestVertex == -1) { better = true; }
                  else { 
                       if (v_sample < ctx.age.size() && bestVertex < ctx.age.size()) {
                           if (ctx.age[v_sample] > ctx.age[bestVertex]) { better = true; } 
                       }
                  }
              }
          } else { 
              if (currentScoreRatio > bestScoreRatio + 1e-9) { better = true; }
              else if (std::abs(currentScoreRatio - bestScoreRatio) < 1e-9) { 
                  if (bestVertex == -1) { better = true; }
                  else { 
                       if (v_sample < ctx.conf.size() && bestVertex < ctx.conf.size()) {
                           if (ctx.conf[v_sample] > ctx.conf[bestVertex]) { better = true; }
                           else if (ctx.conf[v_sample] == ctx.conf[bestVertex]) {
                                if (v_sample < ctx.age.size() && bestVertex < ctx.age.size()) {
                                   if (ctx.age[v_sample] > ctx.age[bestVertex]) { better = true; }
                                } 
                           }
                       }
                  }
              }
          }
          if(better) { bestScoreRatio = currentScoreRatio; bestVertex = v_sample; }
     }
     return bestVertex;
 }

 static int pickRandomFromList(const std::vector<int>& list, std::mt19937_64& rng) {
    if (list.empty()) return -1;
    std::uniform_int_distribution<int> dist(0, (int)list.size() - 1);
    return list[dist(rng)];
 }


 static int pickVertexToAdd(const Graph &g, const MWDSContext &ctx, int initially_uncovered_vertex) {
      int n = g.n;
      if (initially_uncovered_vertex < 0 || initially_uncovered_vertex >= n || initially_uncovered_vertex >= g.deg.size() || g.deg[initially_uncovered_vertex]<0) return -1; 

      int bestVertexToAdd = -1;
      double bestScoreRatio = -std::numeric_limits<double>::infinity();


      bool only_one_uncovered = (ctx.undominated_vertices_list.size() == 1 && 
                                 !ctx.undominated_vertices_list.empty() && 
                                 ctx.undominated_vertices_list[0] == initially_uncovered_vertex);
      if (ctx.undominated_vertices_list.size() == 0 && initially_uncovered_vertex < ctx.coverCount.size() && ctx.coverCount[initially_uncovered_vertex] == 0){
          only_one_uncovered = true; 
      }


      std::vector<int> neighborhood_candidates;
      if (initially_uncovered_vertex < g.N_v_closed.size()) {
          for (int candidate_v : g.N_v_closed[initially_uncovered_vertex]) {
              if (candidate_v >=0 && candidate_v < n && candidate_v < g.deg.size() && g.deg[candidate_v] >=0 && 
                  candidate_v < ctx.inSolution.size() && !ctx.inSolution[candidate_v]) { 
                  neighborhood_candidates.push_back(candidate_v);
              }
          }
      }
      std::sort(neighborhood_candidates.begin(), neighborhood_candidates.end());
      neighborhood_candidates.erase(std::unique(neighborhood_candidates.begin(), neighborhood_candidates.end()), neighborhood_candidates.end());


      for(int candidate_v : neighborhood_candidates) {
          bool allowed_by_cc = (candidate_v < ctx.conf.size() && ctx.conf[candidate_v] != 0); 

          if (!allowed_by_cc && only_one_uncovered) {
              allowed_by_cc = true; 
               if (isDebugEnabled) {std::cout << "DEBUG: pickVertexToAdd ignoring CC=0 for " << candidate_v << " (only one uncovered)" << std::endl;}
          }
          if (!allowed_by_cc) continue; 

          double currentScoreRatio = getScorePerWeight(g, ctx, candidate_v);
          bool better = false;
          if (currentScoreRatio > bestScoreRatio + 1e-9) { better = true; }
          else if (std::abs(currentScoreRatio - bestScoreRatio) < 1e-9) { 
              if (bestVertexToAdd == -1) { better = true; } 
              else { 
                   if (candidate_v < ctx.conf.size() && bestVertexToAdd < ctx.conf.size()) { 
                       if (ctx.conf[candidate_v] > ctx.conf[bestVertexToAdd]) { better = true; } 
                       else if (ctx.conf[candidate_v] == ctx.conf[bestVertexToAdd]) { 
                            if (candidate_v < ctx.age.size() && bestVertexToAdd < ctx.age.size()) { 
                               if (ctx.age[candidate_v] > ctx.age[bestVertexToAdd]) { better = true; } 
                            } } } } } 
          if(better) { bestScoreRatio = currentScoreRatio; bestVertexToAdd = candidate_v; }
      }

       if (bestVertexToAdd == -1 && only_one_uncovered) { 
             if (isDebugEnabled) {std::cout << "DEBUG: pickVertexToAdd fallback (only one uncovered, CC=0 forced override) activated." << std::endl;}
            bestScoreRatio = -std::numeric_limits<double>::infinity(); 
            for(int candidate_v : neighborhood_candidates) { 
                if (candidate_v < 0 || candidate_v >= n || candidate_v >= g.deg.size() || g.deg[candidate_v]<0 || candidate_v >= ctx.inSolution.size() || ctx.inSolution[candidate_v]) continue; 
                 double currentScoreRatio = getScorePerWeight(g, ctx, candidate_v);
                 bool better = false;
                 if (currentScoreRatio > bestScoreRatio + 1e-9) { better = true; }
                 else if (std::fabs(currentScoreRatio - bestScoreRatio) < 1e-9) { 
                      if (bestVertexToAdd == -1) { better = true; }
                      else { 
                           if (candidate_v < g.original_weight.size() && bestVertexToAdd < g.original_weight.size()){ 
                               if (g.original_weight[candidate_v] < g.original_weight[bestVertexToAdd]) { better = true; } 
                           } } }
                 if(better) { bestScoreRatio = currentScoreRatio; bestVertexToAdd = candidate_v; }
            }
       }
      return bestVertexToAdd;
  }

 static long long computeSolutionCost(const Graph &g_original, const std::vector<bool> &solution_vec) {
     long long current_cost = 0LL;
     int n_orig = g_original.n; 
     if (solution_vec.size() != (size_t)n_orig) {
         std::cerr << "ERROR: computeSolutionCost called with mismatched solution vector size! sol_size="<< solution_vec.size() << " n=" << n_orig << std::endl;
         return std::numeric_limits<long long>::max();
     }
     for(int v=0; v<n_orig; v++){
         if(solution_vec[v]) {
             if (v < g_original.original_weight.size()) { 
                 current_cost += g_original.original_weight[v];
             } else {
                  std::cerr << "错误: computeSolutionCost 访问顶点 " << v << " 的 original_weight 超出范围。" << std::endl;
                   return std::numeric_limits<long long>::max(); 
             }
         }
     }
     return current_cost;
 }


 //----------------------------------------------------
 // PART 6. DeepOpt 扰动 
 //----------------------------------------------------
 static void deepOptMWDS(const Graph &g, MWDSContext &ctx, 
                         long long &bestCostLocal,        
                         std::vector<bool> &bestSolLocal,  
                         const Graph &g_original,          
                         int Rcount,
                         int InnerDepth)
 {
     int n_reduced = g.n; 
      if (isDebugEnabled) {std::cout << "DEBUG: Entering DeepOpt..." << std::endl;}

     std::uniform_int_distribution<int> dist_remove_times(1, Rcount);
     int removal_iterations = dist_remove_times(ctx.local_rng);
      if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt removing elements for " << removal_iterations << " iterations." << std::endl;}

     std::vector<int> current_solution_nodes_in_reduced_graph;

     for(int i=0; i<removal_iterations; i++){
          current_solution_nodes_in_reduced_graph.clear();
          for(int v_iter=0; v_iter<n_reduced; v_iter++){
              if(v_iter < g.deg.size() && g.deg[v_iter]>=0 && v_iter < ctx.inSolution.size() && ctx.inSolution[v_iter]) { 
                  current_solution_nodes_in_reduced_graph.push_back(v_iter);
              } }
          if(current_solution_nodes_in_reduced_graph.empty()) {
                if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt removal iteration " << i+1 << ": Solution empty." << std::endl;}
               break; }

         std::uniform_int_distribution<int> dist_sol_idx(0, (int)current_solution_nodes_in_reduced_graph.size()-1);
         int chosen_v_for_N2_removal = current_solution_nodes_in_reduced_graph[dist_sol_idx(ctx.local_rng)];
         if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt removal iter " << i+1 << ": Chosen v = " << chosen_v_for_N2_removal << std::endl;}

          std::unordered_set<int> n2_neighborhood_in_solution;
          std::unordered_set<int> visited_for_n2; 
          std::queue<std::pair<int, int>> q_bfs; 

          if (chosen_v_for_N2_removal>=0 && chosen_v_for_N2_removal<n_reduced && chosen_v_for_N2_removal < g.deg.size() && g.deg[chosen_v_for_N2_removal]>=0) { 
                q_bfs.push({chosen_v_for_N2_removal, 0});
                visited_for_n2.insert(chosen_v_for_N2_removal);

                while(!q_bfs.empty()){
                    auto [curr_node_bfs, dist_bfs] = q_bfs.front();
                    q_bfs.pop();

                    if (curr_node_bfs < ctx.inSolution.size() && ctx.inSolution[curr_node_bfs]){
                        n2_neighborhood_in_solution.insert(curr_node_bfs);
                    }

                    if (dist_bfs < 2 && curr_node_bfs < g.adj.size()){ 
                        for (int neighbor_bfs : g.adj[curr_node_bfs]){
                            if(neighbor_bfs >= 0 && neighbor_bfs < n_reduced && neighbor_bfs < g.deg.size() && g.deg[neighbor_bfs] >= 0 &&
                               visited_for_n2.find(neighbor_bfs) == visited_for_n2.end())
                            {
                                visited_for_n2.insert(neighbor_bfs);
                                q_bfs.push({neighbor_bfs, dist_bfs + 1});
                            }
                        }
                    }
                }
          }

          if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt removing N2[" << chosen_v_for_N2_removal << "] intersect D (in reduced graph): ";}
         bool removed_in_iter = false;
         for(int vertex_to_remove_deepopt : n2_neighborhood_in_solution) {
             if (vertex_to_remove_deepopt >= 0 && vertex_to_remove_deepopt < ctx.inSolution.size() && ctx.inSolution[vertex_to_remove_deepopt]){
                 if (isDebugEnabled) {std::cout << vertex_to_remove_deepopt << " ";}
                 removeFromSolution(g, ctx, vertex_to_remove_deepopt); 
                 removed_in_iter = true;
             }
         }
         if (isDebugEnabled) { if (!removed_in_iter) std::cout << "(none)"; std::cout << std::endl;}
     } 
     
     calculateAllVertexScores(g, ctx);
     updateUndominatedAndRedundantLists(g, ctx);

     std::vector<bool> locked(n_reduced, false);
     int locked_count = 0;
     for(int v_lock=0; v_lock<n_reduced; v_lock++){
         if(v_lock < g.deg.size() && g.deg[v_lock]>=0 && v_lock < ctx.inSolution.size() && ctx.inSolution[v_lock]) { 
              locked[v_lock] = true;
              locked_count++;
         } }
      if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt locked " << locked_count << " vertices." << std::endl;}

     int inner_steps = 0;
     int inner_steps_without_improvement = 0;

     while(inner_steps_without_improvement < InnerDepth) {
          inner_steps++;
          bool improved_in_inner = false;

         while(!isCompleteCover(g, ctx)) { 
             int uncovered = pickRandomFromList(ctx.undominated_vertices_list, ctx.local_rng);
             if(uncovered < 0) {
                  std::cerr << "WARNING: DeepOpt repair loop: No uncovered vertex found despite !isCompleteCover.\n";
                  goto end_deepopt_search; 
             }
             int vertex_to_add_deepopt = pickVertexToAdd(g, ctx, uncovered); 

             if(vertex_to_add_deepopt < 0) { 
                  int fallback_add_deepopt = -1;
                  if(uncovered < g.N_v_closed.size()){
                      for(int fb_cand : g.N_v_closed[uncovered]){
                          if(fb_cand >=0 && fb_cand < n_reduced && fb_cand < g.deg.size() && g.deg[fb_cand] >=0 &&
                             fb_cand < ctx.inSolution.size() && !ctx.inSolution[fb_cand] &&
                             fb_cand < locked.size() && !locked[fb_cand]){
                                fallback_add_deepopt = fb_cand; break; 
                             }
                      }
                  }
                 if (fallback_add_deepopt != -1) vertex_to_add_deepopt = fallback_add_deepopt;
                 else {
                     std::cerr << "警告: DeepOpt 无法修复解（找不到要添加的未锁定顶点）。 Exiting DeepOpt inner search.\n";
                     goto end_deepopt_search; 
                 }
             }

             if (vertex_to_add_deepopt >= 0 && vertex_to_add_deepopt < locked.size() && !locked[vertex_to_add_deepopt]) { 
                 addToSolution(g, ctx, vertex_to_add_deepopt); 
             } else if (vertex_to_add_deepopt < 0) {
                  std::cerr << "警告: DeepOpt 修复循环中 vertex_to_add 无效。\n";
                  goto end_deepopt_search; 
             } 
             
             calculateAllVertexScores(g, ctx); 
             updateUndominatedAndRedundantLists(g, ctx); 

             bool innerRemoved = true;
             while(innerRemoved) {
                  innerRemoved = false;
                  updateUndominatedAndRedundantLists(g, ctx); 
                  int redundant_unlocked_node = -1;
                  for(int vr : ctx.redundant_vertices_list){ 
                       if(vr < locked.size() && !locked[vr] && vr < ctx.inSolution.size() && ctx.inSolution[vr]) { 
                           redundant_unlocked_node = vr; break; 
                       }
                  }
                  if(redundant_unlocked_node != -1){
                        if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt removing redundant unlocked " << redundant_unlocked_node << " during repair." << std::endl;}
                       removeFromSolution(g, ctx, redundant_unlocked_node); 
                       calculateAllVertexScores(g, ctx); 
                       innerRemoved = true; 
                  } 
             }
             updateUndominatedAndRedundantLists(g, ctx); 
         } 


         if (isCompleteCover(g, ctx)) { 
             std::vector<bool> current_sol_reverted_deepopt = ctx.inSolution;
             revertSolution(g, current_sol_reverted_deepopt); 
             long long currentCost = computeSolutionCost(g_original, current_sol_reverted_deepopt); 
             
             if(currentCost < bestCostLocal) {
                 bestCostLocal = currentCost;
                 bestSolLocal = ctx.inSolution; 
                 inner_steps_without_improvement = 0;
                 improved_in_inner = true;
                  if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt found new best local cost: " << bestCostLocal << std::endl;}
                 { 
                     std::lock_guard<std::mutex> lock(best_sol_mutex);
                     if (currentCost < globalBestCost.load(std::memory_order_relaxed)) {
                          if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt updating global best cost to: " << currentCost << std::endl;}
                         globalBestCost.store(currentCost, std::memory_order_relaxed);
                         globalBestSol = ctx.inSolution; 
                     } 
                 }
             }
         } else {
             std::cerr << "WARNING: DeepOpt inner search: Solution infeasible after repair attempt.\n";
         }
         if (!improved_in_inner) { inner_steps_without_improvement++; }

         calculateAllVertexScores(g, ctx); 
         updateUndominatedAndRedundantLists(g, ctx); 

          std::vector<int> unlocked_in_solution;
          for(int v_unlocked=0; v_unlocked<n_reduced; v_unlocked++){
             if(v_unlocked < g.deg.size() && g.deg[v_unlocked]>=0 && v_unlocked < ctx.inSolution.size() && ctx.inSolution[v_unlocked] && v_unlocked < locked.size() && !locked[v_unlocked]) { 
                 unlocked_in_solution.push_back(v_unlocked);
             } }

         if(!unlocked_in_solution.empty()) { 
             int remove1 = pickBestAmongSample(g, ctx, unlocked_in_solution, 100, true); 
             if(remove1 >= 0) {
                  if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt removing unlocked (score based): " << remove1 << std::endl;}
                 removeFromSolution(g, ctx, remove1); 
                  auto it_rem = std::remove(unlocked_in_solution.begin(), unlocked_in_solution.end(), remove1);
                  if (it_rem != unlocked_in_solution.end()) unlocked_in_solution.erase(it_rem, unlocked_in_solution.end());
             } else { if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt remove unlocked (score) failed pick." << std::endl;}}
          } else { if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt remove unlocked (score): no candidates." << std::endl;}}
          
          calculateAllVertexScores(g, ctx); 
          updateUndominatedAndRedundantLists(g, ctx); 

         if(!unlocked_in_solution.empty()) { 
             int remove2 = pickRandomFromList(unlocked_in_solution, ctx.local_rng);
             if (remove2 >=0) {
                if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt removing unlocked (random): " << remove2 << std::endl;}
                removeFromSolution(g, ctx, remove2); 
             }
         } else { if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt remove unlocked (random): no candidates." << std::endl;}}
          
          calculateAllVertexScores(g, ctx); 
          updateUndominatedAndRedundantLists(g, ctx);
          updateFreqAndAge(g, ctx); 
     } 

     end_deepopt_search:; 
      if (isDebugEnabled) {std::cout << "DEBUG: Exiting DeepOpt. Steps=" << inner_steps << ", Steps w/o improve=" << inner_steps_without_improvement << std::endl;}
 }


 //----------------------------------------------------
 // PART 7. 主局部搜索算法 
 //----------------------------------------------------
 static std::pair<long long, std::vector<bool>> deepOptMWDS_solve_instance(
     const Graph &g_original_const, 
     double cutoffTime,
     unsigned int random_seed)
 {
     Graph g = g_original_const; 
     int n = g.n; 
     MWDSContext ctx(n, random_seed);

     ctx.inSolution = ConstructDS(g, ctx); 

     calculateAllVertexScores(g, ctx);
     updateUndominatedAndRedundantLists(g, ctx);
     initConfiguration(ctx,n); 

     long long bestCostLocal = std::numeric_limits<long long>::max();
     std::vector<bool> bestSolLocal = ctx.inSolution; 

     std::fill(ctx.coverCount.begin(), ctx.coverCount.end(), 0);
     for(int v_sol_idx = 0; v_sol_idx < n; ++v_sol_idx) {
         if (v_sol_idx < ctx.inSolution.size() && ctx.inSolution[v_sol_idx]) {
             updateCoverCountForVertexAndNeighbors(g, ctx, v_sol_idx, 1);
         }
     }
     calculateAllVertexScores(g,ctx); 
     updateUndominatedAndRedundantLists(g,ctx); 

     if (isCompleteCover(g, ctx)) { 
          std::vector<bool> initial_reverted_sol = ctx.inSolution;
          revertSolution(g, initial_reverted_sol); 
          bestCostLocal = computeSolutionCost(g_original_const, initial_reverted_sol); 
          bestSolLocal = ctx.inSolution; 
          if (isDebugEnabled) {std::cout << "DEBUG: Initial solution feasible (on reduced graph). Reverted Cost=" << bestCostLocal << ", Size=" << std::count(initial_reverted_sol.begin(), initial_reverted_sol.end(), true) << std::endl;}
     } else {
           std::cerr << "警告: ConstructDS 未能在线程 " << std::this_thread::get_id() << " 中生成可行解！\n";
           bestCostLocal = std::numeric_limits<long long>::max();
           bestSolLocal.assign(n, false);
     }

      if (bestCostLocal != std::numeric_limits<long long>::max()) {
           std::lock_guard<std::mutex> lock(best_sol_mutex);
           if (bestCostLocal < globalBestCost.load(std::memory_order_relaxed)){
                if (isDebugEnabled) {std::cout << "DEBUG: Thread " << std::this_thread::get_id() << " updating global best from initial. Cost=" << bestCostLocal << std::endl;}
                globalBestCost.store(bestCostLocal, std::memory_order_relaxed);
                globalBestSol = ctx.inSolution; 
           }
      }

     auto start_time = std::chrono::steady_clock::now();
     long long steps_without_improvement = 0;
     const long long OuterDepthTrigger = 20000; 
     const int DeepOpt_Rcount = 5;      
     const int DeepOpt_InnerDepth = 50; 
     const int bms_sample_size = 100;

     long long gap_val = 1LL; 
     auto calculate_current_gap = [&]() -> long long {
        long long current_sol_cost_reduced_weights = 0;
        int count_in_sol = 0;
        long long max_w_in_sol_reduced = 0;
        for(int v_gap_idx=0; v_gap_idx<n; ++v_gap_idx){
            if(v_gap_idx < g.deg.size() && g.deg[v_gap_idx]>=0 && v_gap_idx < ctx.inSolution.size() && ctx.inSolution[v_gap_idx]) {
                if (v_gap_idx < g.weight.size()){ 
                    long long current_w = g.weight[v_gap_idx]; 
                    if (current_w > 0) { 
                        current_sol_cost_reduced_weights += current_w; 
                        max_w_in_sol_reduced = std::max(max_w_in_sol_reduced, current_w); 
                        count_in_sol++; 
                    } 
                } 
            }
        }
        long long current_m_reduced = 0; 
        for(int i=0; i<g.n; ++i) if(i < g.deg.size() && g.deg[i]>0) current_m_reduced+=g.deg[i]; // Ensure g.deg access is safe
        current_m_reduced /= 2;

        double density = (n > 1 && (static_cast<double>(n)-1.0) > 1e-9 ) ? static_cast<double>(current_m_reduced * 2.0) / (static_cast<double>(n) * (static_cast<double>(n)-1.0)) : 0.0;
        long long new_gap = 1LL;
        if (count_in_sol > 0) { 
            new_gap = (density < 0.07) ? max_w_in_sol_reduced : (current_sol_cost_reduced_weights / count_in_sol); 
        }
        return std::max(1LL, new_gap);
     };
     gap_val = calculate_current_gap();
      if (isDebugEnabled) {std::cout << "DEBUG: Initial Gap = " << gap_val << " (based on reduced graph)" << std::endl;}

     long long iteration = 0;
     while(true) {
         auto current_time_loop = std::chrono::steady_clock::now();
         double elapsed_seconds = std::chrono::duration<double>(current_time_loop - start_time).count();
         if(elapsed_seconds >= cutoffTime) {
              if (isDebugEnabled) {std::cout << "DEBUG: Thread " << std::this_thread::get_id() << " reached cutoff." << std::endl;}
              break; }
         iteration++;
         steps_without_improvement++; 


         calculateAllVertexScores(g, ctx);
         updateUndominatedAndRedundantLists(g, ctx);

         if(isCompleteCover(g, ctx)) { 
             bool removed_in_step1 = true;
             while(removed_in_step1 && !ctx.redundant_vertices_list.empty()) { 
                 removed_in_step1 = false;
                 int vertex_to_remove_step1 = -1; 
                 long long max_w_step1 = -1LL;
                 
                 std::vector<int> sample_for_step1_remove;
                 if (ctx.redundant_vertices_list.size() <= (size_t)bms_sample_size) {
                     sample_for_step1_remove = ctx.redundant_vertices_list;
                 } else {
                     std::sample(ctx.redundant_vertices_list.begin(), ctx.redundant_vertices_list.end(),
                                 std::back_inserter(sample_for_step1_remove),
                                 bms_sample_size, ctx.local_rng);
                 }

                 for(int vv_step1 : sample_for_step1_remove) { 
                     if (vv_step1 < g.weight.size() && g.weight[vv_step1] > max_w_step1) { 
                         max_w_step1 = g.weight[vv_step1]; 
                         vertex_to_remove_step1 = vv_step1; 
                     } 
                 }

                 if(vertex_to_remove_step1 != -1) {
                      if (isDebugEnabled) {std::cout << "DEBUG: Step 1 removing redundant " << vertex_to_remove_step1 << std::endl;}
                      removeFromSolution(g, ctx, vertex_to_remove_step1); 
                      calculateAllVertexScores(g, ctx); 
                      updateUndominatedAndRedundantLists(g, ctx); 
                      removed_in_step1 = true; 
                 }
             }
             if (isCompleteCover(g, ctx)) {
                 std::vector<bool> current_reverted_sol_step1 = ctx.inSolution;
                 revertSolution(g, current_reverted_sol_step1);
                 long long currentCost_step1 = computeSolutionCost(g_original_const, current_reverted_sol_step1); 

                 if(currentCost_step1 < bestCostLocal) {
                      if (isDebugEnabled) {std::cout << "DEBUG: New best local (after Step 1): " << currentCost_step1 << " (Iter " << iteration << ")" << std::endl;}
                     bestCostLocal = currentCost_step1;
                     bestSolLocal = ctx.inSolution; 
                     steps_without_improvement = 0; 
                     gap_val = calculate_current_gap(); 
                      if (isDebugEnabled) {std::cout << "DEBUG: Updated Gap = " << gap_val << std::endl;}
                     { std::lock_guard<std::mutex> lock(best_sol_mutex);
                        if (currentCost_step1 < globalBestCost.load(std::memory_order_relaxed)) {
                             if (isDebugEnabled) {std::cout << "DEBUG: Thread " << std::this_thread::get_id() << " updating global best: " << currentCost_step1 << std::endl;}
                             globalBestCost.store(currentCost_step1, std::memory_order_relaxed);
                             globalBestSol = ctx.inSolution; 
                        }
                     }
                 }
             }
         } 

         std::vector<bool> temp_reverted_sol_step2 = ctx.inSolution;
         revertSolution(g, temp_reverted_sol_step2);
         long long costForGapCheck_original_weights = computeSolutionCost(g_original_const, temp_reverted_sol_step2);

         while (costForGapCheck_original_weights + gap_val >= bestCostLocal) {
              if (ctx.inSolution.empty() || std::count(ctx.inSolution.begin(), ctx.inSolution.end(), true) == 0) {
                   if (isDebugEnabled){std::cout << "DEBUG: Gap removal stopped: solution empty." << std::endl;} break;
              }
              std::vector<int> removal_candidates_step2;
              for(int v_cand_idx=0; v_cand_idx<n; ++v_cand_idx){ if(v_cand_idx < g.deg.size() && g.deg[v_cand_idx]>=0 && v_cand_idx < ctx.inSolution.size() && ctx.inSolution[v_cand_idx]) { removal_candidates_step2.push_back(v_cand_idx); } }
              if(removal_candidates_step2.empty()) { if (isDebugEnabled){std::cout << "DEBUG: Gap removal stopped: no candidates in solution." << std::endl;} break; }

              calculateAllVertexScores(g, ctx); 
              int vertex_to_remove_step2 = pickBestAmongSample(g, ctx, removal_candidates_step2, bms_sample_size, true); 
              if(vertex_to_remove_step2 < 0) { if (isDebugEnabled){std::cout << "DEBUG: Gap removal stopped: pick failed." << std::endl;} break; }

              if (isDebugEnabled) {std::cout << "DEBUG: Step 2 (Gap removal): removing " << vertex_to_remove_step2 << std::endl;}
              removeFromSolution(g, ctx, vertex_to_remove_step2); 
              temp_reverted_sol_step2 = ctx.inSolution;
              revertSolution(g, temp_reverted_sol_step2);
              costForGapCheck_original_weights = computeSolutionCost(g_original_const, temp_reverted_sol_step2);
              if (std::count(ctx.inSolution.begin(), ctx.inSolution.end(), true) == 0) break; 
         }
         calculateAllVertexScores(g, ctx); 
         updateUndominatedAndRedundantLists(g, ctx);


         while (!isCompleteCover(g, ctx)) { 
             int uncovered_v_step3 = pickRandomFromList(ctx.undominated_vertices_list, ctx.local_rng);
             if(uncovered_v_step3 < 0) {
                  std::cerr << "WARNING: Main loop Step 3: No uncovered vertex found. Breaking add loop.\n";
                  break; 
             }
             calculateAllVertexScores(g, ctx); 
             int v1_to_add_step3 = pickVertexToAdd(g, ctx, uncovered_v_step3); 
             
             if(v1_to_add_step3 < 0) {
                   int fallback_add_step3 = -1;
                   if(uncovered_v_step3 < g.N_v_closed.size()){
                       long long min_w_fb = std::numeric_limits<long long>::max();
                       for(int fb_cand : g.N_v_closed[uncovered_v_step3]){
                           if(fb_cand >=0 && fb_cand < n && fb_cand < g.deg.size() && g.deg[fb_cand]>=0 &&
                              fb_cand < ctx.inSolution.size() && !ctx.inSolution[fb_cand] &&
                              fb_cand < ctx.conf.size() && ctx.conf[fb_cand] != 0 && 
                              fb_cand < g.weight.size() && g.weight[fb_cand] < min_w_fb){
                               min_w_fb = g.weight[fb_cand];
                               fallback_add_step3 = fb_cand;
                           }
                       }
                   }
                  if(fallback_add_step3 != -1) v1_to_add_step3 = fallback_add_step3;
                  else {
                     std::cerr << "WARNING: Main loop Step 3 Add failed completely for uncovered " << uncovered_v_step3 << ". Breaking add loop.\n";
                     break; 
                  }
             }

             std::vector<bool> sol_if_added_step3 = ctx.inSolution;
             if (v1_to_add_step3 < sol_if_added_step3.size()) sol_if_added_step3[v1_to_add_step3] = true; else continue; 
             std::vector<bool> reverted_sol_if_added_step3 = sol_if_added_step3;
             revertSolution(g, reverted_sol_if_added_step3);
             long long cost_if_added_original_weights = computeSolutionCost(g_original_const, reverted_sol_if_added_step3); 

             if(cost_if_added_original_weights + gap_val <= bestCostLocal) { 
                  if (isDebugEnabled) {std::cout << "DEBUG: Add mode 1 (cost_gap): Adding " << v1_to_add_step3 << std::endl;}
                  addToSolution(g, ctx, v1_to_add_step3); 
             } else { 
                 std::vector<int> exchange_candidates_step3;
                 for(int v_exch_idx=0; v_exch_idx<n; ++v_exch_idx){ if(v_exch_idx < g.deg.size() && g.deg[v_exch_idx]>=0 && v_exch_idx < ctx.inSolution.size() && ctx.inSolution[v_exch_idx]) { exchange_candidates_step3.push_back(v_exch_idx); } }
                 
                 int u2_to_remove_exchange_step3 = -1;
                 if (!exchange_candidates_step3.empty()) {
                     calculateAllVertexScores(g, ctx); 
                     u2_to_remove_exchange_step3 = pickBestAmongSample(g, ctx, exchange_candidates_step3, bms_sample_size, true); 
                 }

                 bool exchange_improves = false;
                 if (u2_to_remove_exchange_step3 >= 0 && u2_to_remove_exchange_step3 != v1_to_add_step3) { 
                     if (u2_to_remove_exchange_step3 < ctx.current_vertex_scores.size() && v1_to_add_step3 < ctx.current_vertex_scores.size() &&
                         v1_to_add_step3 < g.weight.size() && u2_to_remove_exchange_step3 < g.weight.size() && g.weight[v1_to_add_step3] > 0 && g.weight[u2_to_remove_exchange_step3] > 0 ) { // Ensure weights are positive for division
                        
                        double score_add_v1 = static_cast<double>(ctx.current_vertex_scores[v1_to_add_step3]) / g.weight[v1_to_add_step3];
                        double score_remove_u2 = static_cast<double>(ctx.current_vertex_scores[u2_to_remove_exchange_step3]) / g.weight[u2_to_remove_exchange_step3]; // score for removal is negative if beneficial

                        // Reference: compare(-v_Score[v_remove], v_Cost[v_remove], v_Score[v_add], v_Cost[v_add]) < 0
                        // This means score_add/cost_add > -score_remove/cost_remove
                        // Or score_add/cost_add + score_remove/cost_remove > 0
                        if (score_add_v1 + score_remove_u2 > 1e-9 ) { // Check if adding v1 is better than keeping u2
                             exchange_improves = true;
                        }
                     }
                 }


                 if (exchange_improves) { 
                      if (isDebugEnabled) {std::cout << "DEBUG: Add mode 2 (exchange): Exchanging " << u2_to_remove_exchange_step3 << " with " << v1_to_add_step3 << std::endl;}
                      removeFromSolution(g, ctx, u2_to_remove_exchange_step3);
                      addToSolution(g, ctx, v1_to_add_step3); 
                 } else { 
                     int undominated_count_val = std::max(1, (int)ctx.undominated_vertices_list.size());
                     std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
                     
                     long long v1_actual_score_if_added = 0;
                     if(v1_to_add_step3 < ctx.current_vertex_scores.size()) v1_actual_score_if_added = ctx.current_vertex_scores[v1_to_add_step3];


                     double probability_threshold = (ctx.undominated_vertices_list.empty() || v1_actual_score_if_added <=0 ) ? 0.0 :
                                                  (static_cast<double>(v1_actual_score_if_added) / static_cast<double>(undominated_count_val * ctx.freq_or_penalty_weight[uncovered_v_step3] ) ); // Approximation
                                                  // The reference uses v_Score_Real[v_add] which is number of items it covers.
                                                  // Let's simplify: if it covers at least one currently undominated vertex (score > 0), then 1.0/undom_count
                     if (v1_actual_score_if_added > 0) probability_threshold = 1.0 / static_cast<double>(undominated_count_val);
                     else probability_threshold = 0.0;


                     if (prob_dist(ctx.local_rng) <= probability_threshold) {
                          if (isDebugEnabled) {std::cout << "DEBUG: Add mode 3 (prob): Adding " << v1_to_add_step3 << " (Prob=" << probability_threshold << ")" << std::endl;}
                         addToSolution(g, ctx, v1_to_add_step3); 
                     } else {
                          if (isDebugEnabled) {std::cout << "DEBUG: Add mode 3 (prob): Skipped adding " << v1_to_add_step3 << ". Breaking add loop." << std::endl;}
                           // break; 
                     }
                 } 
             } 
             calculateAllVertexScores(g, ctx); 
             updateUndominatedAndRedundantLists(g, ctx); 
         } 


         updateFreqAndAge(g, ctx); 

         if(steps_without_improvement >= OuterDepthTrigger) {
              if (isDebugEnabled) {std::cout << "DEBUG: Triggering DeepOpt (Iter " << iteration << ")" << std::endl;}
             deepOptMWDS(g, ctx, bestCostLocal, bestSolLocal, g_original_const, DeepOpt_Rcount, DeepOpt_InnerDepth);
             steps_without_improvement = 0; 
             std::fill(ctx.coverCount.begin(), ctx.coverCount.end(), 0);
             for(int v_sol_idx = 0; v_sol_idx < n; ++v_sol_idx) {
                 if (v_sol_idx < ctx.inSolution.size() && ctx.inSolution[v_sol_idx]) {
                    updateCoverCountForVertexAndNeighbors(g, ctx, v_sol_idx, 1);
                 }
             }
             calculateAllVertexScores(g, ctx);
             updateUndominatedAndRedundantLists(g, ctx);
             gap_val = calculate_current_gap(); 

             std::vector<bool> sol_after_deepopt_reverted = ctx.inSolution;
             revertSolution(g, sol_after_deepopt_reverted);
             long long cost_after_deepopt = computeSolutionCost(g_original_const, sol_after_deepopt_reverted);
             if (isDebugEnabled) {std::cout << "DEBUG: DeepOpt finished. Current reverted cost=" << cost_after_deepopt << std::endl;}
             if (cost_after_deepopt < bestCostLocal) {
                 bestCostLocal = cost_after_deepopt;
                 bestSolLocal = ctx.inSolution; 
             }
         }
     } 

     return {bestCostLocal, bestSolLocal};
 }


//----------------------------------------------------
// PART 8. 主函数 
//----------------------------------------------------
int main(int argc, char** argv) {
    const char* debug_env = std::getenv("MWDS_DEBUG");
    if (debug_env != nullptr && (std::strcmp(debug_env, "1") == 0 || std::strcmp(debug_env, "true") == 0 || std::strcmp(debug_env, "TRUE") == 0 ) ) {
        isDebugEnabled = true;
        // Debug mode message will be printed conditionally later if not --stdout
    } else {
        isDebugEnabled = false;
    }

    std::string graphFile;
    double cutoff = 3600.0;
    int num_threads = 1;
    // GraphFormat format = GraphFormat::AUTO_DETECT; // format is handled by loadGraph directly
    bool auto_cutoff = false;
    bool output_to_stdout_json = false;
    unsigned int master_seed_to_use = 0;
    bool seed_was_provided_by_user = false;

    std::vector<std::string> positional_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg_str = argv[i];
        if (arg_str == "--stdout") {
            output_to_stdout_json = true;
        } else {
            positional_args.push_back(arg_str);
        }
    }

    if (positional_args.empty()) {
        std::cerr << "用法: " << argv[0] << " <graph_file> [<cutoff_seconds | auto>] [<num_threads>] [<seed>] [--stdout]\n";
        std::cerr << "  例如: " << argv[0] << " data/graph.clq 60 4 12345 --stdout\n";
        std::cerr << "  例如: " << argv[0] << " data/large_graph.txt auto 8\n";
        std::cerr << "  要启用调试，请设置环境变量: export MWDS_DEBUG=1 (或 set MWDS_DEBUG=1 在 Windows 上)\n";
        std::cerr << "  解将以 JSON 格式保存在 'solutions/' 目录下。\n";
        return 1;
    }

    graphFile = positional_args[0];

    if (positional_args.size() > 1) {
        std::string cutoff_str = positional_args[1];
        std::string temp_cutoff_str_lower = cutoff_str;
        std::transform(temp_cutoff_str_lower.begin(), temp_cutoff_str_lower.end(), temp_cutoff_str_lower.begin(),
                       [](unsigned char c){ return std::tolower(c); });

        if (temp_cutoff_str_lower == "auto" || temp_cutoff_str_lower == "automatic") {
            auto_cutoff = true;
        } else {
            try {
                cutoff = std::stod(cutoff_str);
                if (cutoff <= 0) {
                    std::cerr << "错误: 截止时间必须为正数。\n";
                    return 1;
                }
            } catch (const std::invalid_argument& ia) {
                std::cerr << "错误: 无效的截止时间值 '" << cutoff_str << "'. 请输入秒数或 'auto'.\n";
                return 1;
            } catch (const std::out_of_range& oor) {
                 std::cerr << "错误: 截止时间值超出范围 '" << cutoff_str << "'.\n";
                 return 1;
            } catch (...) {
                 std::cerr << "错误: 解析截止时间时发生未知错误 '" << cutoff_str << "'.\n";
                 return 1;
            }
        }
    }

    if (positional_args.size() > 2) {
        try {
            num_threads = std::stoi(positional_args[2]);
             if (num_threads <= 0) {
                std::cerr << "错误: 线程数必须为正数。\n";
                return 1;
            }
        } catch (const std::invalid_argument& ia) {
            std::cerr << "错误: 无效的线程数值 '" << positional_args[2] << "'.\n";
            return 1;
        } catch (const std::out_of_range& oor) {
            std::cerr << "错误: 线程数值超出范围 '" << positional_args[2] << "'.\n";
            return 1;
        } catch(...) {
             std::cerr << "错误: 解析线程数时发生未知错误 '" << positional_args[2] << "'.\n";
             return 1;
         }
    }

    if (positional_args.size() > 3) {
        std::string seed_str = positional_args[3];
        try {
            unsigned long temp_seed = std::stoul(seed_str);
            if (temp_seed > std::numeric_limits<unsigned int>::max()) {
                 std::cerr << "错误: 种子值超出范围 '" << seed_str << "'. 最大允许值: " << std::numeric_limits<unsigned int>::max() << ".\n";
                 return 1;
            }
            master_seed_to_use = static_cast<unsigned int>(temp_seed);
            seed_was_provided_by_user = true;
        } catch (const std::invalid_argument& ia) {
            std::cerr << "错误: 无效的种子数值 '" << seed_str << "'.\n";
            return 1;
        } catch (const std::out_of_range& oor) {
            std::cerr << "错误: 种子数值超出范围 '" << seed_str << "'.\n";
            return 1;
        }
    }

    if (!seed_was_provided_by_user) {
        master_seed_to_use = static_cast<unsigned int>(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    global_rng.seed(master_seed_to_use); // Seed the global RNG

    if (isDebugEnabled && !output_to_stdout_json) {
        std::cout << "********** DEBUG MODE ENABLED **********" << std::endl;
        std::cout << "DEBUG: Using master seed: " << master_seed_to_use << std::endl;
    }

    if (graphFile.empty()) { std::cerr << "错误: 未指定图文件。\n"; return 1; }

    Graph g_original_main;
    try {
        g_original_main = loadGraph(graphFile, GraphFormat::AUTO_DETECT); // GraphFormat::AUTO_DETECT is default for loadGraph
        if (g_original_main.n <= 0 && !(graphFile.find("empty") != std::string::npos) ) { // Allow specifically named "empty" graphs for testing
             throw std::runtime_error("图加载后没有顶点。");
        }
        // Ensure all g_original_main vectors are correctly sized if n=0 or after loadGraph
        if (g_original_main.revert_map.size() != (size_t)g_original_main.n) g_original_main.revert_map.assign(g_original_main.n, -1);
        if (g_original_main.original_deg.size() != (size_t)g_original_main.n) {
            g_original_main.original_deg.resize(g_original_main.n);
             for(int i = 0; i < g_original_main.n; ++i) {
                 if (i < g_original_main.adj.size()) { g_original_main.original_deg[i] = g_original_main.adj[i].size(); }
                 else { g_original_main.original_deg[i] = 0;}
             }
        }
         if (g_original_main.weight.size() != (size_t)g_original_main.n) g_original_main.weight.resize(g_original_main.n, 1LL);
         if (g_original_main.original_weight.size() != (size_t)g_original_main.n) g_original_main.original_weight.resize(g_original_main.n, 1LL);
         if (g_original_main.N_v_closed.size() != (size_t)g_original_main.n) g_original_main.N_v_closed.resize(g_original_main.n);


    }
    catch (const std::exception& e) { std::cerr << "加载图时出错: " << e.what() << std::endl; return 1; }

    if (!output_to_stdout_json) {
        std::cout << "图已加载: n=" << g_original_main.n << ", m=" << g_original_main.m << std::endl;
    }

    if (auto_cutoff) {
        long long n_val = g_original_main.n;
        long long m_val = g_original_main.m;
        const double base_time_seconds = 10.0;
        const double max_cutoff_seconds = 12.0 * 3600.0;
        const double edge_to_vertex_ratio = (m_val == 0 && n_val > 1) ? 1.0 : (m_val == 0 && n_val <=1) ? 500.0 : 500.0 ; // Avoid division by zero, adjust if m=0
        const double scaling_factor = 0.06;

        long long effective_n = n_val + static_cast<long long>(static_cast<double>(m_val) / (edge_to_vertex_ratio == 0 ? 1.0 : edge_to_vertex_ratio)); // Avoid div by zero
        double calculated_cutoff = base_time_seconds + scaling_factor * static_cast<double>(effective_n);
        cutoff = std::min(calculated_cutoff, max_cutoff_seconds);
        cutoff = std::max(cutoff, base_time_seconds);
        if (!output_to_stdout_json) {
            std::cout << "自动计算截止时间: " << std::fixed << std::setprecision(2) << cutoff << "s "
                      << "(基于 n=" << n_val << ", m=" << m_val << ", effective_n=" << effective_n << ")" << std::endl;
        }
    }

    if (!output_to_stdout_json) {
        std::cout << "启动 DeepOpt-MWDS 求解器 (使用 " << num_threads << " 个线程), 截止时间=" << std::fixed << std::setprecision(1) << cutoff << "s, 种子=" << master_seed_to_use << "..." << std::endl;
    }

    std::vector<std::future<std::pair<long long, std::vector<bool>>>> futures;
    globalBestCost.store(std::numeric_limits<long long>::max());
    if (g_original_main.n > 0) {
        globalBestSol.assign(g_original_main.n, false);
    } else {
        globalBestSol.clear();
    }


    auto start_time_solve = std::chrono::steady_clock::now();

    for(int i=0; i<num_threads; ++i) {
        unsigned int thread_seed = 0;
        {
            static std::mutex rng_mutex_for_thread_seeds;
            std::lock_guard<std::mutex> lock(rng_mutex_for_thread_seeds);
            thread_seed = global_rng(); // Get a seed for this thread from the master RNG
        }
        futures.push_back(std::async(std::launch::async, deepOptMWDS_solve_instance, std::cref(g_original_main), cutoff, thread_seed));
    }

    for(auto &fut : futures) {
        try {
            fut.get();
        } catch (const std::exception& e) {
            // This error should be printed regardless of --stdout, as it's an execution error
            std::cerr << "线程执行期间捕获到异常: " << e.what() << std::endl;
        }
    }

    auto end_time_solve = std::chrono::steady_clock::now();
    double solve_duration = std::chrono::duration<double>(end_time_solve - start_time_solve).count();

    if (!output_to_stdout_json) {
        std::cout << "所有线程完成 (" << std::fixed << std::setprecision(2) << solve_duration << "s elapsed)." << std::endl;
    }

    long long final_atomic_cost = globalBestCost.load(std::memory_order_relaxed);
    std::vector<bool> final_best_unreverted_sol_for_reduced_graph;
    int final_reverted_solution_size = 0;
    bool is_final_sol_valid = false; // Initialize

    Graph g_for_final_revert = g_original_main;
    std::vector<bool> temp_fixed_for_revert(g_original_main.n > 0 ? g_original_main.n : 0, false);
    if (g_original_main.n > 0) { // Only run reductions if graph is not empty
      if (isDebugEnabled && !output_to_stdout_json) std::cout << "DEBUG: Rerunning reductions on a fresh copy to get final revert map for the best solution..." << std::endl;
      while(applyReductionRules(g_for_final_revert, temp_fixed_for_revert));
    }


    {
        std::lock_guard<std::mutex> lock(best_sol_mutex);
        // The size of globalBestSol should correspond to the graph g on which ConstructDS (and thus reductions) was run.
        // This g is a copy of g_original_main, then reductions are applied.
        // So globalBestSol is for the *reduced* graph structure.
        // `g_for_final_revert` is the state of the graph *after* those same reductions would have been applied.
        // So, the size of `globalBestSol` should align with `g_for_final_revert.n`.
        size_t expected_sol_size = (g_for_final_revert.n > 0) ? (size_t)g_for_final_revert.n : 0;

        if (globalBestSol.empty() && final_atomic_cost != std::numeric_limits<long long>::max() && g_original_main.n > 0){
            if (!output_to_stdout_json) std::cerr << "警告: globalBestSol为空但找到了成本。这出乎意料。\n";
            final_best_unreverted_sol_for_reduced_graph.assign(expected_sol_size, false);
        } else if (!globalBestSol.empty() && globalBestSol.size() != expected_sol_size) {
            if (!output_to_stdout_json) {
                std::cerr << "警告: 全局最佳解的大小 (" << globalBestSol.size()
                          << ") 与最终缩减图大小 (" << expected_sol_size
                          << ") 不匹配。可能存在问题。尝试调整大小。\n";
            }
            final_best_unreverted_sol_for_reduced_graph.assign(expected_sol_size, false);
            size_t common_size = std::min(globalBestSol.size(), expected_sol_size);
            std::copy(globalBestSol.begin(), globalBestSol.begin() + common_size, final_best_unreverted_sol_for_reduced_graph.begin());

        } else {
           final_best_unreverted_sol_for_reduced_graph = globalBestSol;
        }
    }

    std::vector<bool> final_reverted_solution;
    if (g_original_main.n > 0) {
        final_reverted_solution.assign(g_original_main.n, false); // Initialize to original graph size
    }


    if (final_atomic_cost == std::numeric_limits<long long>::max() && g_original_main.n > 0) {
        if (!output_to_stdout_json) std::cerr << "未找到可行的解决方案。\n";
        // final_reverted_solution_size remains 0, is_final_sol_valid remains false
    } else {
        // If g_original_main.n is 0, final_atomic_cost should be 0.
        if (g_original_main.n == 0) {
            final_atomic_cost = 0;
            final_reverted_solution_size = 0;
            is_final_sol_valid = true; // Empty graph has a valid empty solution
            final_reverted_solution.clear();
        } else {
            // We have a potential solution (from globalBestSol, which is for the reduced graph g_for_final_revert)
            // final_best_unreverted_sol_for_reduced_graph should be the solution on g_for_final_revert's structure.
            // We need to map this back to g_original_main.
            // The `revertSolution` function expects a solution vector of size g_for_final_revert.n
            // and modifies it in place, then we need to ensure this reflects on a g_original_main.n sized vector.

            std::vector<bool> sol_to_revert = final_best_unreverted_sol_for_reduced_graph;
            if (sol_to_revert.size() != (size_t)g_for_final_revert.n && g_for_final_revert.n > 0) {
                 if (!output_to_stdout_json) {
                    std::cerr << "警告: 用于复原的解的大小 (" << sol_to_revert.size()
                              << ") 与参考缩减图大小 (" << g_for_final_revert.n
                              << ") 不匹配。尝试调整。\n";
                 }
                std::vector<bool> temp_sol(g_for_final_revert.n, false);
                size_t copy_len = std::min(sol_to_revert.size(), (size_t)g_for_final_revert.n);
                for(size_t i = 0; i < copy_len; ++i) temp_sol[i] = sol_to_revert[i];
                sol_to_revert = temp_sol;
            }


            // Initialize final_reverted_solution with fixed nodes from the reduction process on g_original_main
            // (those nodes that were determined to be in the solution by reduction rules)
            for(int i=0; i < g_original_main.n; ++i) {
                if (i < temp_fixed_for_revert.size() && temp_fixed_for_revert[i]) {
                    final_reverted_solution[i] = true;
                }
            }
            
            // Apply the solution from the reduced graph (sol_to_revert)
            // to the parts of final_reverted_solution that were not fixed.
            // This assumes ConstructDS and subsequent logic in deepOptMWDS_solve_instance operate on non-fixed vertices.
            // And globalBestSol (thus sol_to_revert) is for the *reduced* graph.
            // The revertSolution function is key here. It takes the graph state *after* reductions (g_for_final_revert)
            // and a solution on that reduced graph, and modifies the solution to incorporate reverted elements.

            // The solution `sol_to_revert` is for `g_for_final_revert`.
            // `revertSolution` will modify `sol_to_revert` based on `g_for_final_revert.revert_map`.
            // The items in `sol_to_revert` correspond to indices in `g_for_final_revert`.
            // These indices are the same as in `g_original_main`.
            if (g_for_final_revert.n > 0 && sol_to_revert.size() == (size_t)g_for_final_revert.n) {
                 revertSolution(g_for_final_revert, sol_to_revert);
                 // After revertSolution, sol_to_revert now contains the solution on the structure of g_for_final_revert
                 // but with implications from revert_map handled.
                 // We now merge this into our final_reverted_solution (which is g_original_main sized)
                 for(int i=0; i < g_for_final_revert.n; ++i) {
                     // If a vertex was part of the reduced graph and is in the solution after revert, add it.
                     // This might overwrite a `fixed` node if revertSolution changes its status, though typically fixed nodes remain.
                     if (i < sol_to_revert.size() && sol_to_revert[i]) {
                         final_reverted_solution[i] = true;
                     }
                 }
            } else if (g_for_final_revert.n == 0) {
                // All nodes were reduced and fixed. final_reverted_solution already has these from temp_fixed_for_revert.
            }


            if (final_reverted_solution.size() != (size_t)g_original_main.n && g_original_main.n > 0) {
                if (!output_to_stdout_json) std::cerr << "错误: 最终复原解的大小 (" << final_reverted_solution.size() << ") 与原始图大小 (" << g_original_main.n << ") 不匹配！\n";
                final_reverted_solution_size = 0;
                final_atomic_cost = std::numeric_limits<long long>::max(); // Mark as invalid
                is_final_sol_valid = false;
            } else {
                final_atomic_cost = computeSolutionCost(g_original_main, final_reverted_solution); // Recompute with original weights
                final_reverted_solution_size = 0;
                for(int v=0; v<g_original_main.n; v++){
                    if(v < final_reverted_solution.size() && final_reverted_solution[v]) {
                        final_reverted_solution_size++;
                    }
                }

                // Validate final_reverted_solution on g_original_main
                if (g_original_main.n > 0) {
                    is_final_sol_valid = true; // Assume valid initially
                    std::vector<int> final_cover_check(g_original_main.n, 0);
                    for(int v_fc=0; v_fc < g_original_main.n; ++v_fc) {
                        if (v_fc < final_reverted_solution.size() && final_reverted_solution[v_fc]) {
                            if (v_fc < g_original_main.N_v_closed.size()) { // Check bounds for N_v_closed
                                for (int neighbor_fc : g_original_main.N_v_closed[v_fc]) {
                                    if (neighbor_fc >=0 && neighbor_fc < g_original_main.n) {
                                        final_cover_check[neighbor_fc]++;
                                    }
                                }
                            }
                        }
                    }
                    for(int v_fc=0; v_fc < g_original_main.n; ++v_fc) {
                        // Only check nodes that were supposed to be covered (original_deg >= 0, implying not initially isolated and removed)
                        // The g.deg in applyReductionRules is used to mark removed nodes.
                        // For original graph validation, check all nodes that are not isolated if that's the problem definition.
                        // The paper implies all vertices must be dominated.
                        if (v_fc < final_cover_check.size() && final_cover_check[v_fc] == 0) {
                             // If a vertex has original degree 0, it must be in the solution itself to be covered.
                             // If original_deg[v_fc] == 0, then v_fc must be in final_reverted_solution.
                             // If N_v_closed[v_fc] only contains v_fc (isolated), it must pick itself.
                            bool should_be_covered = true; // Default: all nodes need covering
                            if (g_original_main.original_deg.size() > (size_t)v_fc && g_original_main.original_deg[v_fc] == 0 &&
                                (g_original_main.N_v_closed.size() <= (size_t)v_fc || g_original_main.N_v_closed[v_fc].empty() || (g_original_main.N_v_closed[v_fc].size()==1 && g_original_main.N_v_closed[v_fc][0] == v_fc) ) ) {
                                // This is an isolated vertex. It's covered if it's in the solution.
                                // The final_cover_check already reflects this. If it's 0, it means it wasn't in solution.
                            }

                            if (should_be_covered) { // Re-check condition more carefully
                                if (!output_to_stdout_json) {
                                     std::cerr << "错误: 最终复原解在原始图上不可行! Vertex " << v_fc << " (orig_deg=" << (v_fc < g_original_main.original_deg.size() ? std::to_string(g_original_main.original_deg[v_fc]) : "N/A") << ") is not covered." << std::endl;
                                }
                                is_final_sol_valid = false;
                                break;
                            }
                        }
                    }
                } else { // g_original_main.n == 0
                    is_final_sol_valid = true; // Empty solution is valid for empty graph
                }
            }
        }
    }


    if (output_to_stdout_json) {
        std::stringstream json_out_ss;
        json_out_ss << "{";
        json_out_ss << "\"results\":{";
        json_out_ss << "\"cost\":" << (final_atomic_cost == std::numeric_limits<long long>::max() ? "null" : std::to_string(final_atomic_cost)) << ",";
        json_out_ss << "\"solution_size\":" << final_reverted_solution_size << ",";
        json_out_ss << "\"solve_time_s\":" << std::fixed << std::setprecision(2) << solve_duration << ",";
        json_out_ss << "\"feasible_on_original\":" << (is_final_sol_valid ? "true" : "false");
        json_out_ss << "},";
        json_out_ss << "\"parameters\":{";
        json_out_ss << "\"graph_file\":\"" << escape_json_string(graphFile) << "\",";
        json_out_ss << "\"cutoff_s\":" << std::fixed << std::setprecision(1) << cutoff << ",";
        json_out_ss << "\"num_threads\":" << num_threads << ",";
        json_out_ss << "\"seed_used\":" << master_seed_to_use;
        json_out_ss << "}";
        json_out_ss << "}";
        std::cout << json_out_ss.str() << std::endl;
    } else {
        std::cout << "------------------------------------\n";
        std::cout << "求解器完成。\n";
        if (final_atomic_cost != std::numeric_limits<long long>::max() && (g_original_main.n > 0 || (g_original_main.n==0 && final_atomic_cost ==0)) ) {
            std::cout << "找到的最佳解成本 = " << final_atomic_cost << "\n";
            std::cout << "解的大小 = " << final_reverted_solution_size << "\n";
            if (is_final_sol_valid) {
                std::cout << "最终复原解在原始图上验证为可行。" << std::endl;
            } else {
                 // Error message already printed during validation if !is_final_sol_valid for n > 0
                 if (g_original_main.n > 0) std::cout << "最终复原解在原始图上验证为不可行。" << std::endl;
            }

            if (is_final_sol_valid && final_reverted_solution_size > 0 && final_reverted_solution_size <= 100) {
                std::cout << "解中的顶点 (0-based):\n";
                bool first = true;
                if (final_reverted_solution.size() == (size_t)g_original_main.n) {
                    for(int v=0; v<g_original_main.n; v++){
                        if(final_reverted_solution[v]) {
                            if (!first) std::cout << " "; else first = false;
                            std::cout << v;
                        }
                    }
                    std::cout << std::endl;
                }
            } else if (is_final_sol_valid && final_reverted_solution_size > 100) {
                std::cout << "(解包含超过 100 个顶点，未打印)\n";
            }
        } else if (g_original_main.n > 0) { // Only print "not found" if graph was not empty
            std::cout << "找到的最佳解成本 = N/A (错误或未找到解)\n";
            std::cout << "解的大小 = 0\n";
            std::cout << "(解为空或无效)\n";
        } else { // Empty graph case for terminal
            std::cout << "找到的最佳解成本 = 0\n";
            std::cout << "解的大小 = 0\n";
            std::cout << "最终复原解在原始图上验证为可行。\n";
        }
    }

    // Save solution file (always try if a solution was found, regardless of --stdout)
    // Condition: cost is not max, OR it's an empty graph (cost 0).
    if (final_atomic_cost != std::numeric_limits<long long>::max() || (g_original_main.n == 0 && final_atomic_cost == 0) ) {
        const std::string outputDir = "solutions";
        std::filesystem::path solutionFilePath;
        std::string solutionFilePathStr = "solution.json"; // Default

        try {
            std::filesystem::create_directories(outputDir);
            std::filesystem::path inputPath(graphFile);
            std::string baseFilename = inputPath.filename().string();
            std::string sanitized_filename = "";
            for (char c : baseFilename) {
                if (isalnum(c) || c == '.' || c == '_' || c == '-') {
                    sanitized_filename += c;
                } else {
                    sanitized_filename += '_';
                }
            }
            if (sanitized_filename.empty()) sanitized_filename = "graph";

            std::time_t now_for_filename = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            char time_buf[100] = {0};
            if (std::strftime(time_buf, sizeof(time_buf)-1, "%Y%m%d-%H%M%S", std::localtime(&now_for_filename))) {
                // Successfully formatted
            } else {
                strncpy(time_buf, "timestamp-error", sizeof(time_buf)-1); // Fallback
            }
            std::string timestamp_str(time_buf);

            solutionFilePath = std::filesystem::path(outputDir) / (sanitized_filename + "_" + timestamp_str + "_seed" + std::to_string(master_seed_to_use) + ".sol.json");
            solutionFilePathStr = solutionFilePath.string();
        } catch (const std::exception& e) {
            if (!output_to_stdout_json) {
                std::cerr << "创建解决方案文件路径时出错: " << e.what() << std::endl;
                std::cerr << "将尝试在当前目录保存为 'solution_" << master_seed_to_use << ".json'。\n";
            }
            solutionFilePathStr = "solution_" + std::to_string(master_seed_to_use) + ".json";
        }

        std::ofstream solution_file_json_out(solutionFilePathStr);
        if (solution_file_json_out.is_open()) {
            if (!output_to_stdout_json) {
                std::cout << "将解保存到 " << solutionFilePathStr << "...\n";
            }
            solution_file_json_out << "{" << std::endl;
            solution_file_json_out << "  \"solution_vertices\": [";
            bool first_vertex = true;
            if (final_reverted_solution.size() == (size_t)g_original_main.n) {
                for(int v=0; v<g_original_main.n; ++v) {
                    if(final_reverted_solution[v]) {
                        if (!first_vertex) solution_file_json_out << ", ";
                        solution_file_json_out << v;
                        first_vertex = false;
                    }
                }
            } else if (g_original_main.n == 0 && final_reverted_solution.empty()) {
                // Correct for empty graph: empty array
            } else if (!output_to_stdout_json) { // Print warning if size mismatch and not stdout json
                std::cerr << "警告: 最终解大小与原始图不匹配，顶点列表可能不正确地写入文件。\n";
            }
            solution_file_json_out << "]," << std::endl;

            solution_file_json_out << "  \"solver_run_information\": {" << std::endl;
            solution_file_json_out << "    \"cost\": " << (final_atomic_cost == std::numeric_limits<long long>::max() ? "null" : std::to_string(final_atomic_cost)) << "," << std::endl;
            solution_file_json_out << "    \"solution_size\": " << final_reverted_solution_size << "," << std::endl;
            solution_file_json_out << "    \"feasible_on_original\": " << (is_final_sol_valid ? "true" : "false") << "," << std::endl;
            solution_file_json_out << "    \"solve_time_s\": " << std::fixed << std::setprecision(2) << solve_duration << "," << std::endl;
            solution_file_json_out << "    \"threads_used\": " << num_threads << "," << std::endl;
            solution_file_json_out << "    \"cutoff_time_s\": " << std::fixed << std::setprecision(1) << cutoff << "," << std::endl;
            solution_file_json_out << "    \"graph_file\": \"" << escape_json_string(graphFile) << "\"," << std::endl;
            solution_file_json_out << "    \"graph_vertices_n_orig\": " << g_original_main.n << "," << std::endl;
            solution_file_json_out << "    \"graph_edges_m_orig\": " << g_original_main.m << "," << std::endl;
            solution_file_json_out << "    \"seed_used\": " << master_seed_to_use << std::endl;
            solution_file_json_out << "  }" << std::endl;
            solution_file_json_out << "}" << std::endl;
            solution_file_json_out.close();
            if (!output_to_stdout_json) {
                std::cout << "解已成功保存。\n";
            }
        } else {
            if (!output_to_stdout_json) {
                std::cerr << "错误: 无法打开 " << solutionFilePathStr << " 文件进行写入。\n";
            }
        }
    } else if (!output_to_stdout_json && g_original_main.n > 0) {
         std::cout << "(未找到可行解或解无效，未保存文件)\n";
    }


    if (!output_to_stdout_json) {
        std::cout << "------------------------------------\n";
    }

    return 0;
}