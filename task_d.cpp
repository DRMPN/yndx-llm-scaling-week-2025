#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip> // Для setprecision

using namespace std;

// Структура для хранения точки и ее исходного ID (1-индексированного)
struct Point {
    double x, y;
    int id;
};

// Глобальные переменные для решения
vector<vector<int>> adj; // Список смежности
vector<int> subtree_size;  // Размер поддерева
vector<Point> all_points;  // Все точки
vector<int> permutation;   // Итоговая перестановка ans[node] = point_id

/**
 * @brief DFS 1: Вычисляет размер поддерева для каждого узла.
 * @param u Текущий узел (1-индексированный)
 * @param p Родительский узел (для предотвращения возврата)
 */
void dfs_size(int u, int p) {
    subtree_size[u] = 1; // Учитываем сам узел
    for (int v : adj[u]) {
        if (v != p) {
            dfs_size(v, u);
            subtree_size[u] += subtree_size[v];
        }
    }
}

/**
 * @brief DFS 2: Рекурсивно назначает точки узлам.
 * @param u Текущий узел (1-индексированный)
 * @param p Родительский узел
 * @param available_indices Вектор 0-индексов в `all_points`, доступных для поддерева u
 */
void dfs_assign(int u, int p, vector<int>& available_indices) {
    if (available_indices.empty()) {
        return;
    }

    // 1. Найти "якорь" для u: точку с минимальной X-координатой
    int anchor_vec_idx = 0; // Индекс *внутри* available_indices
    for (int i = 1; i < available_indices.size(); ++i) {
        int current_idx = available_indices[i];
        int best_idx = available_indices[anchor_vec_idx];
        
        if (all_points[current_idx].x < all_points[best_idx].x ||
           (all_points[current_idx].x == all_points[best_idx].x &&
            all_points[current_idx].y < all_points[best_idx].y)) {
            anchor_vec_idx = i;
        }
    }

    // 2. Назначить якорь узлу u
    int anchor_global_idx = available_indices[anchor_vec_idx]; // 0-индекс в all_points
    permutation[u] = all_points[anchor_global_idx].id; // Сохраняем 1-ID точки
    Point& anchor_point = all_points[anchor_global_idx];

    // 3. Подготовить точки для детей
    vector<int> child_point_indices;
    for (int i = 0; i < available_indices.size(); ++i) {
        if (i != anchor_vec_idx) {
            child_point_indices.push_back(available_indices[i]);
        }
    }

    // 4. Отсортировать точки детей по углу
    sort(child_point_indices.begin(), child_point_indices.end(), 
         [&](int idx_a, int idx_b) {
        double angle_a = atan2(all_points[idx_a].y - anchor_point.y, 
                               all_points[idx_a].x - anchor_point.x);
        double angle_b = atan2(all_points[idx_b].y - anchor_point.y, 
                               all_points[idx_b].x - anchor_point.x);
        return angle_a < angle_b;
    });

    // 5. Рекурсивно распределить точки по детям
    int current_point_idx = 0;
    for (int v : adj[u]) {
        if (v != p) {
            int points_needed = subtree_size[v];
            
            // Создаем срез точек для поддерева v
            vector<int> sub_vector;
            for (int i = 0; i < points_needed; ++i) {
                if (current_point_idx < child_point_indices.size()) {
                    sub_vector.push_back(child_point_indices[current_point_idx++]);
                }
            }
            
            if (!sub_vector.empty()) {
                dfs_assign(v, u, sub_vector);
            }
        }
    }
}

void solve() {
    int n;
    cin >> n;

    // Инициализация структур данных
    adj.assign(n + 1, vector<int>());
    subtree_size.assign(n + 1, 0);
    permutation.assign(n + 1, 0);
    all_points.resize(n);

    // Чтение ребер
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // Чтение точек
    for (int i = 0; i < n; ++i) {
        cin >> all_points[i].x >> all_points[i].y;
        all_points[i].id = i + 1; // Сохраняем 1-индексированный ID
    }

    // --- Шаг 1: DFS для подсчета размеров ---
    dfs_size(1, 0); // Корень в 1, родитель 0 (фиктивный)

    // --- Шаг 2: DFS для назначения ---
    vector<int> initial_indices(n);
    for(int i = 0; i < n; ++i) {
        initial_indices[i] = i; // 0-индексы
    }
    dfs_assign(1, 0, initial_indices);

    // --- Шаг 3: Вывод ---
    for (int i = 1; i <= n; ++i) {
        cout << permutation[i] << (i == n ? "" : " ");
    }
    cout << "\n";
}

int main() {
    // Ускорение ввода-вывода
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    
    // Для корректного вывода double (хотя здесь не требуется)
    cout << fixed << setprecision(10);

    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    return 0;
}