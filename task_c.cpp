#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <climits> // Для INT_MAX

using namespace std;

// Используем большое число для "бесконечности",
// чтобы избежать переполнения при +1
const int INF = 1e9;

int main() {
    // Ускоряем ввод-вывод
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    int n, m;
    cin >> n >> m;
    int sx, sy;
    cin >> sx >> sy;
    sx--; // Переводим в 0-индексацию
    sy--; // Переводим в 0-индексацию

    vector<string> grid(n);
    for (int i = 0; i < n; ++i) {
        cin >> grid[i];
    }

    string s;
    cin >> s;

    // dp[i][j] = мин. время, чтобы закончить *текущую* доставку в (i, j)
    vector<vector<int>> dp(n, vector<int>(m, INF));

    // --- Базовый случай (k = 0): доставка s[0] ---
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (grid[i][j] == s[0]) {
                dp[i][j] = abs(i - sx) + abs(j - sy);
            }
        }
    }

    // --- DP шаги (k = 1 ... s.length() - 1) ---
    for (int k = 1; k < s.length(); ++k) {
        char prev_char = s[k - 1];
        char target_char = s[k];

        // Временная таблица для distance transform
        // temp[i][j] = мин. стоимость прибытия из *любой*
        // клетки с prev_char
        vector<vector<int>> temp(n, vector<int>(m, INF));

        // 1. Инициализируем temp значениями из dp
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] == prev_char) {
                    temp[i][j] = dp[i][j];
                }
            }
        }

        // 2. Выполняем L1 distance transform
        // Проходы по строкам
        for (int i = 0; i < n; ++i) {
            // Слева направо
            for (int j = 1; j < m; ++j) {
                temp[i][j] = min(temp[i][j], temp[i][j - 1] + 1);
            }
            // Справа налево
            for (int j = m - 2; j >= 0; --j) {
                temp[i][j] = min(temp[i][j], temp[i][j + 1] + 1);
            }
        }

        // Проходы по столбцам
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                // Сверху вниз
                temp[i][j] = min(temp[i][j], temp[i - 1][j] + 1);
            }
        }
        for (int i = n - 2; i >= 0; --i) {
            for (int j = 0; j < m; ++j) {
                // Снизу вверх
                temp[i][j] = min(temp[i][j], temp[i + 1][j] + 1);
            }
        }

        // 3. Обновляем dp для текущего шага k
        // (temp теперь содержит полные затраты на L1-преобразование)
        vector<vector<int>> new_dp(n, vector<int>(m, INF));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] == target_char) {
                    new_dp[i][j] = temp[i][j];
                }
            }
        }
        dp = new_dp; // Переходим к следующему состоянию
    }

    // --- Финальный ответ ---
    // Находим минимум в последней dp-таблице
    int min_time = INF;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            min_time = min(min_time, dp[i][j]);
        }
    }

    cout << min_time << "\n";

    return 0;
}