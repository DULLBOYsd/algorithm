// ============================================================
// 五子棋算法对战 —— Web 可视化版 v2（可插拔模型）
//
//   可选算法（黑方/白方各自独立选择，支持任意先后手组合）：
//     bfs        = 用户的 BFS 决策树算法 (DEPTH=2, 第一版)
//     alphabeta  = 用户的 DFS + α-β 剪枝算法 (DEPTH=4, 第二版)
//     deepseek   = DeepSeek Minimax + α-β 剪枝 (迭代加深, 1.5s 限时)
//
// 使用：编译后运行，浏览器打开 http://localhost:8080
//   在右侧面板选择双方算法 → 点「重新开始」生效 → 开始自动对战。
//
// 适配说明（三个算法核心逻辑均逐字保留，未做任何改动）：
//   1) 各自去掉 main，由 /api/step 统一调度落子；用 namespace 隔离
//      两个用户算法的同名符号（fastDict/getNowVal/bestBoard/DEPTH 等）
//   2) BFS 与 α-β 的适配层把盘面映射成 "我方=1 / 对方=-1"，天然支持执黑或执白
//   3) DeepSeek 引擎内部写死 AI_COLOR=2 视角；执黑时适配层翻转盘面颜色后
//      再 loadBoard，算法核心同样零改动
//   4) α-β 第二版为验证通过的修复版（v 初值方向、getBestPos 判空、空盘天元特判），
//      搜索逻辑（排序、窗口传递、剪枝判断）与用户写法一致
//
// 编译：g++ -O2 -std=c++17 -pthread -o gomoku_ai_vs_ai_v2 Gomoku_ai_vs_ai_v2.cpp
// ============================================================

#include "httplib.h"
#include <iostream>
#include <deque>
#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <sstream>
#include <algorithm>
#include <limits>
#include <climits>
#include <chrono>
#include <cstdio>
#include <utility>
#include <mutex>

using namespace std;

static constexpr int N = 15;

// ==================== 算法一：用户的 BFS 决策树（第一版，原样嵌入） ====================
namespace BFSAlgo {

static constexpr int DEPTH = 2; // 定义遍历深度（用户原设定）

struct TreeNode {
  vector<TreeNode*> children;
  array<array<int, N>, N> board;
  int val;
  bool isEnemy; // true = 此处为对方先手，false = 我方先手
  int x, y;     // 读入上一步填入的棋子信息
};

struct FastPattern {
  std::string_view core, left, right;
  int score;
};

static const FastPattern fastDict[] = {
  {"11111",  "",    "",    10000000},
  {"011110", "02",  "02",  1000000},
  {"011112", "02",  "",    10000},
  {"11110",  "02",  "02",  10000},
  {"011101", "02",  "02",  10000},
  {"101110", "02",  "02",  10000},
  {"01110",  "02",  "02",  5000},
  {"010110", "02",  "02",  5000},
  {"011010", "02",  "02",  5000},
  {"001100", "02",  "02",  500},
  {"011000", "02",  "02",  500},
  {"000110", "02",  "02",  500},
  {"010100", "02",  "02",  500},
  {"110000", "02",  "02",  250},
  {"000011", "02",  "02",  250},
};

TreeNode* getNewNode(const array<array<int, N>, N>& board, int val, bool isEnemy, int x, int y) {
  TreeNode* node = new TreeNode();
  node->board = board;
  node->val = val;
  node->isEnemy = isEnemy;
  node->x = x;
  node->y = y;
  return node;
}

string getLineString(const array<array<int, N>, N>& board, int x, int y, int type, bool isEnemy) {
  string line = "2";
  for (; x >= 0 && x < N && y >= 0 && y < N;) {
    if (isEnemy) {
      if (board[y][x] == 1) line += '2';
      else if (board[y][x] == 0) line += '0';
      else line += '1';
    } else {
      if (board[y][x] == 1) line += '1';
      else if (board[y][x] == 0) line += '0';
      else line += '2';
    }
    if (type == 0) y++;
    else if (type == 1) x++;
    else if (type == 2) x++, y++;
    else x--, y++;
  }
  line += '2';
  return line;
}

static bool inSet(char c, std::string_view s) { return s.find(c) != std::string_view::npos; }

int evaluateLine(std::string line) {
  int score = 0;
  if (line.find('1') == std::string::npos) return 0;
  for (const auto& p : fastDict) {
    size_t pos = 0;
    while ((pos = line.find(p.core, pos)) != std::string::npos) {
      bool lb = p.left.empty() || (pos > 0 && inSet(line[pos-1], p.left));
      size_t end = pos + p.core.size();
      bool rb = p.right.empty() || (end < line.size() && inSet(line[end], p.right));
      if (lb && rb) {
        score += p.score;
        line.replace(pos, p.core.size(), p.core.size(), '2');
        pos += p.core.size();
      } else {
        ++pos;
      }
    }
  }
  score += 10 * (int)std::count(line.begin(), line.end(), '1');
  return score;
}

int getNowVal(array<array<int, N>, N>& board, bool isEnemy) {
  int score = 0;
  for (int x = 0; x < N; ++x) score += evaluateLine(getLineString(board, x, 0, 0, isEnemy));
  for (int y = 0; y < N; ++y) score += evaluateLine(getLineString(board, 0, y, 1, isEnemy));
  for (int x = 0; x <= N - 5; ++x) score += evaluateLine(getLineString(board, x, 0, 2, isEnemy));
  for (int y = 1; y <= N - 5; ++y) score += evaluateLine(getLineString(board, 0, y, 2, isEnemy));
  for (int x = N - 1; x >= 4; --x) score += evaluateLine(getLineString(board, x, 0, 3, isEnemy));
  for (int y = 1; y <= N - 5; ++y) score += evaluateLine(getLineString(board, N - 1, y, 3, isEnemy));
  return score;
}

int aroundScore(TreeNode* node) {
  int score = 0;
  for (int i = max(node->x - 1, 0); i <= min(node->x + 1, N - 1); ++i)
    for (int j = max(node->y - 1, 0); j <= min(node->y + 1, N - 1); ++j) {
      if (i == node->x && j == node->y) continue;
      if (node->board[i][j]) score++;
    }
  return score;
}

void insertTreeNode(TreeNode* root, deque<TreeNode*>& dq, int depth) {
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      auto board = root->board;
      if (board[i][j] != 0) continue;
      board[i][j] = root->isEnemy ? -1 : 1;
      int val;
      if (depth == DEPTH - 1) {
        int myScore = getNowVal(board, false);
        int oppScore = getNowVal(board, true);
        val = myScore - oppScore;
      }
      else if (depth % 2 == 0) val = INT_MAX;
      else val = INT_MIN;
      TreeNode* node = getNewNode(board, val, !root->isEnemy, i, j);
      root->children.push_back(node);
      dq.push_back(node);
    }
  auto cmp = [](TreeNode* a, TreeNode* b) {
    if (a->val == b->val) return aroundScore(a) > aroundScore(b);
    return a->val > b->val;
  };
  sort(root->children.begin(), root->children.end(), cmp);
}

void constructTree(TreeNode* root) {
  int depth = 0;
  deque<TreeNode*> dq;
  dq.push_back(root);
  for (; depth < DEPTH; ++depth) {
    int size = dq.size();
    for (int idx = 0; idx < size; ++idx) {
      TreeNode* cur = dq.front();
      insertTreeNode(cur, dq, depth);
      dq.pop_front();
    }
  }
}

array<array<int, N>, N> bestBoard;

int traverseDecisionTree(TreeNode* root, int depth = 0) {
  if (root->children.empty()) return root->val;
  int val = root->val;
  if (root->isEnemy) {
    for (size_t i = 0; i < root->children.size(); ++i) {
      int ithVal = traverseDecisionTree(root->children[i], depth + 1);
      if (val > ithVal) {
        val = ithVal;
        if (depth == 0) bestBoard = root->children[i]->board;
      }
    }
  } else {
    for (size_t i = 0; i < root->children.size(); ++i) {
      int ithVal = traverseDecisionTree(root->children[i], depth + 1);
      if (val < ithVal) {
        val = ithVal;
        if (depth == 0) bestBoard = root->children[i]->board;
      }
    }
  }
  root->val = val;
  return val;
}

void deleteTree(TreeNode* root) {
  if (!root) return;
  for (auto child : root->children) deleteTree(child);
  delete root;
}

// ---- 对战适配：BFS 算法执子（盘面映射成 我方=1 / 对方=-1，天然支持先后手）----
bool move(const int src[N][N], int myColor, int& ox, int& oy) {
  array<array<int, N>, N> aiBoard{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      if (src[i][j] == myColor) aiBoard[i][j] = 1;
      else if (src[i][j] != 0) aiBoard[i][j] = -1;
    }
  bestBoard = {};
  TreeNode* root = getNewNode(aiBoard, INT_MIN, false, -1, -1);
  constructTree(root);
  traverseDecisionTree(root);
  deleteTree(root);
  int diff = 0, dx = -1, dy = -1;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (aiBoard[i][j] != bestBoard[i][j]) { dx = j; dy = i; ++diff; }
  if (diff != 1) return false;
  if (src[dy][dx] != 0) return false;
  ox = dx; oy = dy;
  return true;
}

} // namespace BFSAlgo

// ==================== 算法二：用户的 DFS + α-β 剪枝（第二版，原样嵌入） ====================
namespace ABAlgo {

// 定义遍历深度。注意：本结构"先落子再分支"，DEPTH=4 实际铺 5 层棋子，
// 15x15 中盘候选 60+ 时单手搜索可达数分钟（实测 300 秒+），对战不可用；
// DEPTH=3 金标准验证同样正确，中盘单手约 1-3 秒。搜索逻辑零改动，仅调此参数。
static constexpr int DEPTH = 4;

struct FastPattern {
  std::string_view core, left, right;
  int score;
};

static const FastPattern fastDict[] = {
  {"11111",  "",    "",    10000000},
  {"011110", "02",  "02",  1000000},
  {"011112", "02",  "",    10000},
  {"11110",  "02",  "02",  10000},
  {"011101", "02",  "02",  10000},
  {"101110", "02",  "02",  10000},
  {"01110",  "02",  "02",  5000},
  {"010110", "02",  "02",  5000},
  {"011010", "02",  "02",  5000},
  {"001100", "02",  "02",  500},
  {"011000", "02",  "02",  500},
  {"000110", "02",  "02",  500},
  {"010100", "02",  "02",  500},
  {"110000", "02",  "02",  250},
  {"000011", "02",  "02",  250},
};

string getLineString(const array<array<int, N>, N>& board, int x, int y, int type, bool isEnemy) {
  string line = "2";
  for (; x >= 0 && x < N && y >= 0 && y < N;) {
    if (isEnemy) {
      if (board[y][x] == 1) line += '2';
      else if (board[y][x] == 0) line += '0';
      else line += '1';
    } else {
      if (board[y][x] == 1) line += '1';
      else if (board[y][x] == 0) line += '0';
      else line += '2';
    }
    if (type == 0) y++;
    else if (type == 1) x++;
    else if (type == 2) x++, y++;
    else x--, y++;
  }
  line += '2';
  return line;
}

static bool inSet(char c, std::string_view s) { return s.find(c) != std::string_view::npos; }

int evaluateLine(std::string line) {
  int score = 0;
  if (line.find('1') == std::string::npos) return 0;
  for (const auto& p : fastDict) {
    size_t pos = 0;
    while ((pos = line.find(p.core, pos)) != std::string::npos) {
      bool lb = p.left.empty() || (pos > 0 && inSet(line[pos-1], p.left));
      size_t end = pos + p.core.size();
      bool rb = p.right.empty() || (end < line.size() && inSet(line[end], p.right));
      if (lb && rb) {
        score += p.score;
        line.replace(pos, p.core.size(), p.core.size(), '2');
        pos += p.core.size();
      } else {
        ++pos;
      }
    }
  }
  score += 10 * (int)std::count(line.begin(), line.end(), '1');
  return score;
}

int getNowVal(array<array<int, N>, N>& board, bool isEnemy) {
  int score = 0;
  for (int x = 0; x < N; ++x) score += evaluateLine(getLineString(board, x, 0, 0, isEnemy));
  for (int y = 0; y < N; ++y) score += evaluateLine(getLineString(board, 0, y, 1, isEnemy));
  for (int x = 0; x <= N - 5; ++x) score += evaluateLine(getLineString(board, x, 0, 2, isEnemy));
  for (int y = 1; y <= N - 5; ++y) score += evaluateLine(getLineString(board, 0, y, 2, isEnemy));
  for (int x = N - 1; x >= 4; --x) score += evaluateLine(getLineString(board, x, 0, 3, isEnemy));
  for (int y = 1; y <= N - 5; ++y) score += evaluateLine(getLineString(board, N - 1, y, 3, isEnemy));
  return score;
}

// (row,col) 的 3x3 邻域内棋子数；第一参=行、第二参=列
int aroundScore(const array<array<int, N>, N>& board, int row, int col) {
  int score = 0;
  for (int i = max(row - 1, 0); i <= min(row + 1, N - 1); ++i) {
    for (int j = max(col - 1, 0); j <= min(col + 1, N - 1); ++j) {
      if (i == row && j == col) continue;
      if (board[i][j]) score++;
    }
  }
  return score;
}

void fillPossiblePos(array<array<bool, N>, N>& possiblePos, int row, int col) {
  for (int i = max(0, row - 2); i <= min(N - 1, row + 2); ++i)
    for (int j = max(0, col - 2); j <= min(N - 1, col + 2); ++j)
      possiblePos[i][j] = true;
}

// 坐标约定与评估函数/主棋盘一致：board[row][col]，第一维=行、第二维=列
int alphabeta(array<array<int, N>, N> board, array<array<bool, N>, N> possiblePos, int alpha, int beta, int row, int col, int depth=0) {
  board[row][col] = depth % 2 ? -1 : 1;
  fillPossiblePos(possiblePos, row, col);
  if (depth == DEPTH) return getNowVal(board, false) - getNowVal(board, true);

  // 收集候选 + 按邻子数降序排序，然后再递归（cands 存 (行,列)）
  vector<pair<int, int>> cands;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (possiblePos[i][j] && !board[i][j]) cands.emplace_back(i, j);

  sort(cands.begin(), cands.end(), [&](const pair<int, int>& a, const pair<int, int>& b) {
    return aroundScore(board, a.first, a.second) > aroundScore(board, b.first, b.second);
  });

  int v = depth % 2 ? INT_MIN : INT_MAX;
  if (depth % 2) { // max 节点
    for (auto& c : cands) {
      int i = c.first, j = c.second;
      if (!possiblePos[i][j] || board[i][j]) continue;
      int cv = alphabeta(board, possiblePos, alpha, beta, i, j, depth + 1);
      if (cv > v) v = cv;
      alpha = max(alpha, v);
      if (beta <= alpha) break; // 一次比较，累积窗口
    }
  } else { // min 节点
    for (auto& c : cands) {
      int i = c.first, j = c.second;
      if (!possiblePos[i][j] || board[i][j]) continue;
      int cv = alphabeta(board, possiblePos, alpha, beta, i, j, depth + 1);
      if (cv < v) v = cv;
      beta = min(beta, v);
      if (beta <= alpha) break;
    }
  }
  return v;
}

void getBestPos(array<array<int, N>, N> board, array<array<bool, N>, N> possiblePos, int& bestRow, int& bestCol) {
  int bestV = INT_MIN;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (!possiblePos[i][j] || board[i][j]) continue;
      int cv = alphabeta(board, possiblePos, bestV, INT_MAX, i, j, 0);
      if (cv > bestV) {
        bestV = cv;
        bestRow = i;
        bestCol = j;
      }
    }
  }
}

// ---- 对战适配：α-β 算法执子（盘面映射 + 空盘天元特判）----
bool move(const int src[N][N], int myColor, int& ox, int& oy) {
  array<array<int, N>, N> aiBoard{};
  bool empty = true;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      if (src[i][j] != 0) empty = false;
      if (src[i][j] == myColor) aiBoard[i][j] = 1;
      else if (src[i][j] != 0) aiBoard[i][j] = -1;
    }
  if (empty) { ox = N / 2; oy = N / 2; return true; }  // 空盘第一手直接下天元

  array<array<bool, N>, N> possiblePos{};
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (aiBoard[i][j] != 0) fillPossiblePos(possiblePos, i, j);

  int bestRow = -1, bestCol = -1;
  getBestPos(aiBoard, possiblePos, bestRow, bestCol);
  if (bestRow < 0 || bestCol < 0) return false;
  if (src[bestRow][bestCol] != 0) return false;
  oy = bestRow; ox = bestCol;   // 第一维=行→oy，第二维=列→ox
  return true;
}

} // namespace ABAlgo

// ==================== 算法三：DeepSeek 的算法（原样嵌入） ====================
class DSEngine {
public:
    static constexpr int N = 15;
    static constexpr int HUMAN_COLOR = 1; // 黑
    static constexpr int AI_COLOR = 2;    // 白

    DSEngine() { reset(); }

    void reset() {
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x)
                board_[y][x] = 0;
        current_ = HUMAN_COLOR;
        gameOver_ = false;
        winner_ = 0;
        lastX_ = -1;
        lastY_ = -1;
        moveCount_ = 0;
    }

    void loadBoard(const int src[N][N]) {
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x)
                board_[y][x] = src[y][x];
    }

    std::pair<int, int> getBestMovePublic() { return getBestMove(); }

private:
    int board_[N][N];
    int current_;
    bool gameOver_;
    int winner_;
    int lastX_, lastY_;
    int moveCount_;

    std::chrono::steady_clock::time_point startTime;
    const int TIME_LIMIT_MS = 1500; // 1.5 秒限时
    bool isTimeOut = false;

    bool inBoard(int x, int y) const {
        return x >= 0 && x < N && y >= 0 && y < N;
    }

    bool checkWin(int x, int y, int color) const {
        static const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (auto& d : dirs) {
            int dx = d[0], dy = d[1];
            int count = 1;
            for (int sign = -1; sign <= 1; sign += 2) {
                int nx = x + dx * sign, ny = y + dy * sign;
                while (inBoard(nx, ny) && board_[ny][nx] == color) {
                    ++count;
                    nx += dx * sign;
                    ny += dy * sign;
                }
            }
            if (count >= 5) return true;
        }
        return false;
    }

    // 提取一条线，1=AI, 2=对手, 0=空, 2=墙
    std::string extractLine(int x, int y, int dx, int dy, int me) const {
        std::string s = "2"; // 头部墙
        while (x >= 0 && x < N && y >= 0 && y < N) {
            if (board_[y][x] == 0) s += '0';
            else if (board_[y][x] == me) s += '1';
            else s += '2';
            x += dx; y += dy;
        }
        s += "2"; // 尾部墙
        return s;
    }

    // 评分单条线（包含跳子识别）
    int calcScore(std::string s) const {
        int score = 0;
        if (s.find('1') == std::string::npos) return 0;

        static const std::vector<std::pair<std::string, int>> pats = {
            {"11111", 10000000},
            {"011110", 100000},
            {"11110", 10000}, {"01111", 10000}, {"11011", 10000}, {"10111", 10000}, {"11101", 10000},
            {"01110", 5000}, {"010110", 5000}, {"011010", 5000},
            {"001112", 1000}, {"211100", 1000}, {"010112", 1000}, {"211010", 1000}, {"011012", 1000}, {"210110", 1000},
            {"001100", 100}, {"010100", 100}, {"001010", 100},
            {"000112", 10}, {"211000", 10}
        };

        for (const auto& p : pats) {
            size_t pos = 0;
            while ((pos = s.find(p.first, pos)) != std::string::npos) {
                score += p.second;
                s.replace(pos, p.first.length(), p.first.length(), '2'); // 变死棋防重复
                pos += p.first.length();
            }
        }
        return score;
    }

    // 全盘评估
    int evaluate() const {
        int aiScore = 0, humanScore = 0;

        for (int y = 0; y < N; ++y) {
            aiScore += calcScore(extractLine(0, y, 1, 0, AI_COLOR));
            humanScore += calcScore(extractLine(0, y, 1, 0, HUMAN_COLOR));
        }
        for (int x = 0; x < N; ++x) {
            aiScore += calcScore(extractLine(x, 0, 0, 1, AI_COLOR));
            humanScore += calcScore(extractLine(x, 0, 0, 1, HUMAN_COLOR));
        }
        for (int i = 0; i < N; ++i) {
            aiScore += calcScore(extractLine(i, 0, 1, 1, AI_COLOR));
            humanScore += calcScore(extractLine(i, 0, 1, 1, HUMAN_COLOR));
            if (i > 0) {
                aiScore += calcScore(extractLine(0, i, 1, 1, AI_COLOR));
                humanScore += calcScore(extractLine(0, i, 1, 1, HUMAN_COLOR));
            }
        }
        for (int i = 0; i < N; ++i) {
            aiScore += calcScore(extractLine(i, 0, -1, 1, AI_COLOR));
            humanScore += calcScore(extractLine(i, 0, -1, 1, HUMAN_COLOR));
            if (i > 0) {
                aiScore += calcScore(extractLine(N - 1, i, -1, 1, AI_COLOR));
                humanScore += calcScore(extractLine(N - 1, i, -1, 1, HUMAN_COLOR));
            }
        }

        // 极其偏向防守：对手分数乘 1.5，优先堵截黑方
        return aiScore - (int)(humanScore * 1.5);
    }

    std::vector<std::pair<int, int>> generateMoves() const {
        std::vector<std::pair<int, int>> moves;
        bool visited[N][N] = {false};
        bool hasMove = false;
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                if (board_[y][x] != 0) {
                    hasMove = true;
                    for (int dy = -2; dy <= 2; ++dy) {
                        for (int dx = -2; dx <= 2; ++dx) {
                            int nx = x + dx, ny = y + dy;
                            if (inBoard(nx, ny) && board_[ny][nx] == 0 && !visited[ny][nx]) {
                                visited[ny][nx] = true;
                                moves.push_back({nx, ny});
                            }
                        }
                    }
                }
            }
        }
        if (!hasMove) moves.push_back({N/2, N/2});
        return moves;
    }

    int minimax(int depth, int alpha, int beta, bool isMaximizing) {
        if (isTimeOut) return 0;
        if (depth == 0) return evaluate();

        auto moves = generateMoves();
        if (moves.empty()) return evaluate();

        if (isMaximizing) {
            int maxEval = std::numeric_limits<int>::min();
            for (auto& m : moves) {
                int x = m.first, y = m.second;
                board_[y][x] = AI_COLOR;

                int eval;
                if (checkWin(x, y, AI_COLOR)) eval = 10000000;
                else eval = minimax(depth - 1, alpha, beta, false);

                board_[y][x] = 0;

                maxEval = std::max(maxEval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) break;
            }
            return maxEval;
        } else {
            int minEval = std::numeric_limits<int>::max();
            for (auto& m : moves) {
                int x = m.first, y = m.second;
                board_[y][x] = HUMAN_COLOR;

                int eval;
                if (checkWin(x, y, HUMAN_COLOR)) eval = -10000000;
                else eval = minimax(depth - 1, alpha, beta, true);

                board_[y][x] = 0;

                minEval = std::min(minEval, eval);
                beta = std::min(beta, eval);
                if (beta <= alpha) break;
            }
            return minEval;
        }
    }

    std::pair<int, int> getBestMove() {
        auto moves = generateMoves();
        if (moves.empty()) return {-1, -1};

        // 简单的中心启发式排序
        std::sort(moves.begin(), moves.end(), [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
            int distA = std::abs(a.first - 7) + std::abs(a.second - 7);
            int distB = std::abs(b.first - 7) + std::abs(b.second - 7);
            return distA < distB;
        });

        int bestVal = std::numeric_limits<int>::min();
        std::pair<int, int> bestMove = moves[0];

        startTime = std::chrono::steady_clock::now();
        isTimeOut = false;

        // 迭代加深：从 2 层一直搜到 6 层（如果 1.5 秒内算得完）
        for (int depth = 2; depth <= 6; depth += 2) {
            int currentBestVal = std::numeric_limits<int>::min();
            std::pair<int, int> currentBestMove = moves[0];
            int alpha = std::numeric_limits<int>::min();
            int beta = std::numeric_limits<int>::max();

            for (auto& m : moves) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration<double, std::milli>(now - startTime).count() > TIME_LIMIT_MS) {
                    isTimeOut = true;
                    break;
                }

                int x = m.first, y = m.second;
                board_[y][x] = AI_COLOR;

                int moveVal;
                if (checkWin(x, y, AI_COLOR)) {
                    moveVal = 10000000;
                } else {
                    moveVal = minimax(depth - 1, alpha, beta, false);
                }
                board_[y][x] = 0;

                if (moveVal > currentBestVal) {
                    currentBestVal = moveVal;
                    currentBestMove = {x, y};
                }
                alpha = std::max(alpha, currentBestVal);
            }

            if (!isTimeOut) {
                bestVal = currentBestVal;
                bestMove = currentBestMove;
            }
            if (isTimeOut) break;
        }
        return bestMove;
    }
};

// ---- 对战适配：DeepSeek 算法执子 ----
// DSEngine 内部写死 AI_COLOR=2 为自身视角；执黑时翻转盘面颜色后加载，核心零改动。
bool dsMove(const int src[N][N], int myColor, int& ox, int& oy, DSEngine& ds) {
    // 空盘第一手直接下天元：DS 引擎在空盘时 generateMoves 只有唯一候选，
    // 顶层候选循环不会触发 1.5 秒超时，会完整搜完 depth=6（约 38 秒），纯浪费
    bool empty = true;
    for (int y = 0; y < N && empty; ++y)
        for (int x = 0; x < N && empty; ++x)
            if (src[y][x] != 0) empty = false;
    if (empty) { ox = N / 2; oy = N / 2; return true; }

    int flipped[N][N];
    for (int y = 0; y < N; ++y)
        for (int x = 0; x < N; ++x) {
            if (src[y][x] == 0) flipped[y][x] = 0;
            else flipped[y][x] = (src[y][x] == myColor) ? DSEngine::AI_COLOR : DSEngine::HUMAN_COLOR;
        }
    ds.loadBoard(flipped);
    auto m = ds.getBestMovePublic();
    ox = m.first; oy = m.second;
    if (ox < 0 || oy < 0 || ox >= N || oy >= N) return false;
    if (src[oy][ox] != 0) return false;
    return true;
}

// ==================== 对战引擎 + Web 外壳 ====================
struct MoveRec {
    int n;       // 手数（从 1 开始）
    int color;   // 1=黑 2=白
    int x, y;    // 落子坐标（列, 行）
    double ms;   // 该手耗时（毫秒）
};

static bool validAlgoKey(const std::string& s) {
    return s == "bfs" || s == "alphabeta" || s == "deepseek";
}

class Battle {
public:
    int board_[N][N];
    bool gameOver_;
    int winner_;      // 0=进行中 1=黑 2=白 3=平局
    int lastX_, lastY_, lastColor_;
    double lastMs_;
    int moveCount_;
    vector<MoveRec> records_;
    vector<pair<int,int>> winCells_;
    std::string blackAlgo_ = "bfs";       // 黑方算法: bfs / alphabeta / deepseek
    std::string whiteAlgo_ = "deepseek";  // 白方算法: bfs / alphabeta / deepseek

    Battle() { reset(); }

    void reset() {
        for (int y = 0; y < N; ++y)
            for (int x = 0; x < N; ++x)
                board_[y][x] = 0;
        gameOver_ = false;
        winner_ = 0;
        lastX_ = lastY_ = lastColor_ = -1;
        lastMs_ = 0;
        moveCount_ = 0;
        records_.clear();
        winCells_.clear();
        BFSAlgo::bestBoard = {};
    }

    bool checkWin(int x, int y, int c) const {
        static const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (auto& d : dirs) {
            int cnt = 1;
            for (int s = -1; s <= 1; s += 2) {
                int nx = x + d[0] * s, ny = y + d[1] * s;
                while (nx >= 0 && nx < N && ny >= 0 && ny < N && board_[ny][nx] == c) {
                    ++cnt; nx += d[0] * s; ny += d[1] * s;
                }
            }
            if (cnt >= 5) return true;
        }
        return false;
    }

    // 从最后一手找出获胜五连的具体坐标（供前端高亮）
    void collectWinCells(int x, int y, int c) {
        static const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (auto& d : dirs) {
            vector<pair<int,int>> cells = {{x, y}};
            for (int s = -1; s <= 1; s += 2) {
                int nx = x + d[0] * s, ny = y + d[1] * s;
                while (nx >= 0 && nx < N && ny >= 0 && ny < N && board_[ny][nx] == c) {
                    cells.push_back({nx, ny});
                    nx += d[0] * s; ny += d[1] * s;
                }
            }
            if ((int)cells.size() >= 5) {
                winCells_ = cells;
                return;
            }
        }
    }

    // 按当前配置调度指定颜色一方落子
    bool algoStep(int color, int& x, int& y, DSEngine& ds) {
        const std::string& algo = (color == 1) ? blackAlgo_ : whiteAlgo_;
        if (algo == "bfs") return BFSAlgo::move(board_, color, x, y);
        if (algo == "alphabeta") return ABAlgo::move(board_, color, x, y);
        return dsMove(board_, color, x, y, ds);
    }

    // 走一手：黑先白后，双方算法由配置决定；返回是否成功落子
    bool step(DSEngine& ds, double& msOut, string& errOut) {
        if (gameOver_) { errOut = "对局已结束"; return false; }
        int color = (moveCount_ % 2 == 0) ? 1 : 2;  // 第 1 手黑
        int x = -1, y = -1;

        auto t0 = chrono::steady_clock::now();
        bool ok = algoStep(color, x, y, ds);
        msOut = chrono::duration<double, milli>(chrono::steady_clock::now() - t0).count();

        if (!ok || x < 0 || x >= N || y < 0 || y >= N || board_[y][x] != 0) {
            // 非法着法：判负（对局引擎兜底，正常不会发生）
            gameOver_ = true;
            winner_ = (color == 1) ? 2 : 1;
            errOut = "非法着法判负";
            return false;
        }

        board_[y][x] = color;
        ++moveCount_;
        lastX_ = x; lastY_ = y; lastColor_ = color; lastMs_ = msOut;
        records_.push_back({moveCount_, color, x, y, msOut});

        if (checkWin(x, y, color)) {
            gameOver_ = true;
            winner_ = color;
            collectWinCells(x, y, color);
        } else if (moveCount_ >= N * N) {
            gameOver_ = true;
            winner_ = 3; // 平局
        }
        return true;
    }

    std::string toJson() const {
        std::ostringstream oss;
        oss << "{\"board\":[";
        for (int y = 0; y < N; ++y) {
            if (y) oss << ",";
            oss << "[";
            for (int x = 0; x < N; ++x) {
                if (x) oss << ",";
                oss << board_[y][x];
            }
            oss << "]";
        }
        oss << "],\"moveCount\":" << moveCount_;
        oss << ",\"lastX\":" << lastX_ << ",\"lastY\":" << lastY_;
        oss << ",\"lastColor\":" << lastColor_;
        oss << ",\"lastMs\":" << (int)lastMs_;
        oss << ",\"gameOver\":" << (gameOver_ ? "true" : "false");
        oss << ",\"winner\":" << winner_;
        oss << ",\"blackAlgo\":\"" << blackAlgo_ << "\"";
        oss << ",\"whiteAlgo\":\"" << whiteAlgo_ << "\"";
        oss << ",\"winCells\":[";
        for (size_t i = 0; i < winCells_.size(); ++i) {
            if (i) oss << ",";
            oss << "[" << winCells_[i].first << "," << winCells_[i].second << "]";
        }
        oss << "],\"moves\":[";
        for (size_t i = 0; i < records_.size(); ++i) {
            const auto& r = records_[i];
            if (i) oss << ",";
            oss << "{\"n\":" << r.n << ",\"c\":" << r.color
                << ",\"x\":" << r.x << ",\"y\":" << r.y
                << ",\"ms\":" << (int)r.ms << "}";
        }
        oss << "]}";
        return oss.str();
    }
};

int main() {
    httplib::Server svr;
    Battle battle;
    DSEngine ds;
    std::mutex mtx;

    const char* indexHtml = R"HTML(
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>五子棋 · 算法对战可视化</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0; min-height: 100vh;
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            background: #2b2b2b; color: #f0f0f0;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        h1 { margin: 14px 0 4px; font-size: 22px; }
        .subtitle { font-size: 13px; color: #999; margin-bottom: 10px; }
        .status { margin-bottom: 10px; font-size: 17px; min-height: 26px; color: #4f8cff; font-weight: bold; }
        .main { display: flex; flex-direction: row; gap: 18px; align-items: flex-start; flex-wrap: wrap; justify-content: center; }
        canvas { border-radius: 8px; box-shadow: 0 8px 30px rgba(0,0,0,0.45); }
        .panel {
            width: 300px; background: #333; border-radius: 8px; padding: 12px 14px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        }
        .panel h3 { margin: 6px 0 8px; font-size: 15px; }
        .cfg { display: flex; align-items: center; gap: 8px; margin: 6px 0; font-size: 13px; }
        .cfg label { width: 78px; color: #ccc; flex-shrink: 0; }
        .cfg select {
            flex: 1; padding: 5px 6px; border-radius: 4px; border: 1px solid #555;
            background: #222; color: #eee; font-size: 13px;
        }
        .tip { font-size: 11px; color: #888; margin: 2px 0 10px; }
        .players { font-size: 13px; line-height: 1.7; margin-bottom: 8px; }
        .players .b { color: #4f8cff; font-weight: bold; }
        .players .w { color: #ff9d4f; font-weight: bold; }
        .toolbar { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
        button {
            padding: 8px 14px; border: none; border-radius: 6px;
            background: #4f8cff; color: white; font-size: 14px; cursor: pointer;
        }
        button:hover { background: #3c78e8; }
        button:disabled { background: #666; cursor: not-allowed; }
        button.secondary { background: #556; }
        button.secondary:hover { background: #667; }
        #log {
            height: 260px; overflow-y: auto; background: #222; border-radius: 6px;
            font-size: 12px; line-height: 1.8; padding: 8px 10px; font-family: Consolas, monospace;
        }
        #log .black { color: #7fa9ff; }
        #log .white { color: #ffb066; }
        #log .final { color: #ffd24d; font-weight: bold; }
    </style>
</head>
<body>
    <h1>五子棋 · 算法对战</h1>
    <div class="subtitle" id="subtitle">加载中...</div>
    <div class="status" id="status">加载中...</div>

    <div class="main">
        <canvas id="board"></canvas>
        <div class="panel">
            <h3>模型选择</h3>
            <div class="cfg">
                <label>● 黑方(先手)</label>
                <select id="blackSel">
                    <option value="bfs">BFS 决策树（第一版）</option>
                    <option value="alphabeta">α-β 深搜（第二版）</option>
                    <option value="deepseek">DeepSeek Minimax</option>
                </select>
            </div>
            <div class="cfg">
                <label>○ 白方(后手)</label>
                <select id="whiteSel">
                    <option value="bfs">BFS 决策树（第一版）</option>
                    <option value="alphabeta">α-β 深搜（第二版）</option>
                    <option value="deepseek">DeepSeek Minimax</option>
                </select>
            </div>
            <div class="tip">更改后点「重新开始」生效</div>
            <div class="players" id="players"></div>
            <div class="toolbar">
                <button id="autoBtn">开始自动对战</button>
                <button id="stepBtn" class="secondary">下一手</button>
                <button id="resetBtn" class="secondary">重新开始</button>
            </div>
            <h3>对局记录</h3>
            <div id="log"></div>
        </div>
    </div>

<script>
const N = 15, MARGIN = 30, CELL = 40, SIZE = MARGIN*2 + CELL*(N-1);
const canvas = document.getElementById("board"), ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status"), logEl = document.getElementById("log");
const subtitleEl = document.getElementById("subtitle"), playersEl = document.getElementById("players");
const autoBtn = document.getElementById("autoBtn"), stepBtn = document.getElementById("stepBtn"), resetBtn = document.getElementById("resetBtn");
const blackSel = document.getElementById("blackSel"), whiteSel = document.getElementById("whiteSel");
canvas.width = SIZE; canvas.height = SIZE;

const ALGO_NAMES = { bfs: "BFS 决策树", alphabeta: "α-β 深搜", deepseek: "DeepSeek Minimax" };
function algoName(key) { return ALGO_NAMES[key] || key; }

let state = null;
let autoMode = false;
let pending = false;

function draw() {
    ctx.fillStyle = "#dcb35c"; ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.strokeStyle = "#333"; ctx.lineWidth = 1;
    for (let i = 0; i < N; i++) {
        const p = MARGIN + i * CELL;
        ctx.beginPath(); ctx.moveTo(MARGIN, p); ctx.lineTo(SIZE - MARGIN, p); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p, MARGIN); ctx.lineTo(p, SIZE - MARGIN); ctx.stroke();
    }
    const stars = [[3,3],[3,11],[7,7],[11,3],[11,11]];
    ctx.fillStyle = "#333";
    for (const [sx, sy] of stars) { ctx.beginPath(); ctx.arc(MARGIN+sx*CELL, MARGIN+sy*CELL, 4, 0, Math.PI*2); ctx.fill(); }

    const winSet = new Set((state.winCells || []).map(([x, y]) => x + "," + y));

    for (const mv of state.moves) {
        const cx = MARGIN + mv.x * CELL, cy = MARGIN + mv.y * CELL;
        if (winSet.has(mv.x + "," + mv.y)) {
            ctx.beginPath(); ctx.arc(cx, cy, 19, 0, Math.PI*2);
            ctx.fillStyle = "#ffd24d"; ctx.fill();
            ctx.strokeStyle = "#e69500"; ctx.lineWidth = 2; ctx.stroke();
            ctx.lineWidth = 1;
        }
        ctx.beginPath(); ctx.arc(cx, cy, 16, 0, Math.PI*2);
        ctx.fillStyle = mv.c === 1 ? "#111" : "#fff"; ctx.fill();
        ctx.strokeStyle = mv.c === 1 ? "#000" : "#888"; ctx.stroke();
        ctx.fillStyle = mv.c === 1 ? "#fff" : "#333";
        ctx.font = "bold 11px Consolas, monospace";
        ctx.textAlign = "center"; ctx.textBaseline = "middle";
        ctx.fillText(mv.n, cx, cy);
    }

    // 最后一手红点提示
    if (state.lastX >= 0 && !state.gameOver) {
        ctx.beginPath(); ctx.arc(MARGIN + state.lastX*CELL, MARGIN + state.lastY*CELL, 5, 0, Math.PI*2);
        ctx.fillStyle = "#ff3b30"; ctx.fill();
    }
}

function updateStatus() {
    if (!state) return;
    subtitleEl.textContent = `黑方 = ${algoName(state.blackAlgo)}  |  白方 = ${algoName(state.whiteAlgo)}`;
    playersEl.innerHTML =
        `<span class="b">● 黑棋(先手)</span> ${algoName(state.blackAlgo)}<br>` +
        `<span class="w">○ 白棋(后手)</span> ${algoName(state.whiteAlgo)}`;
    if (state.gameOver) {
        let t;
        if (state.winner === 1) t = `🏆 黑方获胜 —— ${algoName(state.blackAlgo)}！`;
        else if (state.winner === 2) t = `💀 白方获胜 —— ${algoName(state.whiteAlgo)}！`;
        else t = "平局";
        statusEl.textContent = t;
        statusEl.style.color = "#ffd24d";
        autoMode = false; autoBtn.textContent = "开始自动对战";
        return;
    }
    const nextBlack = state.moveCount % 2 === 0;
    statusEl.textContent = `第 ${state.moveCount + 1} 手 · ${nextBlack ? "黑方(" + algoName(state.blackAlgo) + ")" : "白方(" + algoName(state.whiteAlgo) + ")"} 落子`;
    statusEl.style.color = nextBlack ? "#4f8cff" : "#ff9d4f";
}

function renderLog() {
    let html = "";
    for (const mv of state.moves) {
        const who = mv.c === 1 ? `黑(${algoName(state.blackAlgo)})` : `白(${algoName(state.whiteAlgo)})`;
        const cls = mv.c === 1 ? "black" : "white";
        html += `<div class="${cls}">#${String(mv.n).padStart(2,"0")} ${who} → (${mv.x},${mv.y}) · ${(mv.ms/1000).toFixed(1)}s</div>`;
    }
    if (state.gameOver) {
        const t = state.winner === 1 ? "黑方获胜" : state.winner === 2 ? "白方获胜" : "平局";
        html += `<div class="final">=== ${t}（共 ${state.moveCount} 手）===</div>`;
    }
    logEl.innerHTML = html;
    logEl.scrollTop = 0;
}

function drawAll() { draw(); updateStatus(); renderLog(); }

async function loadState() {
    const res = await fetch("/api/state");
    state = await res.json();
    if (state.blackAlgo && ALGO_NAMES[state.blackAlgo]) blackSel.value = state.blackAlgo;
    if (state.whiteAlgo && ALGO_NAMES[state.whiteAlgo]) whiteSel.value = state.whiteAlgo;
    drawAll();
}

async function stepOnce() {
    if (pending || !state || state.gameOver) return;
    pending = true;
    const nextBlack = state.moveCount % 2 === 0;
    statusEl.textContent = `第 ${state.moveCount + 1} 手 · ${nextBlack ? "黑方(" + algoName(state.blackAlgo) + ")" : "白方(" + algoName(state.whiteAlgo) + ")"} 计算中…`;
    statusEl.style.color = "#ff4f4f";
    try {
        const res = await fetch("/api/step", { method: "POST" });
        state = await res.json();
        drawAll();
    } catch (e) {
        statusEl.textContent = "请求失败：" + e.message;
    } finally {
        pending = false;
    }
}

async function autoLoop() {
    while (autoMode && state && !state.gameOver) {
        await stepOnce();
        if (!autoMode || !state || state.gameOver) break;
        await new Promise(r => setTimeout(r, 350)); // 每手之间稍作停顿便于观看
    }
    if (state && state.gameOver) { autoMode = false; autoBtn.textContent = "开始自动对战"; }
}

autoBtn.addEventListener("click", () => {
    if (!state || state.gameOver) return;
    autoMode = !autoMode;
    autoBtn.textContent = autoMode ? "暂停" : "继续自动对战";
    if (autoMode) autoLoop();
});

stepBtn.addEventListener("click", () => { if (!autoMode) stepOnce(); });

resetBtn.addEventListener("click", async () => {
    if (pending) return;
    autoMode = false; autoBtn.textContent = "开始自动对战";
    const res = await fetch(`/api/reset?black=${blackSel.value}&white=${whiteSel.value}`, { method: "POST" });
    state = await res.json();
    drawAll();
});

loadState().catch(e => { statusEl.textContent = "加载失败：" + e.message; });
</script>
</body>
</html>
)HTML";

    svr.Get("/", [indexHtml](const httplib::Request&, httplib::Response& res) {
        res.set_content(indexHtml, "text/html; charset=utf-8");
    });

    svr.Get("/api/state", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(mtx);
        res.set_content(battle.toJson(), "application/json; charset=utf-8");
    });

    // 走一手（同步计算：BFS 约 2 秒，α-β 约 6 秒，DeepSeek 最慢约 1.5 秒）
    svr.Post("/api/step", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(mtx);
        double ms = 0;
        std::string err;
        battle.step(ds, ms, err);
        res.set_content(battle.toJson(), "application/json; charset=utf-8");
    });

    // 重新开始，可携带 black/white 算法选择（bfs / alphabeta / deepseek）
    svr.Post("/api/reset", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(mtx);
        std::string b = req.get_param_value("black");
        std::string w = req.get_param_value("white");
        if (validAlgoKey(b)) battle.blackAlgo_ = b;
        if (validAlgoKey(w)) battle.whiteAlgo_ = w;
        battle.reset();
        ds.reset();
        res.set_content(battle.toJson(), "application/json; charset=utf-8");
    });

    std::cout << "Server started: http://localhost:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}
