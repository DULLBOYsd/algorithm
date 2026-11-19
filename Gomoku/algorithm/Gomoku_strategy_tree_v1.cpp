#include <iostream>

// STL container
#include <deque>
#include <vector>
#include <array>

// algorithm header
#include <regex>

// string header
#include <string> 

#include <string_view>
using namespace std;

static constexpr int N = 15; // 定义棋盘边长
static constexpr int DEPTH = 2; // 定义遍历深度

array<array<int, N>, N> g_board; // 棋盘中 1 代表我方棋子，-1 代表对方棋子

// ===== 定义结构体 =====
// 定义策略树节点
struct TreeNode {
  vector<TreeNode*> children;
  // trend 为 1 时我方下棋，此处值越高越好（收集下一层子节点中最大的值）；trend 为 0 时对方下棋，此时值越小越好（收集下一层子节点中最小的值）  
  array<array<int, N>, N> board;
  int val;
  bool isEnemy; // 此处趋势中 true 代表此处为对方先手，false 代表我方先手
  int x, y; // 读入上一步填入的棋子信息
};

struct FastPattern {
  std::string_view core;   // 棋形主体
  std::string_view left;   // 允许的左边界字符集，空串 = 不限
  std::string_view right;  // 允许的右边界字符集，空串 = 不限
  int score;
};

// 纯数据表
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
  // 单独的 "1" 不在这里处理，最后按剩余个数统一计分
};


// 获取新节点
TreeNode* getNewNode(const array<array<int, N>, N>& board, int val, bool isEnemy, int x, int y) {
  TreeNode* node = new TreeNode();
  node->board = board; 
  node->val = val;
  node->isEnemy = isEnemy;
  node->x = x;
  node->y = y;
  return node;
}

// 将数组数值转换成字符串
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

// 计算字符串的值
static bool inSet(char c, std::string_view s) { return s.find(c) != std::string_view::npos; }

int evaluateLine(std::string line) {
  int score = 0;
  if (line.find('1') == std::string::npos) return 0;   // 快速短路

  for (const auto& p : fastDict) {
    size_t pos = 0;
    while ((pos = line.find(p.core, pos)) != std::string::npos) {
      bool lb = p.left.empty() || (pos > 0 && inSet(line[pos-1], p.left));
      size_t end = pos + p.core.size();
      bool rb = p.right.empty() || (end < line.size() && inSet(line[end], p.right));
      if (lb && rb) {
        score += p.score;
        line.replace(pos, p.core.size(), p.core.size(), '2');  // 屏蔽已计分区域
        pos += p.core.size();
      } else {
        ++pos;
      }
    }
  }
  score += 10 * (int)std::count(line.begin(), line.end(), '1');  // 零散单点
  return score;
}

// 获取当前棋盘对应的评分
int getNowVal(array<array<int, N>, N>& board, bool isEnemy) {
  int score = 0; 
  
  // type: 0: row, 1: col, 2: 从左上斜向右下, 3: 从左下斜向右上
  // 计算横纵方向的分数
  // 计算横向分数
  for (int x = 0; x < N; ++x) {
    string line = getLineString(board, x, 0, 0, isEnemy);
    score += evaluateLine(line);        
  }
  // 计算纵向分数
  for (int y = 0; y < N; ++y) {
    string line = getLineString(board, 0, y, 1, isEnemy);
    score += evaluateLine(line);  
  }

  // 计算从左上斜向右下的分数（如果斜着位数不满 5 就跳过了）
  for (int x = 0; x <= N - 5; ++x) {
    string line = getLineString(board, x, 0, 2, isEnemy);
    score += evaluateLine(line);
  }
  for (int y = 1; y <= N - 5; ++y) {
    string line = getLineString(board, 0, y, 2, isEnemy);
    score += evaluateLine(line);
  }
  
  // 计算从左下斜向右上的分数
  for (int x = N - 1; x >= 4; --x) {
    string line = getLineString(board, x, 0, 3, isEnemy);
    score += evaluateLine(line);
  }
  for (int y = 1; y <= N - 5; ++y) {
    string line = getLineString(board, N - 1, y, 3, isEnemy);
    score += evaluateLine(line);
  }

  return score;
}

// 计算该棋子周围的分数，作为判定棋子落点的依据
int aroundScore(TreeNode* node) {
  int score = 0; 
  for (int i = max(node->x - 1, 0); i <= min(node->x + 1, N - 1); ++i) {
    for (int j = max(node->y - 1, 0); j <= min(node->y + 1, N - 1); ++j) {
      if (i == node->x && j == node->y) continue;
      if (node->board[i][j]) score++;
    }
  }
  return score;
}

// 将节点插入树中
void insertTreeNode(TreeNode* root, deque<TreeNode*>& dq, int depth) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      auto board = root->board;
      if (board[i][j] != 0) continue;
      board[i][j] = root->isEnemy ? -1 : 1;
     
      // 读入当前的评分  
      int val;
      if (depth == DEPTH - 1) {
        // 同时评估双方的得分
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
  }
  
  // 候选排序，同分按照离棋子更近优先
  auto cmp = [](TreeNode* node1, TreeNode* node2) {
    if (node1->val == node2->val) return aroundScore(node1) > aroundScore(node2);
    return node1->val > node2->val;
  };
  sort(root->children.begin(), root->children.end(), cmp);

  return;
}

// 使用 BFS 进行节点的插入
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

// 获取最佳评分的数组
array<array<int, N>, N> bestBoard;

// 遍历策略树
int traverseDecisionTree(TreeNode* root, int depth=0) {
  if (root->children.empty()) return root->val;
  int val = root->val;
  
  if (root->isEnemy) {
    for (int i = 0; i < root->children.size(); ++i) {
      int ithVal = traverseDecisionTree(root->children[i], depth + 1);
      if (val > ithVal) {
        val = ithVal;
        if (depth == 0) bestBoard = root->children[i]->board;
      } 
    }
  } else {
    for (int i = 0; i < root->children.size(); ++i) {
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

int main() {
  // 获取变量
  array<array<int, N>, N> board = {};
  TreeNode* root = getNewNode(board, INT_MIN, false);

  // 构造策略树
  constructTree(root);
  
  traverseDecisionTree(root);
  return 0;
}
