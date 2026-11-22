#include <iostream>

// STL container
#include <deque>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>

// algorithm header
#include <regex>

// string header
#include <string> 
#include <string_view>

// algorithm header
#include <climits>
#include <algorithm>

using namespace std;

static constexpr int N = 15; // 定义棋盘边长
static constexpr int DEPTH = 4; // 定义遍历深度

// ===== 定义结构体 =====
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
  {"011112", "02",  "",    100000},
  {"11110",  "02",  "02",  100000},
  {"011101", "02",  "02",  100000},
  {"101110", "02",  "02",  100000},
  {"11011",  ""  ,  ""  ,  100000},
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

long long evaluateLine(std::string line) {
  long long score = 0;
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
long long getNowVal(array<array<int, N>, N>& board, bool isEnemy) {
  long long score = 0; 
  
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
int aroundScore(const array<array<int, N>, N>& board, int x, int y) {
  int score = 0; 
  for (int i = max(x - 1, 0); i <= min(x + 1, N - 1); ++i) {
    for (int j = max(y - 1, 0); j <= min(y + 1, N - 1); ++j) {
      if (i == x && j == y) continue;
      if (board[i][j]) score++;
    }
  }
  return score;
}

// 将可能评分的位置填入 possiblePos 数组
void fillPossiblePos(array<array<bool, N>, N>& possiblePos, int x, int y) {
  for (int i = max(0, x - 2); i <= min(N - 1, x + 2); ++i) 
    for (int j = max(0, y - 2); j <= min(N - 1, y + 2); ++j) 
      possiblePos[i][j] = true;
}

// 活五直接判胜
bool absoluteResult(const array<array<int, N>, N>& board, int x, int y, int color) {
  bool win = false; 
  static const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

  for (auto& d : dirs) {
    int cnt = 1;

    // 正向统计
    int nx = x + d[0], ny = y + d[1];
    while (nx >= 0 && nx < N && ny >= 0 && ny < N && board[nx][ny] == (color ? -1 : 1)) cnt++, nx += d[0], ny += d[1];

    // 反向统计
    nx = x - d[0], ny = y - d[1];
    while (nx >= 0 && nx < N && ny >= 0 && ny < N && board[nx][ny] == (color ? -1 : 1)) cnt++, nx -= d[0], ny -= d[1];

    // 五连判胜
    if (cnt >= 5) {
      win = true;
      break;
    }
  }
  return win;
}

// 检查到有四颗连子的情况将其加入候选遍历数组中
bool checkFour(const array<array<int, N>, N>& board, int x, int y, int color) {
  bool isFour = false;
  static const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

  for (auto& d : dirs) {
    int cnt = 1; 
    int space = 0;

    // 1.先正向统计，再反向统计
    int nx = x + d[0], ny = y + d[1];
    while (nx >= 0 && nx < N && ny >= 0 && ny < N && board[nx][ny] != (color ? 1 : -1) && space == 0) {
      if (board[nx][ny] == 0) space++;
      else cnt++;
      nx += d[0], ny += d[1];
    }
    nx = x - d[0], ny = y - d[1];
    while (nx >= 0 && nx < N && ny >= 0 && ny < N && board[nx][ny] != (color ? 1 : -1) && space == 0) {
      if (board[nx][ny] == 0) space++;
      else cnt++;
      nx -= d[0], ny -= d[1];
    }
    
    if (cnt + space >= 5) {
      isFour = true;
      break;
    }
    
    cnt = 1; 
    space = 0;
    
    // 2.先反向统计，再正向统计
    nx = x - d[0], ny = y - d[1];
    while (nx >= 0 && nx < N && ny >= 0 && ny < N && board[nx][ny] != (color ? 1 : -1) && space == 0) {
      if (board[nx][ny] == 0) space++;
      else cnt++;
      nx -= d[0], ny -= d[1];
    }
    nx = x + d[0], ny = y + d[1];
    while (nx >= 0 && nx < N && ny >= 0 && ny < N && board[nx][ny] != (color ? 1 : -1) && space == 0) {
      if (board[nx][ny] == 0) space++;
      else cnt++;
      nx += d[0], ny += d[1];
    }
    
    if (cnt + space >= 5) {
      isFour = true;
      break;
    }
  }
  return isFour;
}

// 将四子的情况加入到遍历数组中
void pushFour(const array<array<int, N>, N>& board, const vector<pair<int, int>>& cands, vector<pair<int, int>>& existFour) {
  set<pair<int, int>> seen(existFour.begin(), existFour.end()); 

  // 遍历 cands 中前 topK 位，如果 existFour 中没有，就追加到 existFour 中，并对 existFour 排序
  for (int topK = 0; topK < 15 && topK < (int)cands.size(); ++topK) {
    if (seen.find(cands[topK]) == seen.end()) {
      existFour.push_back(cands[topK]);
      seen.insert(cands[topK]);
    }
  }

  // 对 existFour 排序
  sort(existFour.begin(), existFour.end(), [&](const pair<long long, long long>& a, const pair<long long, long long>& b) {
    return aroundScore(board, a.first, a.second) > aroundScore(board, b.first, b.second);
  });
}

array<array<int, N>, N> bestBoard;

// 使用 DFS + alphabeta 进行节点的插入，同时计算是否可以舍弃当前之后的分数
long long alphabeta(array<array<int, N>, N> board, array<array<bool, N>, N> possiblePos, long long alpha, long long beta, int x, int y, int depth=0) { 
  bool color = depth % 2; // color 为 true 是对手，false 为己方
  board[x][y] = color ? -1 : 1;
  fillPossiblePos(possiblePos, x, y);
  if (depth == DEPTH) return getNowVal(board, false) - getNowVal(board, true);

  if (absoluteResult(board, x, y, color)) return color ? INT_MIN : INT_MAX;

  // 加入候选 + 排序，然后再递归
  vector<pair<int, int>> cands;
  vector<pair<int, int>> existFour; // 强制将当前 x，y 对应新棋子周围存在四子的情况加入到 cands 里面
  
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      if (possiblePos[x][y] && !board[x][y]) {
        cands.emplace_back(x, y); 
        if (checkFour(board, x, y, !color) || checkFour(board, x, y, color)) existFour.emplace_back(x, y);
      }
    }
  }

  sort(cands.begin(), cands.end(), [&](const pair<long long, long long>& a, const pair<long long, long long>& b) {
    return aroundScore(board, a.first, a.second) > aroundScore(board, b.first, b.second);
  });

  pushFour(board, cands, existFour);

  long long v = depth % 2 ? INT_MIN : INT_MAX;
  if (color) { // max 节点
    for (int topK = 0; topK < (int)existFour.size(); ++topK) {
      long long x = existFour[topK].first, y = existFour[topK].second;
      if (!possiblePos[x][y] || board[x][y]) continue;
      long long cv = alphabeta(board, possiblePos, alpha, beta, x, y, depth + 1);
      if (cv > v) v = cv;
      alpha = max(alpha, v);
      if (beta <= alpha) break; // 一次比较，累积窗口
    }
  } else { // min 节点
    for (int topK = 0; topK < (int)existFour.size(); ++topK) {
      long long x = existFour[topK].first, y = existFour[topK].second; 
      if (!possiblePos[x][y] || board[x][y]) continue;
      long long cv = alphabeta(board, possiblePos, alpha, beta, x, y, depth + 1);
      if (cv < v) v = cv;
      beta = min(beta, v);
      if (beta <= alpha) break;
    }
  }
  return v;
} 

void getBestPos(array<array<int, N>, N> board, array<array<bool, N>, N> possiblePos, int& bestX, int& bestY) {
  long long bestV = INT_MIN;
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      if (!possiblePos[x][y] || board[x][y]) continue;
      long long cv = alphabeta(board, possiblePos, bestV, INT_MAX, x, y, 0);
      if (cv > bestV) {
        bestV = cv; 
        bestX = x;
        bestY = y;
      }
    }
  }
}

int main() {
  // 获取变量
  array<array<int, N>, N> board = {};
  array<array<bool, N>, N> possiblePos = {};

  // 将 board 上已有的棋子的周围 5 * 5 填入 possiblePos里面，代表这些位置都是可能取到的 
  for (int x = 0; x < N; ++x) {
    for (int y = 0; y < N; ++y) {
      if (board[x][y] != 0) fillPossiblePos(possiblePos, x, y);
    }
  }

  int bestX, bestY;
  getBestPos(board, possiblePos, bestX, bestY);
  
  return 0;
}


