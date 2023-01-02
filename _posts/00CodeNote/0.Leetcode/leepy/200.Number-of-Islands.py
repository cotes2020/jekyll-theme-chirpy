# 200. Number of Islands
# Medium

# Given an m x n 2d grid map of '1's (land) and '0's (water), return the number of islands.
# 1 是岛屿， 0 是水，计算有几个岛屿

# An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
# You may assume all four edges of the grid are all surrounded by water.

# Example 1:
# Input: grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# Output: 1

# Example 2:
# Input: grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# Output: 3


# ---------------------- solution java ----------------------
# check 每一个格格，
# - 格格 == 1
#   - check 每一个四周的格格，是 1 就翻转成 0
#   - check 每一个四周的格格，不是 1 不管
#   - 岛屿 + 1
# - 格格 == 0
#   - 不作为

# check 第二个格格，
# - 格格 == 1
#   - check 每一个四周的格格，是 1 就翻转成 0
#   - check 每一个四周的格格，不是 1 不管
#   - 岛屿 + 1
# - 格格 == 0
#   - 不作为

# class Solution {
#     public int numIslands(char[][] grid) {
#         // exception:
#         if(grid == null || grid.length == 0){
#             return 0;
#         }

#         // start
#         int numIslands = 0;
#         for (int i=0; i < grid.length; i++){
#             for (int j=0; j < grid[i].length; j++){
#                 // got the block
#                 if (grid[i][j] == '1'){
#                     // change it to 0
#                     numIslands += dfs(grid, i ,j);
#                 }
#             }
#         }
#         return numIslands;
#     }

#     public int dfs(char[][] grid, int i , int j){
#         // exception:
#         if(i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0'){
#             return 0;
#         }
#         grid[i][j] = '0';
#         dfs(grid, i-1, j);      // up
#         dfs(grid, i+1, j);      // down
#         dfs(grid, i, j-1);      // left
#         dfs(grid, i, j+1);      // right
#         return 1;
#     }
# }
# Runtime: 1 ms, faster than 99.99% of Java online submissions for Number of Islands.
# Memory Usage: 41.2 MB, less than 87.87% of Java online submissions for Number of Islands.


# ---------------------- solution py ----------------------
class Solution:
    def numIslands(self, grid):
        if grid is [] or len(grid) == 0:
            return 0

        count = 0

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    # start from the point, make all nearber 1 to 0
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != "1":
            return
        grid[i][j] = "0"
        self.dfs(grid, i + 1, j)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)


# Runtime: 124 ms, faster than 67.91% of Python online submissions for Number of Islands.
# Memory Usage: 21.2 MB, less than 87.00% of Python online submissions for Number of Islands.


# ---------------------- solution py ----------------------
# def numIslands(self, grid):
#     def sink(i, j):
#         if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
#             grid[i][j] = '0'
#             map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1))
#             return 1
#         return 0
# return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))


# ---------------------- solution java ----------------------
# public class Solution {
#     public int numIslands(char[][] grid) {
#         int islands = 0;
#         for (int i=0; i<grid.length; i++)
#             for (int j=0; j<grid[i].length; j++)
#                 islands += sink(grid, i, j);
#         return islands;
#     }
#     int sink(char[][] grid, int i, int j) {
#         if (i < 0 || i == grid.length || j < 0 || j == grid[i].length || grid[i][j] == '0')
#             return 0;
#         grid[i][j] = '0';
#         for (int k=0; k<4; k++)
#             sink(grid, i+d[k], j+d[k+1]);
#         return 1;
#     }
#     int[] d = {0, 1, 0, -1, 0};
# }


#
