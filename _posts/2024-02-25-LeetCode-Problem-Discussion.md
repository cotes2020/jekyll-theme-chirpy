---
title: Discussing Leetcode problems and their different solutions
author: arpiku 
date: 2024-02-24 19:42:00 +0530
categories: [C++, Programming, Competitive Programming, Leetcode]
tags: [C++, Algorithms, Leetcode]
pin: false 
math: true
---

### Two Sum 
- The problem is rather simple when going with a naive solution, using a O($n^2$)  solution by lopping through all the unique
pairs possibe combinations (not considering the order) and returning the one that is meets our requirement.
- Time Complexity O($n^2$), Space Complexity - O($1$)

{% raw %}
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        if(nums.empty()) 
            return {};
        for(int i = 0; i<nums.size(); i++) 
            for(int j = i+1; j<nums.size(); j++)
                if(nums[i]+nums[j]==target)
                    return {i,j};
        return {};
        
    }
};
```
{% endraw %}

- The solution is rather slow when n >> 1. How do we improve it?
- We consider the fact that if the solution exist in the array, then the two numbers get related by being complement of each 
other, as only one unique solution leads to the correct sum.
- I.E if a + b = T, then a = T - b, and b = T - a.
- Now while iterating we can store the positions against these complements
- I.E fn(T-b) = pos(a), fn(T-a) = pos(b).
- We build a hash map to store the relevant information and we build it while iterating through the array itself.

{% raw %}
```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        if(nums.size() ==  0) 
            return {};
        std::unordered_map<int,int> m;
        for(int i = 0; i<nums.size(); i++) {
            if(m.find(target - nums[i]) != m.end())
                return {i, m[target-nums[i]]};
            m[nums[i]] = i;
        }

        return {};
    }
};
```
{% endraw %}

- Notice how the time and space complexity has changed, we iterate through the array at most in n steps, however it may require us to construct a map that can increase in size as much as the array. Hence

- Time Complexity O($n$), Space Complexity - O($n$)


## Valid Parentheses
- Let's first see an interesting solution to this problem that uses counting a single variable to find the solution.
- It doesn't work for all the cases, but does provide an insight into the idea of using a single variable to keep more information just than the count of something, but may relate to the structure and values provide powerful solutions.

- **The Following Solution is Wrong!!**

{% raw %}
```cpp
class Solution {
public:
    bool isValid(string s) {
        int balance = 0; // To keep track of parentheses balance

    for (char c : s) {
        if (c == '(') {
            balance++;
        } else if (c == ')') {
            balance--;
        } else if (c == '[') {
            balance++;
        } else if (c == ']') {
            balance--;
        } else if (c == '{') {
            balance++;
        } else if (c == '}') {
            balance--;
        }
    }

    return balance == 0;
    }
};
```
{% endraw %}

- It will fail in case two different types of bracket close each other.
- Time Complexity - O($n$), Space Complexity - O($1$)
- Let's see a valid solution in python
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        memo = {'}':'{', ')':'(', ']':'['}
        for ch in s:
            if(len(stack) == 0 and ch in memo):
                return False
            if(len(stack) != 0 and ch in memo):
                if(stack.pop() != memo[ch]):
                    return False
            else:
                stack.append(ch)
        return len(stack) == 0
```
- Time Complexity - O($n$), Space Complexity O($n$)
- We introduce the concept of stack, but why?
- In this problem we are concerned with how things are ordered, and we know if a bracket is opened than it must be closed too in 
order, which is what the stack helps us with, let's look at a cpp implementation of a solution.

{% raw %}
```cpp
  class Solution {
  public:
      char topAndPop(std::stack<char>& s) {
          const char ch = s.top();
          s.pop();
          return ch;
      }
      bool isValid(string s) {
          std::stack<char> ch_stack;
          for(auto& ch:s) {
              if(hmap.find(ch) != hmap.end())
                  ch_stack.push(hmap[ch]);
              else if(ch_stack.empty() || ch != topAndPop(ch_stack))
                  return false;
          }
          return ch_stack.empty();
      }
  };
```
{% endraw %}
- Notice how in cpp stack.pop() does not return a value, so we wrap that functionality using another function.
- Time Complexity O($n$), Space Complexity O($n$)



## Merge Two Lists
- This problem requires merge two lists, think of a function that takes the top front nodes and compares them and then properly links them in order and them moves to next comparison step.
- The iterative solution will be:
{% raw %}
```cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* curr = &dummy;
        while(l1 && l2) {
            if(l1->val > l2->val)  {
                curr->next = l2;
                l2 = l2->next;
            } else {
                curr->next = l1;
                l1 = l1->next;
            }
            curr = curr->next;
        }

        if(l1 != NULL) {
            curr->next = l1;
        } else {
            curr->next = l2;
        }

        return dummy.next;
        
    }
};
```
{% endraw %}
- Time Complexity O($n$), Space Complexity O($1$)
- Thing to notice are the two variables we created, the problem that they solve is interesting.
- When iterating through a linked list, if you have an iterator node, that gets modified continuously, 
we cannot return that, so we need to store the value of original head somewhere.
- But since the head of the returned list maybe from either of the LLs we cannot not know before hand 
which to choose.
- So we create a dummy variable that exists before either of the linked list and then return the next 
node of dummy which will be the sorted head of our answer.
- We use another variable curr to iterate and correctly join our lists.
- Things to notice:
	- See we only use a single if statement and do not do and explicit comparison on the else statement as 
  '>' not being true implies '<=' being true for the variable, one can miss this and write code like this.

{% raw %}
```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(!list1)
            return list2;
        if(!list2)
            return list1;
        ListNode dummy(0);
        ListNode* curr = &dummy;

        while(list1&&list2) {
            if(list1->val > list2->val) {
                curr->next = list2;
                list2 = list2->next;
            }
            if(list2->val > list1->val)  {
                curr->next = list1;
                list1 = list1->next;
            } curr = curr->next;
        }
        if(list1)
            curr->next = list1;
        if(list2)
            curr->next = list2;
        return dummy->next;
    }
};
```
{% endraw %}

- This code fails because both statements can fail and result in an access attempt on null pointer ('curr->next')

- **Is there a way to remove the dummy variable?**
- What about recursion, each recursive step is function waiting for the function within to pass it the results, the very structure of recursion can be used to replace the need of a dummy variable.
- Let's see!

{% raw %}
```cpp

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        if(!list1)
            return list2;
        if(!list2)
            return list1;
        if(list1->val > list2->val) {
            list2->next = mergeTwoLists(list1,list2->next);
            return list2;
        }
        else  {
            list1->next = mergeTwoLists(list1->next,list2);
            return list1;
        }
    }
};
```
{% endraw %}
- Time Complexity O($m+n$) Space Complexity O($m+n$)


## Best Time to Buy and Sell Stocks
- Well only if I could figure that out!
- Let's analyse the problem:
	- The order matters in this problem, we can use two variables one of which is always lower than the other one, making sure we can't sell even before buying.
	- We store the maxProfit in a variable, and update it if the profit for a particular pair of days is greater.
	- In case the profit for a pair of days is -ve, that implies there is price lower than current price using which more profit can be made potentially.
- The following approach uses two variables to find the max profit
{% raw %}
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty())
            return 0;
        int maxProfit = 0;
        int currProfit = 0;
        int buyDay = 0;
        int sellDay = 1;
        while(sellDay < prices.size()) {
            currProfit = prices[sellDay] - prices[buyDay];
            if(currProfit > maxProfit) 
                maxProfit = currProfit;
            if(currProfit < 0) 
                buyDay = sellDay;
            sellDay++;
        }
        return maxProfit;
    }
};
```
{% endraw %}
- Time ComplexityO($n$), Space Complexity O($1$)
- One can also solve this problem by finding the minimum price, and calculating the relative profit w.r.t minimum price and return the maximum, but we do not need to make sure that the order is maintained.
- The order thing can be taken care easily by the very nature of iteration itself.

{% raw %}
```cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if(prices.empty())
            return true;
        int minPrice = prices[0];
        int maxProfit = 0;
        for(auto& price: prices) {
            minPrice = std::min(minPrice,price);
            int currProfit = price - minPrice;
            maxProfit = std::max(maxProfit, currProfit);
        }
        return maxProfit;
    }
};
```
{% endraw %}
- Timpe Complexity O($n$), Space Complexity O($1$)

## Invert a Binary Tree
- Binary Trees in general lend themselves well to recursion, imagine
	- Being a node, if its null return
	- swap the children otherwise
	- call the function on it's children then just simply return the root that was given

{% raw %}
```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(!root)
            return root;
        std::swap(root->left, root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```
{% endraw %}
- Timpe Complexity O($n$), **Space Complexity O($h$)->O($log(n)$) if tree is balanced**

- Notice the order, we go to left revert everything then we go to the right and revert everything.
- Since the reversal of both sides is independent these steps can be parallelised?
- This solution was an implementation of depth first search algorithm
- We can also solve it using breadth first search 
- Let's see that solution:

{% raw %}
```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if(!root) 
            return root;
        std::queue<TreeNode*> q;
        q.push(root);
        while(!q.empty()) {
            TreeNode* currNode = q.front();
            q.pop();
            if(currNode->left)
                q.push(currNode->left);
            if(currNode->right)
                q.push(currNode->right);
            std::swap(currNode->left, currNode->right);
        }
        return root; 
    }
};
```
{% endraw %}

- Time Complexity-O($n$), SpaceComplexity O(w) -> (O(logn),O(n))
- Interesting thing to notice is the **swap statement**, one can put it just below the pop() statement too.

## Valid Anagram
- We have two string is they are anagrams ,then their sorted version will be have to be equal (always?)

{% raw %}
```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        std::sort(s.begin(),s.end());
        std::sort(t.begin(),t.end());
        return s == t;
    }
};
```
{% endraw %}
- TO($nlong$),Space Complexity-O($1$)
- The above solution is rather straight forward and easy, but what if we are not allowed to reverse the lists?

{% raw %}
```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        unordered_map<char,int> ms;
        unordered_map<char,int> mt;
        for(auto& ch:s) {
            ms[ch]++;
        }
        for(auto& ch:t) {
            mt[ch]++;
        }
        for(auto& z:ms) {
            if(z.second != mt[z.first])
                return false;
        }
        return true;
    }
};
```

{% endraw %}
- Time Complexity O($n$), SO($n$)
- Here's another cooler looking piece of code! , which I wrote a little later on 

```cpp
class Solution {
public:
    bool isAnagram(string s, string t) {
        auto stringHasher = [](string s) {
            std::unordered_map<char,int> m;
            for(auto& ch:s) {
                m[ch]++;
            }
            return m;
        };
        return stringHasher(s) == stringHasher(t);
    }
};
```

## Binary Search
- This solution requires a simple implementation of binary search.


{% raw %}
```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0;
        int r = nums.size() -1;
        int mid = 0;
        while(r>=l) {
            mid = l + (r-l)/2;
            if(nums[mid] < target) {
                l = mid + 1;
            }
            else if (nums[mid] > target) {
                r = mid - 1;
            }
            else {
                return mid;
            }
        }
        return  -1;
    }
};
```
{% endraw %}
- TO(logn), SO(1)
- One important thing to notice is the positioning of 'if else statement', i.e the following code


{% raw %}
```cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l = 0;
        int r = nums.size() -1;
        int mid = 0;
        while(r>=l) {
            mid = l + (r-l)/2;
            if(nums[mid] < target) {
                l = mid + 1;
            }
            if (nums[mid] > target) {
                r = mid - 1;
            }
            else {
                return mid;
            }
        }
        return  -1;
    }
};
```
{% endraw %}
- Will not work, as the target may not exist in the array, the code looks rather similar in the above two cases, I have 
that mistake, following agains is a bit cooler looking code.

{% raw %}
```cpp
class Solution {
public:
    int _search(vector<int>& nums, int target, size_t hi, size_t lo) { 
        std::cout << hi << lo << std::endl;
        if(hi >= nums.size() || lo < 0 || lo > hi) return -1; // Check for out of bounds
        int mid = lo + (hi - lo)/2; //Look at the middle
        if(target == nums[mid]) return mid; // Found it!Here
        if(target > nums[mid]) return _search(nums,target,hi,mid+1);//Nah! //You Look Higher
        return _search(nums,target,mid-1,lo); //You Look Lower!
    }
    int search(vector<int>& nums, int target) {
        size_t hi = nums.size()-1;
        size_t lo = 0;
        return _search(nums,target,hi,lo);
    }
};
```
{% endraw %}



## Flood Fill
- We can do a dfs, i.e we will go the first element, check whether it does or does not have the correct color, if it does then we simply return.
- If not we change it and call the function on all the neighbours that are valid.

{% raw %}
```cpp
class Solution {
public:
    void _floodFill(vector<vector<int>>& image, int sr, int sc, int color, int org) {
        if(sr >=image.size() || sr < 0 || sc < 0 || sc >=image[0].size())
            return;

        if(image[sr][sc] == org) {
            image[sr][sc] = color;
        _floodFill(image,sr+1,sc,color,org);
        _floodFill(image,sr-1,sc,color,org);
        _floodFill(image,sr,sc+1,color,org);
        _floodFill(image,sr,sc-1,color,org);
        }
    }
vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color)  { 
        if(image.empty())   
            return {};
        if(image[sr][sc] == color)
            return image;
        int org = image[sr][sc];
        _floodFill(image,sr,sc,color, org);
        return image;
    }
};
```
{% endraw %}
- The above code uses depth first search, imagine all the image points as a node connected four way with other ndoes.
- We can also use breathd first search to do the same.

{% raw %}
```cpp
class Solution {
public:
    vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int color) {
        if(image.empty())
            return {};
        if(image[sr][sc] == color)
            return image;

        std::queue<std::pair<int,int>> q;
        const int dr[] = {-1,1,0,0};
        const int dc[] = {0,0,-1,1};
        const int R = image.size()-1;
        const int C = image[0].size()-1;
        const int org = image[sr][sc];

        q.push(std::make_pair(sr,sc));


        while(!q.empty()) {
            int r,c;
            std::tie(r,c) = q.front();
            q.pop();
            image[r][c] = color;
            for(int i = 0; i<4; i++) {
                int nr = r + dr[i];
                int nc = c + dc[i];
                if(nc <= C && nc >= 0 && nr >= 0 && nr <=R && image[nr][nc] == org)
                    q.push(std::make_pair(nr,nc));
                
            }
        }

        return image;
        
    }
};
```
{% endraw %}


## Lowest Common Ancestor
- Again we have a binary search tree, it's very structure gives us a rather elegant solution.
{% raw %}
```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root)
            return root;
        if(p->val > root->val && q->val > root->val)
            return lowestCommonAncestor(root->right, p,q);
        if(p->val < root->val && q->val < root->val) 
            return lowestCommonAncestor(root->left,p,q);
        return root;
    }
};
```
{% endraw %}

- Look how the very structure dictates the direction of movement for search, just like the binary search we had seen before.
- Above is a **dfs**  approach, let's see how a **bfs** approach will look like.


{% raw %}
```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */

class Solution {
public:
   TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root) {
        return root;
    }

    // Create a queue for BFS
    std::queue<TreeNode*> bfsQueue;
    bfsQueue.push(root);

    while (!bfsQueue.empty()) {
        TreeNode* current = bfsQueue.front();
        bfsQueue.pop();
        std::cout << current->val << std::endl;

        if ((current->val >= p->val && current->val <= q->val) || (current->val <= p->val && current->val >= q->val)) {
            return current; // Found the LCA
        }

            if(current->right)
                bfsQueue.push(current->right);

            if (current->left) {
                bfsQueue.push(current->left);
        }
    }

    return NULL; // LCA not found
}
};
```
{% endraw %}

- The **IMPORTANT** thing is the equality check, see it's not just '<>' but '<>='. 

- Follwing is another solution, the two lines above make the code run faster! Do not use them for actual code though.

```cpp

// Little trick to give your code steroids
 #pragma GCC optimize("Ofast")
static auto _ = [] () {ios_base::sync_with_stdio(false);cin.tie(nullptr);cout.tie(nullptr);return 0;}();


class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(!root)
            return root;
        if(root->val > p->val && root->val > q->val)
            return lowestCommonAncestor(root->left,p,q);
        if(root->val < p->val && root->val < q->val)
            return lowestCommonAncestor(root->right,p,q);
        return root;
    }
};
```

## Balanced Binary Tree
- For this problem the important insight is solving the problem of figuring out the height of the left and right subTrees.
- We can write another function that specifically gets the height of a subtree
- We see if the difference of two subtree height is greater than 1, if it is then , we simple return true.


{% raw %}
```cpp
class Solution {
public:
    int getHeight(TreeNode* node) {
        if(!node)
            return 0;
        return std::max(1 + getHeight(node->right), 1 + getHeight(node->left));
    }


    bool isBalanced(TreeNode* root) {
        if(!root)
            return true;
        if(abs(getHeight(root->left) - getHeight(root->right) > 1))
            return false;
        return isBalanced(root->right) && isBalanced(root->left);
        
    }
};
```
{% endraw %}
-  TO, SO
- The thing to notice is how we use && operator in our recursive function call.

## Linked List Cycle.
- This problem uses property of numbers in order to find the cycle.
- Think of hands of clocks they move at different speeds but come at the same point through out the day.
- That point is the LCM of the hands.
- We can use the same idea to find if there is a cycle, imagine two iterators going through the linked list, with one moving at a higher speed, if there is a cycle then they will eventually reach the same node.
- If the cycle is not there, well then our loop will simply exit and result in false being returned.


{% raw %}
```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* s = head;
        ListNode* f = head;
        while(f && f->next) {
            s = s->next;f = f->next->next;
            if (s == f)
                return true;
        } return false;
        
    }
};
```
{% endraw %}
- Something interesting to notice is about C++ code is that following line doesn't work.


{% raw %}
```cpp
        ListNode* s,f;
        s = head;
        f = head;
```
{% endraw %}
- The correct way of doing it is:
{% raw %}
```cpp
        ListNode *s,*f;
        s = head;
        f = head;
```
{% endraw %}
- More you know! This is becomes type\* a,b means type \*a, b not \*a \*b.
- One more thing to keep to keep the idea in the head, is to think of the smalled linked list possible, it will have atleast
two nodes, otherwise you can make a mistake in checking for the right condition in the while loop.

## Implement Queue using stacks.
- This question is rather trivial and not fun : (.
- But here is the code
{% raw %}
```cpp
class MyQueue {
private:
    std::stack<int> stack1;
    std::stack<int> stack2;
public:

    
    MyQueue() {
    }
    
    void push(int x) {
        stack1.push(x);
    }
    
    int pop() {
        int val = 0;
        if(stack2.empty())
            while(!stack1.empty()) {
                stack2.push(stack1.top());
                    stack1.pop();
            }
        
        val = stack2.top();
        stack2.pop();
        return val;
        
    }
    
    int peek() {
                if(stack2.empty())
            while(!stack1.empty()) {
                stack2.push(stack1.top());
                    stack1.pop();
            }
        return stack2.top();
    }
    
    bool empty() {
        return stack1.empty()&&stack2.empty();
    }
};

```
{% endraw %}
- Imagine a deck of cards as the stack, implementing queue is similar to rotating the stack, i.e getting it upside down.
- This is done using the second stack, take everything in s1 and put in s2
	- s1 -> s2
- Now the top of the s2 will be the bottom piece in s1, hence we successfully implement FIFO, using FILO. Amazing!


## First Bad Version
- The problem here is not that of writing the algorithm simply, because following is valid solution.
{% raw %}
```cpp
class Solution {
public:
    int firstBadVersion(int n) {
        int i = 1;
        while(i<=n) 
            if(isBadVersion(i++))
                return i-1;
        return -1;
    }
};
```
{% endraw %}
- The solution they want from us is the following, which is another implementation of **binary search** on the very number rather than an array.
{% raw %}
```cpp
class Solution {
public:
    int firstBadVersion(int n) {
        int l=1;
        int h=n;
        while(l<h){
            const int mid=l+(h-l)/2;
            if(isBadVersion(mid)){
                h=mid;
            }
            else{
                l=mid+1;
            }
        }
        return l;
    }
};
```
{% endraw %}


## Ransom Note
- [[hashMap]] [[string]] [[counting]]
- This is an important algorithm for kidnappers, and have been widely used by Big Crime, so prepare it well before interviewing for the Mafia, Cartel etc.
- The idea is rather simple, count all the characters available in magazine and store the relevant counts against the characters in a hash table.
- Then iterate through the ransomNote string, if there is a character missing in magazine hash map or that character count has reached zero we return fasle.
{% raw %}
```cpp
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        if(magazine.size() == 0)
            return false;
        unordered_map<char,int>  m;
        for(auto& ch:magazine) {
            m[ch]++;
        }

        for(auto& ch:ransomNote) {
            if(m.find(ch) == m.end() || m[ch] == 0)
                return false;
            m[ch]--;
        }
        return true;
    }
};
```
{% endraw %}

- One can solve this problem other ways too, we can delete the charater from magazine as we iterator through the ransom Note, 
and in the end if there are still characters left in the ransom note, then we know that magazine didn't have enough letters.
- That algorithm will be slower as well will have to search for each character in the next string.
- Other would be making a Trie structure and seeing if one is a possible subtree of the other.

## Climbing Stairs
- This problem is just like implementing a recursive memoized function for Fibonacci series.
{% raw %}
```cpp
class Solution {
public:
    int cs(unordered_map<int,int>& m, int n) {
        if(m.find(n) == m.end())
            m[n] = cs(m,n-1)+cs(m,n-2);
        return m[n];
    }

    int climbStairs(int n) {
        if(!n)
            return 0;
        unordered_map<int,int> hmap = {{1,1}, {2,2}};
        return cs(hmap,n);
    }
};```
{% endraw %}
- Simple stuff!

## Longest Palindrome
- This problem introduces interesting idea to solve it, imagine you are given a string, consider all the elements with even 
count and start filling the string at both ends, if their count is even then it means that the final string will infact be a 
palindrome.
- The characters that have a odd count say 2n+1 then the 2n times we can just follow the even approach and put the remaining one in the the centre.
- If there are characters with only one count then any of them can be used to form the centre of the string but only one.
{% raw %}
```cpp
class Solution {
public:
    int longestPalindrome(string s) {
        unordered_map<char,int> map;
        int odd = 0;
        for(auto& ch:s) {
            map[ch]++;
            if(map[ch]%2 == 1) 
                odd++;
            else
                odd--;
        }
        if(odd>1)
            return s.size()-odd+1;
        return s.size();
        
    }
};
```
{% endraw %}


## Reverse Linked List
- Just like all the other recursion problem, we can use it here as well
{% raw %}
```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next) 
            return head;
        ListNode* newHead = reverseList(head->next);
        head->next->next = head;
        head->next = NULL;
        return newHead;
    }
};
```
{% endraw %}

- Imagine getting to each head, then wondering if there is anymore left, if there is you go there.
- Finally when you are at the end and only have one node in front of you, reverese the two nodes, by placing your current head after the next node and then severing
the connection to next node.
- Finally you return the new head.


### Majority Element

{% raw %}
```cpp
class Solution {
public:
    int majorityElement(const vector<int>& nums) {
        int element = 0;
        int count = 0;
        for(auto& i:nums) {
            if(count < 0)
                element = i;
            else if (element == i)
                count++;
            else
                count --;
        }
        return element;
    }
};
```
{% endraw %}

- Another perhaps more straight forward approach would be 
{% raw %}
```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        std::sort(nums.begin(),nums.end());
        return nums[nums.size()/2];
    }
};
```
- Also you cannot pass  const iterators (.cbegin(), .cend()) into std::sort.
- Another way of dealing with this problem is by using a hashMap as following.

{% raw %}
```cpp
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        unordered_map<int,int> m;
        for(auto i:nums) {
            m[i]++;
        }
        for(auto z:m) {
            if (z.second > nums.size()/2)
                return z.first;
        }
        return -1;
    }
};
```
{% endraw %}


## Add Binary
- An interesting problem
{% raw %}
```cpp
class Solution {
public:
    string addBinary(string a, string b) {
        int i = a.size()-1;
        int j = b.size()-1;
        int carry = 0;
        std::string ans = "";
        while(i>=0 || j>=0 || carry >0) {
            int sum = carry;
            if(i>=0) 
                sum += a[i--] - '0';
            if(j>=0)
                sum += b[j--] - '0';
            ans = std::to_string(sum%2) + ans;
            carry = sum/2;
        }
        return ans;
    }
};
```
{% endraw %}

- In this solution we are "subtracting" "0", that doesn't make sense but in ASCII all characters are basically represented using
numbers.
- The char "1" and "0" are numerically 1 integer away from each other. So yeah technically you can do something like 
int("b" - "a")
- Try out in python as (ord('b') - ord('a')), see what you get.

### Diameter of a binary tree.
- We can use simple dfs to solve this problem
{% raw %}
```cpp
class Solution {
public:
    int getHeight(TreeNode* root, int& dia) {
        if(!root)
            return 0;
        int lheight = getHeight(root->left,dia);
        int rheight = getHeight(root->right,dia);
        dia = std::max(dia, lheight+rheight);
        return 1 + std::max(getHeight(root->left,dia), getHeight(root->right,dia));

    }

    int diameterOfBinaryTree(TreeNode* root) {
        if(!root)
            return 0;
        int dia = 0;
        getHeight(root,dia);
        return dia;

    }
};
```
{% endraw %}
### Middle of the linked list

- This question builds upon the solutions discussed, as you can see we are traversing the LL using two pointers.
- As we hit the end, as the faster node is traveling at twice the speed of the slower one, the slower one must be 
at the middle.
{% raw %}
```cpp
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        if(!head)
            return head;
        ListNode* slow = head;
        while(head && head->next) {
            slow = slow->next;
            head = head->next->next;
        }
        return slow;
    }
};
```
{% endraw %}

- Don't try to memorize the solutin, but think of trivial case and how the algorithm would perform, if there were only three 
nodes, in the first iteration itself the fast node would be at the end.
- While the slow node would be at the middle, hence we would just return it.
- For this Time Complexity O($n/2$), and Space Complexity O(1)

### Contains Duplicate
- Again this problem leads to multiple solution, one being the following which uses a hash map.
{% raw %}
```cpp
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        unordered_map<int,int> m;
        for(auto& i:nums) {
            m[i]++;
            if(m[i] > 1)
                return true;
        }
        return false;
    }
};
```
{% endraw %}
- The following uses sorting
{% raw %}
```cpp
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        std::sort(nums.begin(),nums.end());
        auto itr = nums.begin();
        while(itr+1 != nums.end()) {
            if(*itr == *(++itr))
                return true; 

        }
        return false;
    }
};
```
{% endraw %}
Also all the following are equivalent
{% raw %}
```cpp
*++itr = * ++itr = *(++itr)
```
{% endraw %}
- But the following won't work
{% raw %}
```cpp
class Solution {
public:
    bool containsDuplicate(vector<int>& nums) {
        std::sort(nums.begin(),nums.end());
        auto itr = nums.begin();
        while(itr+1 != nums.end()) {
            if(*itr == *(itr++))
                return true; 

        }
        return false;
    }
};
```
{% endraw %}

### Maximum Depth of a binary tree.
- Again seeing that we have a binary tree, the problem leads itself well to a very simple recursive approach.
{% raw %}
```cpp
class Solution {
public:
    int getHeight(TreeNode* root) {
        if(!root)
            return 0;
        return 1 + std::max(getHeight(root->left), getHeight(root->right));
    }

    int maxDepth(TreeNode* root) {
        if(!root)
            return 0;
        return getHeight(root);
    }
};
```
{% endraw %}

- Let us also see how a bfs approach would have worked.

{% raw %}
```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root)
            return 0;
        std::queue<TreeNode*> q;
        uint height = 0; 
        q.push(root);
        while(!q.empty()) {
            int lvlwidht = q.size();
            for(int i =0; i<lvlwidht; i++) {
                TreeNode* tn = q.front();
                q.pop();
                if(tn->left) 
                    q.push(tn->left);
                if(tn->right)
                    q.push(tn->right);
            }

            height++;
        }
        return height;
    }
};
```
{% endraw %}
- The interesting thing to notice about this implementation is how we have seperated the task of equeing from the main loop by in
troducing another loop.
- One more thing, since now in this way we are also aware of the width of the tree, we can also find the max width of the tree.


### Roman to Integer
- The problem introduces an insight into how the data is behaving and we use that to our advantage.
{% raw %}
```cpp
class Solution {
public:
    int romanToInt(string s) {
        int res = 0;
        stack<char> stack;
        unordered_map<char,int> rti = {{'I',1},{'V',5},{'X',10},{'L',50},{'C',100},{'D',500},{'M',1000}};

        for(auto& ch:s) {
            if(!stack.empty() &&rti[stack.top()] < rti[ch] ) {
                res += (rti[ch] - rti[stack.top()]);
                stack.pop();
                continue;
            }
            stack.push(ch);
        } 

        while(!stack.empty()) {
            res += rti[stack.top()];
            stack.pop();
        }
        return res;
    }
};
```
{% endraw %}

- I mean if you want something faster, follwing is something to look at, although not in a pleasant way tbh...

{% raw %} 
```cpp
class Solution {
public:
    int romanToInt(string s) {
        // I can be placed before V (5) and X (10) to make 4 and 9. 
        // X can be placed before L (50) and C (100) to make 40 and 90. 
        // C can be placed before D (500) and M (1000) to make 400 and 900.
        int sum = 0;
        for (int i = 0; i < s.length(); i++){
            switch(s[i]){
                case 'I':
                    if (s[i+1] == 'V'){
                        sum += 4;
                        i++;
                    }
                    else if (s[i+1] == 'X'){
                        sum += 9;
                        i++;
                    }
                    else{
                        sum += 1;
                    }
                break;
                case 'V':
                    sum += 5;
                    break;
                case 'X':
                    if (s[i+1] == 'L'){
                        sum += 40;
                        i++;
                    }
                    else if (s[i+1] == 'C'){
                        sum += 90;
                        i++;
                    }
                    else{
                        sum += 10;
                    }
                    break;
                case 'L':
                    sum += 50;
                    break;
                case 'C':
                    if (s[i+1] == 'D'){
                        sum += 400;
                        i++;
                    }
                    else if (s[i+1] == 'M'){
                        sum += 900;
                        i++;
                    }
                    else{
                        sum += 100;
                    }
                    break;
                case 'D':
                    sum += 500;
                    break;
                case 'M':
                    sum += 1000;
                    break;
            
            }
        }
        return sum;
    }
};
```
{% endraw %}


### BackSpace String Compare
- We can solve this in a more natural way if use the stack.
{% raw %}
```cpp
class Solution {
public:
    bool backspaceCompare(const string& s, const string& t) {
        std::stack<char> s_stack;
        std::stack<char> t_stack;
        for(auto& ch:s) {
            if(ch == '#' && !s_stack.empty())
                s_stack.pop();
            else if (ch != '#') 
                s_stack.push(ch);
        }

        for(auto& ch:t) {
            if(ch == '#' && !t_stack.empty())
                t_stack.pop();
            else if (ch != '#') 
                t_stack.push(ch);
        }

        return s_stack == t_stack;
        
    }
};
```
{% endraw %}
- Notice how the stacks mimics what we acutally do when typing.
- The other approach would be to use two variables as follows.
{% raw %}
```cpp
class Solution {
public:
    bool backspaceCompare(std::string S, std::string T) {
    int i = S.length() - 1;
    int j = T.length() - 1;
    int skipS = 0, skipT = 0;

    while (i >= 0 || j >= 0) {
        while (i >= 0) {
            if (S[i] == '#') {
                skipS++;
                i--;
            } else if (skipS > 0) {
                skipS--;
                i--;
            } else {
                break;
            }
        }

        while (j >= 0) {
            if (T[j] == '#') {
                skipT++;
                j--;
            } else if (skipT > 0) {
                skipT--;
                j--;
            } else {
                break;
            }
        }

        // Compare the current characters if both are non-backspace characters
        if (i >= 0 && j >= 0 && S[i] != T[j]) {
            return false;
        }

        // If one string reaches the end but the other doesn't, they are not equal
        if ((i >= 0) != (j >= 0)) {
            return false;
        }

        i--;
        j--;
    }

    return true;
}
};
```
{% endraw %}

## Counting bits 
- [[Bit-Manipulation]]
{% raw %}
```cpp
class Solution {
public:
    uint8_t hammingWeight(uint32_t n) {
        if(!n)
            return 0;
        uint8_t count = 0;
        while(n) {
            count += n & 1;
            n >>= 1;
        }
        return count;
    }


    vector<int> countBits(int n) {
        vector<int> res(n+1,0);
        for(int i = 0; i<=n; ++i) {
            res[i] = hammingWeight(i);
        }
        return res;
    }
};
```
{% endraw %}
- Another straight forward approach would be
{% raw %}
```cpp
class Solution {
public:

    
    vector<int> countBits(int n) {
        vector<int> res(n+1,0);
        for(int i = 0; i<=n; ++i) {
            res[i] = __builtin_popcount(i);
        }
        return res;
    }
};
```
{% endraw %}

## Same Tree
- [[binaryTree]] [[recursion]] [[dfs]] [[bfs]]
{% raw %}
```cpp
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if(!p && !q)
            return true;
        if(!q || !p)
            return false;
        
        if(q->val != p->val)
            return false;

        return isSameTree(p->left,q->left) && isSameTree(p->right,q->right);
    }
```
{% endraw %}
- The above is a dfs approach.
- And the following is a bfs approach.

## Number of 1 bits
- [[Bit-Manipulation]] 
{% raw %}
```cpp
class Solution {
public:
    uint8_t hammingWeight(uint32_t n) {
        if(!n)
            return 0;
        uint8_t count = 0;
        while(n) {
            count += n & 1;
            n >>= 1;
        }
        return count;
    }
};
```
{% endraw %}

## Longest Common Prefix
- [[string]] [[trie]]
- One can solve this problem rather easily using the following solution.
{% raw %}
```cpp
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        if(strs.empty())
            return "";
        if(strs.size() <= 1)
            return strs[0];
        int i = 0;
        bool flag = true;
        while(flag) {
            char cmp = strs[0][i];
            for(auto& str:strs) {
                if(str[i] == cmp)
                    continue;
                else {
                    flag = false;
                    break;
                }
            }
            i++;
        }
        if(!i)
            return "";
        return strs[0].substr(0,i-1);
    }
};
```
{% endraw %}
- But we have a special data structure when it comes to dealing with strings. The solution to that can be found in [[trie]].


## Single Number 
- [[Bit-Manipulation]] [[array]]
{% raw %}
```cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        if(nums.empty()) 
            return -1;
        int ans = 0;
        for(auto& i:nums)
            ans ^= i;
        return ans;

    }
};
```
{% endraw %}


## Palindrome Linked List
- [[stack]] [[recursion]] [[twoPointers]] [[linkedList]] 
{% raw %}
```cpp
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        std::stack<int> intstack;
        ListNode* temp = head;
        while(temp!=NULL) {
            intstack.push(temp->val);
                temp = temp->next;
        }
        
        while(head != NULL) {
            if(head->val != intstack.top())
                return false;
            head = head->next;
            intstack.pop();
        }

        return true;
            
    }
};
```
{% endraw %}
- We can also use a recursive approach as follows
{% raw %}
```cpp
class Solution {
public:
    bool isPalindrome(ListNode*& left, ListNode* right) {
    if (!right) {
        return true;
    }
    bool isPal = isPalindrome(left, right->next);
    isPal = isPal && (left->val == right->val);
    left = left->next;
    return isPal;
}

    bool isPalindrome(ListNode* head) {
        return isPalindrome(head,head);
        
    }
};
```
{% endraw %}

## Convert Sorted Array to Binary Search Tree
- [[binaryTree]] [[binarySearchTree]] [[tree]] [[recursion]] [[divideAndConquer]]
{% raw %}
```cpp
class Solution {
public:
    TreeNode* makeTree(std::vector<int>& nums,int l,int r) {
        if(l>r) 
            return NULL;
        int mid = l + (r-l)/2;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = makeTree(nums,l,mid-1);
        root->right = makeTree(nums,mid+1,r);
        return root;
    }
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        if(nums.empty())
            return NULL;
        return makeTree(nums,0,nums.size()-1);
    }
};
``` 
{% endraw %}

## Reverse Bits
- [[Bit-Manipulation]] [[divideAndConquer]]
{% raw %}
```cpp
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        int res = 0;
        int bits = 31;
        while(bits >= 0) {
            int currBit = n & 1;
            res = res | (currBit << bits);
            n >>= 1;
            bits--;
        }
        return res;
    }
};
```
{% endraw %}

## SubTree of another Tree
- [[binaryTree]] [[recursion]]
{% raw %}
```cpp
bool isIdentical(TreeNode* s, TreeNode* t) {
    if (!s && !t) {
        return true; 
    }
    if (!s || !t) {
        return false; 
    }

    return (s->val == t->val) && isIdentical(s->left, t->left) && isIdentical(s->right, t->right);
}

bool isSubtree(TreeNode* s, TreeNode* t) {
    if (!s) {
        return false; 
    }
    if (isIdentical(s, t)) {
        return true;
    }
    return isSubtree(s->left, t) || isSubtree(s->right, t);
}
```
{% endraw %}
- The bfs approach would be like this
{% raw %}

```cpp
bool isIdentical(TreeNode* s, TreeNode* t) {
    if (!s && !t) {
        return true; 
    }
    if (!s || !t) {
        return false;
    }

    return (s->val == t->val) && isIdentical(s->left, t->left) && isIdentical(s->right, t->right);
}

bool isSubtree(TreeNode* s, TreeNode* t) {
    if (!s) {
        return false; 
    }

    std::queue<TreeNode*> bfsQueue;
    bfsQueue.push(s);

    while (!bfsQueue.empty()) {
        TreeNode* currentNode = bfsQueue.front();
        bfsQueue.pop();
        if (isIdentical(currentNode, t)) {
            return true;
        }
        if (currentNode->left) {
            bfsQueue.push(currentNode->left);
        }
        if (currentNode->right) {
            bfsQueue.push(currentNode->right);
        }
    }

    return false; 
}
```
{% endraw %}


## Squares of a Sorted Array
{% raw %}
```cpp
bool isIdentical(TreeNode* s, TreeNode* t) {
    if (!s && !t) {
        return true; 
    }
    if (!s || !t) {
        return false;
    }

    return (s->val == t->val) && isIdentical(s->left, t->left) && isIdentical(s->right, t->right);
}

bool isSubtree(TreeNode* s, TreeNode* t) {
    if (!s) {
        return false; 
    }

    std::queue<TreeNode*> bfsQueue;
    bfsQueue.push(s);

    while (!bfsQueue.empty()) {
        TreeNode* currentNode = bfsQueue.front();
        bfsQueue.pop();
        if (isIdentical(currentNode, t)) {
            return true;
        }
        if (currentNode->left) {
            bfsQueue.push(currentNode->left);
        }
        if (currentNode->right) {
            bfsQueue.push(currentNode->right);
        }
    }
return false; }
```
{% endraw %}

## Maximum SubArray
- This problem would be more interesting if we were asked to find the array that results in the max sum as well.
- But since we only need the sum, we can just store the max at right place and upadate it as needed and then return it.

{% raw %}
```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if(nums.empty())
            return 0;
        int max = INT_MIN;
        int currSum = 0;
        for(int i = 0; i<nums.size(); i++) {
            currSum += nums[i];
            if(currSum > max) 
                max = currSum;
            if(currSum < 0)
                currSum = 0;
        }    
        return max;    
    }
};
```
- Following is an updated version with the same trick as before for speed.
{% endraw %}

{% raw %}
#pragma GCC optimize("Ofast")
static auto _ = [] () {ios_base::sync_with_stdio(false);cin.tie(nullptr);cout.tie(nullptr);return 0;}();

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        if(nums.empty()) return 0;
        int maxSum = INT_MIN, currSum = 0;
        for(auto num : nums) {
            currSum = std::max(num, currSum + num);
            maxSum = std::max(maxSum,currSum);
        }
        return maxSum; 
    }
};
{% endraw %}

## Insert Interval
- This is also a rather simple problem, we start of by taking all the intervals that will end before the newInterval begins.
- Afterwards we will check if the new interval is overlapping with existing one and update it's value accordingly
- Afterwards we just push all the remaining intervals back and return.

{% raw %}
```cpp
class Solution {
public:
    vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
        int i = 0;
        int n = intervals.size();
        std::vector<std::vector<int>> newIntervals;

        while(i < n && intervals[i][1] < newInterval[0]) {
            newIntervals.push_back(intervals[i++]);
        }

        while(i < n && intervals[i][0] <= newInterval[1]) {
            newInterval[0] = std::min(newInterval[0],intervals[i][0]);
            newInterval[1] = std::max(newInterval[1], intervals[i][1]);
            i++;
        }
        newIntervals.push_back(newInterval);

        while(i < n) {
            newIntervals.push_back(intervals[i++]);
        }
        return newIntervals;
    }
};
```
{% endraw %}

## 01 Matrix
- This problem can be by first figuring out where all the zeroes are.
- Once we have them we do a search around them in a matrix where all elements have been initialised with INT_MAX, this allows
us to compare the value.
- Using this we can figure out the distances of all near by non-zero points and store the value of the distance at those points.

{% raw %}
```cpp
class Solution {
public:
std::vector<std::vector<int>> updateMatrix(std::vector<std::vector<int>>& matrix) {
        if (matrix.empty()) {
            return matrix; 
        }

        int rows = matrix.size();
        int cols = matrix[0].size();

        std::vector<std::vector<int>> result(rows, std::vector<int>(cols, INT_MAX));

        std::queue<std::pair<int, int>> bfsQueue;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 0) {
                    result[i][j] = 0;
                    bfsQueue.push(std::make_pair(i, j));
                }
            }
        }
        std::vector<int> dr = {-1, 1, 0, 0};
        std::vector<int> dc = {0, 0, -1, 1};
        while (!bfsQueue.empty()) {
            int r = bfsQueue.front().first;
            int c = bfsQueue.front().second;
            bfsQueue.pop();
            for (int dir = 0; dir < 4; dir++) {
                int newR = r + dr[dir];
                int newC = c + dc[dir];
                if (newR >= 0 && newR < rows && newC >= 0 && newC < cols &&
                    result[newR][newC] > result[r][c] + 1) {
                    result[newR][newC] = result[r][c] + 1;
                    bfsQueue.push(std::make_pair(newR, newC));
                }
            }
        }

        return result;
    }
};
```
{% endraw %}
- The **dfs** approach will be this 
{% raw %}
```cpp
class Solution {
public:
std::vector<std::vector<int>> updateMatrix(std::vector<std::vector<int>>& matrix) {
        if (matrix.empty()) {
            return matrix;
        }

        int rows = matrix.size();
        int cols = matrix[0].size();
        std::vector<std::vector<int>> result(rows, std::vector<int>(cols, INT_MAX));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (matrix[i][j] == 0) {
                    dfs(matrix, result, i, j, 0);
                }
            }
        }

        return result;
    }

private:
    void dfs(std::vector<std::vector<int>>& matrix, std::vector<std::vector<int>>& result, int r, int c, int distance) {
        int rows = matrix.size();
        int cols = matrix[0].size();

        if (r < 0 || r >= rows || c < 0 || c >= cols || distance >= result[r][c]) {
            return;
        }
        result[r][c] = distance;

        int dr[] = {-1, 1, 0, 0};
        int dc[] = {0, 0, -1, 1};

        for (int dir = 0; dir < 4; dir++) {
            int newR = r + dr[dir];
            int newC = c + dc[dir];

            dfs(matrix, result, newR, newC, distance + 1);
        }
    }
};
```
{% endraw %}
- But the above approach will fail due high time complexity
- Interestingly both algorithms have same time complexities theoretically but **bfs** will actually be faster.
- Also in general recursive alogrithms can fail if they reach recursive depth, there's only so much you cpu can keep track of.

## K Closest Points to origin
- A straight forward approach would be 
{% raw %}
```cpp
class Solution {
public:
    float _getdist(float x,float y) {
        return std::sqrt(pow(x,2)+pow(y,2));
    }
    vector<vector<int>> kClosest(vector<vector<int>>& points, int k) {
        std::map<float,std::vector<vector<int>>> pointMap;
        vector<vector<int>> res;
        for(auto& point:points) {
            float dist = _getdist(point[0],point[1]);
            if(pointMap.find(dist) != pointMap.end()) {
                pointMap[dist].push_back(point);
                continue;
            }
            pointMap[dist].push_back(point);
        }
        auto it = pointMap.begin();
        int i = 0;
        while(i<k) {
            auto vec = it->second;
            for(auto v:vec) {
                res.push_back(v); i++; 
                if(!(i<k))
                    break;
            }
            it++;
        }
        return res;
    }
};
```
{% endraw %}

- This is simply calculating the distances that are around and returning the k closest points.
- A cooler solution would be
{% raw %}
```cpp
struct CompareDistance {
    bool operator()(const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
        return (p1.first * p1.first + p1.second * p1.second) >
               (p2.first * p2.first + p2.second * p2.second);
    }
};

class Solution {
public:
    std::vector<std::vector<int>> kClosest(std::vector<std::vector<int>>& points, int k) {
        // Create a priority queue to store points sorted by distance from origin
        std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, CompareDistance> pq;

        // Push the first K points into the priority queue
        for (int i = 0; i < k; i++) {
            pq.push({points[i][0], points[i][1]});
        }

        // Continue iterating through the remaining points and update the queue if closer points are found
        for (int i = k; i < points.size(); i++) {
            int dist = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            int maxDist = pq.top().first * pq.top().first + pq.top().second * pq.top().second;
            
            if (dist < maxDist) {
                pq.pop();
                pq.push({points[i][0], points[i][1]});
            }
        }

        // Extract the K closest points from the priority queue
        std::vector<std::vector<int>> result;
        while (!pq.empty()) {
            result.push_back({pq.top().first, pq.top().second});
            pq.pop();
        }

        return result;
    }
};
```
{% endraw %}

## Longest Substring Without Repeating Characters
- [[slidingWindow]] [[string]] [[hashMap]]
{% raw %}
```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        std::set<char> memory;
        int rightSide=0, leftSide=0;
        int maxLen = 0;
        while(rightSide < s.size()) {
            if(memory.find(s[rightSide]) == memory.end()) {
                    memory.insert(s[rightSide++]);
            }
            else {
                memory.erase(s[leftSide++]);
            }
            maxLen = std::max(maxLen,rightSide - leftSide);
            
        }
        return maxLen;
    }
};
```
{% endraw %}

## 3Sum
- [[array]] [[twoPointers]] [[sorting]]
{% raw %}
```cpp
class Solution {
public:
    std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
    std::vector<std::vector<int>> result;
    int n = nums.size();

    if (n < 3) {
        return result; 
    }

    // Sort the input array.
    std::sort(nums.begin(), nums.end());

    for (int i = 0; i < n - 2; i++) {
        // Avoid duplicates.
        if (i > 0 && nums[i] == nums[i - 1]) {
            continue;
        }

        int target = -nums[i];
        int left = i + 1;
        int right = n - 1;

        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum == target) {
                result.push_back({nums[i], nums[left], nums[right]});
                // Avoid duplicates.
                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;
                left++;
                right--;
            } else if (sum < target) {
                left++;
            } else {
                right--;
            }
        }
    }
    return result;
}
};
```
{% endraw %}

## Binary Tree Level Order Traversal
- [[binaryTree]] [[bfs]]
{% raw %}
```cpp
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(!root)
            return {};

        std::vector<std::vector<int>> res;
        std::queue<TreeNode*> nq;
        nq.push(root);
        while(!nq.empty()) {
            int N = nq.size();
            std::vector<int> temp;
            for(int i = 0; i<N;i++) {
                TreeNode* curr = nq.front(); nq.pop();
                temp.push_back(curr->val);
                if(root->right)
                    nq.push(root->right);
                if(root->left)
                    nq.push(root->left);
            } res.push_back(temp);
        }
        return res;
    }
};
```
{% endraw %}

## Clone Graph
- [[hashMap]] [[bfs]] [[dfs]] [[graph]]
{% raw %}
```cpp
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> neighbors;
    Node() {
        val = 0;
        neighbors = vector<Node*>();
    }
    Node(int _val) {
        val = _val;
        neighbors = vector<Node*>();
    }
    Node(int _val, vector<Node*> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};
*/

class Solution {
public:
    std::unordered_map<Node*, Node*> visited;

Node* cloneGraph(Node* node) {
    if (!node) {
        return nullptr; // Return nullptr for an empty graph.
    }

    // If the node has already been cloned, return its clone.
    if (visited.find(node) != visited.end()) {
        return visited[node];
    }

    // Create a clone of the current node.
    Node* cloneNode = new Node(node->val);

    // Mark the node as visited and store its clone.
    visited[node] = cloneNode;

    // Recursively clone all neighbors of the current node.
    for (Node* neighbor : node->neighbors) {
        cloneNode->neighbors.push_back(cloneGraph(neighbor));
    }

    return cloneNode;
}
};
```
{% endraw %}

## Min Stack
- Too easy
{% raw %}
```cpp
class MinStack {
public:
    typedef struct node{
        int v;
        int minUntilNow;
        node* next;
    }node;

    MinStack() : topN(nullptr){
        
    }
    
    void push(int val) {
        node* n = new node;
        n->v = n->minUntilNow = val;
        n->next = nullptr;
        
        if(topN == nullptr){
            topN = n;
        }

        else{
            n->minUntilNow = min(n->v,topN->minUntilNow);
            n->next = topN;
            topN = n;
        }
    }
    
    void pop() {
        topN = topN->next;
    }
    
    int top() {
        return topN->v;
    }
    
    int getMin() {
        return topN->minUntilNow;
    }

    private:
    node* topN;
};
```
{% endraw %}

## Evaluate Reverse Polish Notation
- [[math]] [[stack]] [[arra]]
{% raw %}
```cpp
class Solution {
public:
    bool isNum(const std::string& s) {
    std::istringstream ss(s);
    double num;
    return (ss >> num) && ss.eof();
    }

    int calc(std::stack<int>& s, std::string op) {
        int a = s.top(); s.pop();
        int b = s.top(); s.pop();
        if(op == "*")
            return b*a;
        if(op == "/")
            return b/a;
        if(op == "-")
            return b-a;
        if(op == "+")
            return b+a;
        return 0;
    }


    int evalRPN(vector<string>& tokens) {
        std::stack<int> stk;

        for(auto& s:tokens) {
            if(isNum(s)) {
                stk.push(atoi(s.c_str()));
            }
            else {
                stk.push(calc(stk, s));
            }

        }
        return stk.top();
        
    }
};
```
{% endraw %}

## Course Schedule
- [[topological sorting]] [[Khan's Algorithm]]
{% raw %}
```cpp
class Solution {
public:
    bool canFinish(int numCourses, std::vector<std::vector<int>>& prerequisites) {
    std::vector<std::vector<int>> graph(numCourses);
    std::vector<int> inDegree(numCourses, 0);
    for (const auto& prerequisite : prerequisites) {
        int course = prerequisite[0];
        int prereq = prerequisite[1];
        graph[prereq].push_back(course);
        inDegree[course]++;
    }
    std::queue<int> q;
    for (int i = 0; i < numCourses; i++) {
        if (inDegree[i] == 0) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        int course = q.front();
        q.pop();
        numCourses--;
        for (int dependentCourse : graph[course]) {
            if (--inDegree[dependentCourse] == 0) {
                q.push(dependentCourse);
            }
        }
    }
    return numCourses == 0;
}
};
```
{% endraw %}

## Implement Trie
- Just simply implement the datastructure
{% raw %}
```cpp
class TrieNode {
public:
    std::unordered_map<char, TrieNode*> children;
    bool isEndOfWord;

    TrieNode() {
        isEndOfWord = false;
    }
};


class Trie {
private:
    TrieNode* root;
public:
    Trie() {
        root = new TrieNode(); 
    }
    
    void insert(string word) {
        TrieNode* node = root;
        for(auto c : word) {
            if(!node->children[c]) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->isEndOfWord = true;
    }
    
    bool search(string word) {
        TrieNode* node = root;
        for(auto c: word) {
            if(!node->children[c])
                return false;
            node = node->children[c];
        }
        return node->isEndOfWord;
    }
    
    bool startsWith(string prefix) {
        TrieNode* node = root;
        for(auto c: prefix) {
            if(!node->children[c])
                return false;
            node = node->children[c];
        }
        return true;

    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */
```
{% endraw %}


## Coin Change 
- There are multiple ways to solve this problem, following is a **dfs** approach, but it will fail due to over recursion.
{% raw %}
```cpp
class Solution {
public:
    void dfs(std::vector<int>& coins, int amount, int currentAmount, int& minCoins, int& currentCoins) {
    if (currentAmount == 0) {
        minCoins = std::min(minCoins, currentCoins);
        return;
    }

    if (currentAmount < 0 || currentCoins >= minCoins) {
        return;
    }

    for (int coin : coins) {
        currentCoins++;
        dfs(coins, amount, currentAmount - coin, minCoins, currentCoins);
        currentCoins--;
    }
}

int coinChange(std::vector<int>& coins, int amount) {
    int minCoins = amount + 1; 
    int currentCoins = 0;

    dfs(coins, amount, amount, minCoins, currentCoins);

    return (minCoins > amount) ? -1 : minCoins;
}

};
```
{% endraw %}
- Following is a **bfs** approach
{% raw %}
```cpp
class Solution {
public:
    int coinChange(std::vector<int>& coins, int amount) {
    std::queue<int> q;
    std::vector<bool> visited(amount + 1, false);

    q.push(amount);
    visited[amount] = true;
    int level = 0; 
    while (!q.empty()) {
        int size = q.size();

        for (int i = 0; i < size; i++) {
            int currentAmount = q.front();
            q.pop();

            if (currentAmount == 0) {
                return level; 
            }

            for (int coin : coins) {
                int nextAmount = currentAmount - coin;

       
                if (nextAmount >= 0 && !visited[nextAmount]) {
                    q.push(nextAmount);
                    visited[nextAmount] = true;
                }
            }
        }

        level++; 
    }

    return -1; 
}
};
```
{% endraw %}
- And below is the iterative approach
{% raw %}
```cpp
int coinChange(std::vector<int>& coins, int amount) {

    std::vector<int> dp(amount + 1, amount + 1);

    dp[0] = 0;


    for (int i = 1; i <= amount; i++) {
        for (int coin : coins) {
            if (coin <= i) {
                dp[i] = std::min(dp[i], dp[i - coin] + 1);
            }
        }
    }


    return (dp[amount] > amount) ? -1 : dp[amount];
}
```
{% endraw %}

## Product of Array except Self


## Validate Binary Search Tree
- [[binaryTree]] [[recursion]]
{% raw %}
```cpp
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if(root && !root->right && !root->left)
            return true;
        else if(!root->left && root->right)
            return isValidBST(root->right);
        else if(!root->right && root->left)
            return isValidBST(root->left);
        else if(root->val >= root->right->val || root->val <= root->left->val)
            return false;
        return isValidBST(root->right) && isValidBST(root->left);
    }
};

```
{% endraw %}

## Number Of Islands
- [[graph]] [[dfs]] [[bfs]] [[unionFind]] [[matrix]]
{% raw %}
```cpp
class Solution {
public:
    int numIslands(std::vector<std::vector<char>>& grid) {
        if (grid.empty() || grid[0].empty()) {
            return 0; // Empty grid has no islands.
        }

        int numIslands = 0;
        int rows = grid.size();
        int cols = grid[0].size();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == '1') {
                    numIslands++;
                    dfs(grid, i, j);
                }
            }
        }

        return numIslands;
    }

private:
    void dfs(std::vector<std::vector<char>>& grid, int row, int col) {
        int rows = grid.size();
        int cols = grid[0].size();

        if (row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] == '0') {
            return; // Out of bounds or water cell, stop recursion.
        }

        // Mark the current cell as visited.
        grid[row][col] = '0';

        // Recursively explore neighboring cells.
        dfs(grid, row - 1, col); // Up
        dfs(grid, row + 1, col); // Down
        dfs(grid, row, col - 1); // Left
        dfs(grid, row, col + 1); // Right
    }
};
```
{% endraw %}
## Rotating Oranges
- [[array]] [[bfs]] [[matrix]]
{% raw %}
```cpp
class Solution {
public:
    int orangesRotting(std::vector<std::vector<int>>& grid) {
        int rows = grid.size();
        int cols = grid[0].size();
        int minutes = 0;

        std::queue<std::pair<int, int>> rottenQueue;

        // Count the number of fresh oranges and add rotten oranges to the queue.
        int freshOranges = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 2) {
                    rottenQueue.push({i, j});
                } else if (grid[i][j] == 1) {
                    freshOranges++;
                }
            }
        }

        // Define possible directions to adjacent cells.
        std::vector<std::pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        while (!rottenQueue.empty() && freshOranges > 0) {
            int queueSize = rottenQueue.size();

            for (int i = 0; i < queueSize; i++) {
                std::pair<int, int> curr = rottenQueue.front();
                rottenQueue.pop();

                for (const auto& direction : directions) {
                    int newRow = curr.first + direction.first;
                    int newCol = curr.second + direction.second;

                    if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols &&
                        grid[newRow][newCol] == 1) {
                        grid[newRow][newCol] = 2; // Mark the adjacent fresh orange as rotten.
                        rottenQueue.push({newRow, newCol});
                        freshOranges--;
                    }
                }
            }

            if (!rottenQueue.empty()) {
                minutes++; // Increment the minute if there are more rotten oranges.
            }
        }

        return (freshOranges == 0) ? minutes : -1; // Return -1 if some oranges remain fresh.
    }
};
```
{% endraw %}


