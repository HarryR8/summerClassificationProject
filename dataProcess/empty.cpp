#include <vector>
class Solution {
public:
    int sumOfMultiples(int n) {
std::vector<int> matches;
        for (int i{1}; i <= n; i++) {
            if (i%5 == 0) {
                matches.push_back(i);
            }
            else if (i%3 == 0) {
                matches.push_back(i);
            }
            else if (i%7 == 0) {
                matches.push_back(i);
            }
        }
        int sum = 0;
        for (int val : matches) {
            sum += val;
        }
        return sum;
    }
};