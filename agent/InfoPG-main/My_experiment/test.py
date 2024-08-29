class Solution:
    def minimumFuelCost(self, roads: list, seats: int) -> int:
        self.createtree(roads)
        self.seats = seats
        self.ans = 0
        n = self.traverse_tree(-1, 0)
        return self.ans



    def traverse_tree(self, father: int, son:int) -> int:
        num = 1
        for node in self.road_dict[son]:
            if node != father:
                num += self.traverse_tree(son, node)
        if son:
            self.ans += (int((num-1)/self.seats) + 1)
        return num


    def createtree(self, roads):
        self.road_dict = {}
        for edge in roads:
            self.road_dict[edge[0]] = []
            self.road_dict[edge[1]] = []
        for edge in roads:
            self.road_dict[edge[0]].append(edge[1])
            self.road_dict[edge[1]].append(edge[0])


a = Solution()
b = a.minimumFuelCost(roads=[[3, 1], [3, 2], [1, 0], [0, 4], [0, 5], [4, 6]], seats=2)
print(b)
