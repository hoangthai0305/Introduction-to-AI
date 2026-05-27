import heapq
import math
from Graph import Graph

class Search:
    def __init__(self, graph, position):
        self.graph = graph
        self.position = position
    
    @staticmethod
    def heuristic(position, graph):
        # tinh kc manhattan
        def manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        x, y, food, pie, step = position
        manhattan_temp = []
        min_manhattan_7_goal = []
        for x_food, y_food in food: 
            manhattan_temp.append(manhattan_distance((x, y), (x_food, y_food)))
            manhattan_temp.append(manhattan_distance((x, y), (x_food + 36 - 3, y_food + 18 - 3)))
            manhattan_temp.append(manhattan_distance((x, y), (x_food - 36 + 3, y_food - 18 + 3)))
            manhattan_temp.append(manhattan_distance((x, y), (x_food + 36 - 3, y_food - 18 + 3)))
            manhattan_temp.append(manhattan_distance((x, y), (x_food - 36 + 3, y_food + 18 - 3)))
            min_manhattan_7_goal.append(min(manhattan_temp))
        if not food:
            return 0
        return min(min_manhattan_7_goal)
    
    @staticmethod
    def improved_heuristic(position, graph):
        def manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
            
        x, y, food, pie, step = position
        if not food:
            return 0

        # ham tinh chi phi giua pacman va goal, so sanh voi teleport dua tren y tuong heuristic cu
        def cost_with_teleport(pacman, goal):
            direct_cost = manhattan_distance(pacman, goal)
            teleport_costs = []
            # duyet qua tat ca cac teleport
            for entrance, destination in graph.corners.items():
                # chi phi di tu pacman den entrance + từ destination den goal
                cost = manhattan_distance(pacman, entrance) + manhattan_distance(destination, goal)
                teleport_costs.append(cost)
            return min(direct_cost, min(teleport_costs))
        
        # khoang cach tu pacman den goal gan nhat
        current_to_food = min(cost_with_teleport((x, y), f) for f in food)
        
        # tinh chi phi cay bao trum toi thieu mst cho cac food con lai
        food_list = list(food)
        if len(food_list) <= 1:
            mst_cost = 0
        else:
            mst_cost = 0
            connected = {food_list[0]}
            remaining = set(food_list[1:])
            while remaining:
                min_edge = float('inf')
                next_food = None
                for a in connected:
                    for b in remaining:
                        d = cost_with_teleport(a, b)
                        if d < min_edge:
                            min_edge = d
                            next_food = b
                mst_cost += min_edge
                connected.add(next_food)
                remaining.remove(next_food)

        return current_to_food + mst_cost



    @staticmethod
    def combined_heuristic(position, graph): # ket hop giua improve heuristic va heuristic ban dau
        def manhattan_distance(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Nang cap chinh sua tu heuristic ban dau lai sao cho no clean code
        def adjusted_cost(p1, p2):
            x2, y2 = p2
            distances = []
            distances.append(manhattan_distance(p1, p2))
            distances.append(manhattan_distance(p1, (x2 + 36 - 3, y2 + 18 - 3)))
            distances.append(manhattan_distance(p1, (x2 - 36 + 3, y2 - 18 + 3)))
            distances.append(manhattan_distance(p1, (x2 + 36 - 3, y2 - 18 + 3)))
            distances.append(manhattan_distance(p1, (x2 - 36 + 3, y2 + 18 - 3)))
            return min(distances)
        
        x, y, food, pie, step = position
        if not food:
            return 0

        current_to_food = min(adjusted_cost((x, y), f) for f in food)
        
        food_list = list(food)
        if len(food_list) <= 1:
            mst_cost = 0
        else:
            mst_cost = 0
            connected = {food_list[0]}
            remaining = set(food_list[1:])
            while remaining:
                min_edge = float('inf')
                next_food = None
                for a in connected:
                    for b in remaining:
                        d = adjusted_cost(a, b)
                        if d < min_edge:
                            min_edge = d
                            next_food = b
                mst_cost += min_edge
                connected.add(next_food)
                remaining.remove(next_food)
        
        return current_to_food + mst_cost



    def find_neighbours(self, position):
        # x, y la toa do cua pacman
        # food la tap hop food chua an
        # pie la tap hop cac banh ma thuat
        # step la so magic step con lai
        x, y, food, pie, step = position 

        # dinh nghia cac huong di chuyen cua pacman
        directions = [(0, 1, 'South'), (1, 0, 'East'), (0, -1, 'North'), (-1, 0, 'West')]
        

        for dx, dy, action in directions:
            # tinh vi tri moi cua pacman
            nr, nc = x + dx, y + dy
            
            # Kiem tra xem pacman con o trong me cung khong
            if nr < 0 or nr >= self.graph.rows or nc < 0 or nc >= self.graph.cols:
                continue

            # giam so buoc magic xuong 1
            new_step = max(0, step - 1) 

            # Neu gap tuong va khong co magic step thi khong di huong do
            if self.graph.is_wall((nr, nc)) and new_step == 0:
                continue
            
            # tao bien danh dau teleport
            teleport_used = False
            teleport_food = None
            # thiet lap vi tri moi 
            new_x, new_y = nr, nc
            # tao ban sao cua tap food
            new_food = food.copy()
            
            # check neu nr, nc la entrance cua goc
            if (nr, nc) in self.graph.corners:
                # danh dau bien teleport
                teleport_used = True
                # ghi nho vi tri entrance cua teleport
                teleport_food = (nr, nc)
                # loai bo food o entrance neu co
                if (nr, nc) in new_food:
                    new_food = new_food - frozenset({(nr, nc)})
                # thuc hien tele den vi tri destination
                new_x, new_y = self.graph.corners[(nr, nc)]
                # neu destination la wall va khong co magic step thi bo qua huong di
                if self.graph.is_wall((new_x, new_y)) and new_step == 0:
                    continue
                # loai bo food o destination neu co
                if (new_x, new_y) in new_food:
                    new_food = new_food - frozenset({(new_x, new_y)})
            else:
                # neu khong phai la corner thi kiem tra va loai bo food o vi tri do
                if (new_x, new_y) in new_food:
                    new_food = new_food - frozenset({(new_x, new_y)})
            
            # cap nhat lai tap hop pie
            new_pie = pie.copy()
            if (new_x, new_y) in new_pie:
                new_pie = new_pie - frozenset({(new_x, new_y)})
                new_step = 5  # dat lai magic step neu nhat duoc pie
            # giam chi phi de uu tien su dung teleport
            extra_cost = 0.5 if teleport_used else 1
            
            # tra ve trang thai moi 
            yield (new_x, new_y, frozenset(new_food), frozenset(new_pie), new_step), action, extra_cost, teleport_food


    def astar_search(self):
        start_state = (self.graph.start[0], self.graph.start[1], frozenset(self.graph.food), frozenset(self.graph.pie), 0)
        frontier = [(self.improved_heuristic(start_state, self.graph), 0, start_state)]

        came_from = {start_state: (None, None)}
        cost_so_far = {start_state: 0}
        
        while frontier:
            f, g, state = heapq.heappop(frontier)
            if cost_so_far[state] > g:
                continue
            
            x, y, food, pie, step = state
            if not food:
                path = []
                while state in came_from and came_from[state][0] is not None:
                    state, action = came_from[state]
                    path.append(action)
                return path[::-1], cost_so_far[state]
            
            for next_state, action, cost, _ in self.find_neighbours(state):
                new_cost = cost_so_far[state] + cost
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self.improved_heuristic(next_state, self.graph)
                    heapq.heappush(frontier, (priority, new_cost, next_state))
                    came_from[next_state] = (state, action)
        
        return None, None

    def apply_action(self, state, action):
        for next_state, act, cost, teleport_food in self.find_neighbours(state):
            if act == action:
                return next_state, teleport_food
        return state, None
    
    @staticmethod #phuong thuc tinh 
    def remove_eaten(graph, position):
        if position in graph.food:
            graph.food.remove(position)
        if position in graph.pie:
            graph.pie.remove(position)
        return graph
