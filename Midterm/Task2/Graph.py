class Graph:
    def __init__(self, wall, food, pie, start):
        self.wall = wall                #list chua toa do wall
        self.food = food                #list chua toa do food
        self.pie = pie                  #list chua toa do pie
        self.start = start              #toa do start
        self.rows = max(x for x, y in wall) + 1 #count row bang cach tim gia tri x lon nhat tu list wall roi + them 1 because toa do bat dau tu 0
        self.cols = max(y for x, y in wall) + 1 #count collumn bang cach tim gia tri y lon nhat tu list wall roi + them 1 because toa do bat dau tu 0
        self.corners = {  # dictionary of corners for teleport 
            (1, 1): (self.rows - 2, self.cols - 2),
            (1, self.cols - 2): (self.rows - 2, 1),
            (self.rows - 2, 1): (1, self.cols - 2),
            (self.rows - 2, self.cols - 2): (1, 1)
        }
    # check wall
    def is_wall(self, position):
        x, y = position
        #kiem tra gioi han me cung va kiem tra toa do do co trong list wall hay khong
        if (x < 0 or y < 0 or x >= self.rows or y >= self.cols) or ((x, y) in self.wall): 
            return True
        return False
    # check food
    def is_food(self, position):
        return position in self.food
    # check pie
    def is_pie(self, position):
        return position in self.pie
    # check corner
    def is_corner(self, position):
        return position in self.corners.values()
    # read file
    @staticmethod
    def readfile(filename):
        wall = []
        food = []
        pie = []
        start = None
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            for y, line in enumerate(lines):
                for x, char in enumerate(line.strip()):
                    if char == '%': #neu la tuong thi append toa do vao list
                        wall.append((x, y))
                    elif char == '.': #neu la food thi append toa do vao list
                        food.append((x, y))
                    elif char == 'P': #neu la start thi gan vao
                        start = (x, y)
                    elif char == 'O': #neu la pie thi append toa do vao list
                        pie.append((x, y))
        return Graph(wall, food, pie, start)  # Return a Graph object instead of a tuple
    