import pygame
from Graph import Graph
from Search import Search
import time

class Visualize:
    def __init__(self, graph, path, cell_size=30):
        self.graph = graph
        self.path = path
        self.search = Search(graph, graph.start) 
        self.cell_size = cell_size
        self.WIDTH = graph.rows * cell_size
        self.HEIGHT = graph.cols * cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.SysFont("Arial", 24)
        pygame.display.set_caption("Pacman Pathfinding")

    def draw(self, pacman_pos, visited_positions=[]):
        self.screen.fill((0, 0, 0))
        for x, y in self.graph.wall:
            pygame.draw.rect(self.screen, (0, 0, 255), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.graph.food:
            pygame.draw.rect(self.screen, (255, 255, 255), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        for x, y in self.graph.pie:
            pygame.draw.rect(self.screen, (255, 255, 0), (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size))
        for i in range(len(visited_positions) - 1):
            pygame.draw.line(self.screen, (0, 255, 0), 
                             ((visited_positions[i][0] + 0.5) * self.cell_size, (visited_positions[i][1] + 0.5) * self.cell_size),
                             ((visited_positions[i + 1][0] + 0.5) * self.cell_size, (visited_positions[i + 1][1] + 0.5) * self.cell_size), 3)
        pygame.draw.circle(self.screen, (255, 0, 0), ((pacman_pos[0] + 0.5) * self.cell_size, (pacman_pos[1] + 0.5) * self.cell_size), self.cell_size // 3)
        pygame.display.flip()

    def path_Pacman(self):
        # Mo phong buoc di cua Pacman
        current_state = (self.graph.start[0], self.graph.start[1], frozenset(self.graph.food), frozenset(self.graph.pie), 0)
        state_sequence = [(current_state, None)]
        for action in self.path:
            current_state, teleportfood = Search.apply_action(self.search, current_state, action)
            state_sequence.append((current_state, teleportfood))

        
        index = 0
        move_delay = 0.3
        last_move_time = time.time()
        if self:
            print("Actions:", ", ".join(self.path))
            print("Total cost:", len(self.path))
            print("\nHeuristic values at each step:")
            for state, _ in state_sequence:
                h_value = Search.improved_heuristic(state, self.graph)
                print(f"State: {state[:2]}, Heuristic: {h_value}")
        else:
            print("No path found")
        visited_positions = []

        start_screen = True  # Flag de hien thi man hinh chay
        animation_started = False  # Flag de theo doi animation


        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                
                # Kiem tra su kien
                if event.type == pygame.KEYDOWN and start_screen and not animation_started:
                    animation_started = True 
                    start_screen = False  
            
            if start_screen:
                # Hien man hinh
                self.screen.fill((0, 0, 0))  
                font = pygame.font.SysFont("verdana", 32)
                text = font.render("Press any key to Start", True, (255, 255, 255))
                self.screen.blit(text, (self.WIDTH // 2 - text.get_width() // 2, self.HEIGHT // 2 - text.get_height() // 2))
                pygame.display.flip()
                continue  
            
            # Lam Pacman di chuyen theo duong da tim
            if index < len(state_sequence) and (time.time() - last_move_time) >= move_delay:
                current_state, teleportfood = state_sequence[index]
                index += 1
                last_move_time = time.time()
                pacman_pos = (current_state[0], current_state[1])

                # Xoa thuc an hoac teleport neu co
                if teleportfood is not None:
                    self.graph = self.search.remove_eaten(self.graph, teleportfood)
                self.graph = self.search.remove_eaten(self.graph, pacman_pos)

                # Luu lai duong di
                visited_positions.append(pacman_pos)
                self.draw(pacman_pos, visited_positions)

        pygame.quit()