import pygame
import sys
import subprocess
import time

class TrafficControllerHome:
    def __init__(self):
        pygame.init()
        
       
        self.width = 1000
        self.height = 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Smart Traffic Controller")
        
       
        self.background = pygame.image.load('traffic_bg.jpg')
        self.background = pygame.transform.scale(self.background, (self.width, self.height))
        
      
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 122, 255)
        self.DARK_BLUE = (0, 76, 153)
        
        
        self.button_width = 300
        self.button_height = 60
        self.button_spacing = 40
        
        
        self.font = pygame.font.Font(None, 36)
        self.loading_font = pygame.font.Font(None, 48)
        
    def cleanup(self):
        """Properly cleanup pygame resources"""
        pygame.quit()
        sys.exit()
    
    def show_loading_screen(self, message):
        self.screen.fill(self.BLACK)
        loading_text = self.loading_font.render(message, True, self.WHITE)
        text_rect = loading_text.get_rect(center=(self.width/2, self.height/2))
        self.screen.blit(loading_text, text_rect)
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cleanup()
    
    def draw_button(self, x, y, width, height, text, color):
        button_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, color, button_rect, border_radius=10)
        
        text_surface = self.font.render(text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=button_rect.center)
        self.screen.blit(text_surface, text_rect)
        
        return button_rect
    
    def run_random_forest(self):
        try:
            # Show loading screen
            self.show_loading_screen("Training Random Forest Model...")
            pygame.display.flip()
            
            # Run the training script
            subprocess.run(["python", "traffic_random_forest.py"])
            
            # Show completion message
            self.show_loading_screen("Training Complete! Starting Simulation...")
            pygame.display.flip()
            time.sleep(3)
            
            # Run the simulation
            subprocess.run(["python", "traffic_rf_simulation_fixed.py"])
        except:
            self.cleanup()
    
    def run(self):
        try:
            while True:
                # Draw background
                self.screen.blit(self.background, (0, 0))
                
                # Add semi-transparent overlay for better text readability
                overlay = pygame.Surface((self.width, self.height))
                overlay.fill(self.BLACK)
                overlay.set_alpha(128)  # 50% transparency
                self.screen.blit(overlay, (0, 0))
                
                # Draw title
                title = self.font.render("Choose Traffic Determination Method", True, self.WHITE)
                title_rect = title.get_rect(center=(self.width/2, 150))
                self.screen.blit(title, title_rect)
                
                # Calculate button positions
                center_x = self.width/2 - self.button_width/2
                first_button_y = 250
                second_button_y = first_button_y + self.button_height + self.button_spacing
                third_button_y = second_button_y + self.button_height + self.button_spacing
                
                # Draw buttons
                dynamic_button = self.draw_button(
                    center_x, first_button_y,
                    self.button_width, self.button_height,
                    "Dynamic Way", self.BLUE
                )
                
                rf_button = self.draw_button(
                    center_x, second_button_y,
                    self.button_width, self.button_height,
                    "Random Forest Way", self.BLUE
                )
                
                static_button = self.draw_button(
                    center_x, third_button_y,
                    self.button_width, self.button_height,
                    "Static Way", self.BLUE
                )
                
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.cleanup()
                        
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = event.pos
                        
                        if dynamic_button.collidepoint(mouse_pos):
                            pygame.quit()
                            try:
                                subprocess.run(["python", "simulation Dy.py"], check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Error running simulation: {e}")
                            sys.exit()
                            
                        if rf_button.collidepoint(mouse_pos):
                            self.run_random_forest()
                            return

                        if static_button.collidepoint(mouse_pos):
                            pygame.quit()
                            try:
                                subprocess.run(["python", "simulation state.py"], check=True)
                            except subprocess.CalledProcessError as e:
                                print(f"Error running simulation: {e}")
                            sys.exit()
                    
                    if event.type == pygame.MOUSEMOTION:
                        mouse_pos = event.pos
                        
                        # Hover effects
                        if dynamic_button.collidepoint(mouse_pos):
                            dynamic_button = self.draw_button(
                                center_x, first_button_y,
                                self.button_width, self.button_height,
                                "Dynamic Way", self.DARK_BLUE
                            )
                        
                        if rf_button.collidepoint(mouse_pos):
                            rf_button = self.draw_button(
                                center_x, second_button_y,
                                self.button_width, self.button_height,
                                "Random Forest Way", self.DARK_BLUE
                            )
                            
                        if static_button.collidepoint(mouse_pos):
                            static_button = self.draw_button(
                                center_x, third_button_y,
                                self.button_width, self.button_height,
                                "Static Way", self.DARK_BLUE
                            )
        except:
            self.cleanup()

if __name__ == "__main__":
    try:
        app = TrafficControllerHome()
        app.run()
    except KeyboardInterrupt:
        app.cleanup()