import pygame
import sys
from src.simulation.simulation import Simulation

def main():
    # 初始化pygame
    pygame.init()
    pygame.font.init()
    
    # 创建仿真实例
    simulation = Simulation()
    
    # 主循环
    while simulation.running:
        simulation.run()
    
    # 退出游戏
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()