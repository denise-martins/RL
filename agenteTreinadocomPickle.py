import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
import pickle
class MeuAmbiente(gym.Env):

    def __init__(self):

        # 4 ações:
        # 0 = esquerda
        # 1 = baixo
        # 2 = direita
        # 3 = cima
        self.action_space = spaces.Discrete(4)

        # estado = [linha, coluna]
        # linha: 0-3
        # coluna: 0-3
        self.observation_space = spaces.MultiDiscrete([4,4])

        pygame.init()

        self.window_size = 400

        self.window = pygame.display.set_mode(
        (self.window_size, self.window_size))

        pygame.display.set_caption("Meu FrozenLake")

        self.clock = pygame.time.Clock()

        self.cell_size = self.window_size // 4

    def reset(self, seed=None, options=None):

        self.state = np.array([0, 0], dtype=np.int32)
        print('Resetou')
        return self.state, {}



    def step(self, action):

        ''' 1. Altera o estado
            2. Calcula Recompensa
            3. verifica se terminou
            4. retorna tudo'''
        
        reward = 0
        terminated = False
        truncated = False

        # lógica do ambiente

        ################# DistÂncia euclidiana ######################## 

        objetivo = np.array([3,3])
        distancia = np.linalg.norm(self.state - objetivo) # Distância euclidiana

        


        ################## Ações/Movimento do Agente ######################
        
        if action == 0:    ###### move para a Esquerda <--
            self.state[1] -= 1
            

        elif action == 1:  ##### Move para baixo |
            self.state[0] += 1
            

        elif action == 2:   ##### Move para a Direita -->
            self.state[1] += 1
            

        elif action == 3:  ##### Move para cima ^.
            self.state[0] -=1
            

        self.state[0] = np.clip(self.state[0],0,3)
        self.state[1] = np.clip(self.state[1],0,3)


        ############### Vitória! chegou ao prêmio ##############
        
        premio = self.state[0]==3 and self.state[1]==3
        if premio == True:
            print("conseguiu o prêmio")
            reward = 100
            terminated = True
            
             
        
        ############### Buracos ##############################
        
        buraco = self.state[0]==3 and self.state[1]==1 or  self.state[0]==1 and self.state[1]==2

        if buraco == True:
            print("Caiu no buraco")
            reward = -100
            terminated = True

        
        ############### Calculando a Distância Euclidiana  #####################
        objetivo = np.array([3,3])
        distancia = np.linalg.norm(self.state - objetivo) # Distância euclidiana

        ############## Recomepensa ###################################
        if float(distancia) != 0 and buraco == False: #Se não cair em buraco e também não chegar ao prêmio calcule a recompensa
            reward = 1 / float(distancia)
            

        #print('Recompensa:',reward,'Estado:',self.state)
         

        return self.state, reward, terminated, truncated, {}


    def render(self):

        # fecha janela corretamente
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        # cor de fundo
        self.window.fill((173, 216, 230))

        # desenhar grade
        for linha in range(4):
            for coluna in range(4):

                rect = pygame.Rect(
                    coluna * self.cell_size,
                    linha * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                # chão
                pygame.draw.rect(
                    self.window,
                    (240, 248, 255),
                    rect
                )

                # borda
                pygame.draw.rect(
                    self.window,
                    (0, 0, 0),
                    rect,
                    2
                )

        # desenhar buraco

        # posições dos buracos
        buracos = [(3,1),(1,2)]

        # desenhar buracos
        for linha, coluna in buracos:

            buraco_rect = pygame.Rect(
                coluna * self.cell_size,
                linha * self.cell_size,
                self.cell_size,
                self.cell_size)

            pygame.draw.rect(
                self.window,
                (0, 0, 0),
                buraco_rect)
        

        # desenhar objetivo
        objetivo_rect = pygame.Rect(
            3 * self.cell_size,
            3 * self.cell_size,
            self.cell_size,
            self.cell_size
        )

        pygame.draw.rect(
            self.window,
            (255, 215, 0),
            objetivo_rect
        )

        # desenhar agente
        linha = self.state[0]
        coluna = self.state[1]

        center_x = coluna * self.cell_size + self.cell_size // 2
        center_y = linha * self.cell_size + self.cell_size // 2

        pygame.draw.circle(
            self.window,
            (220, 20, 60),
            (center_x, center_y),
            self.cell_size // 4
        )

        pygame.display.update()

        self.clock.tick(5)

    
env = MeuAmbiente()



with open('q_table.pkl','rb') as arquivo_q_table:
    q_table = pickle.load(arquivo_q_table)
print(q_table)


# =====================================
# TESTE FINAL
# =====================================

print("\nTREINAMENTO FINALIZADO\n")

state, info = env.reset()

done = False

while not done:

    state_key = tuple(state)

    # pega melhor ação aprendida
    action = np.argmax(q_table[state_key])

    state, reward, terminated, truncated, info = env.step(action)

    env.render()

    print("Estado:", state)
    print("Ação:", action)
    print("Reward:", reward)

    done = terminated or truncated


""" bibliotecas pandas gera um csv DataFrame salvar episódio-recompensa, biblioteca pickling como usar para salvar uma estrutura de códigos salvar o treinamento do agente 
    imnprimir um gráfico do csv, 
    ver a recompensa --foto no celular-- traçar um gráfico com recompensa euclidiana e um com a recompensa diferente"""

