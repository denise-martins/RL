import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt



'''  
     Recompensas calculadas Linha + Coluna
     0  1   2   3   
     1  2   3   4
     2  3   4   5
     3  4   5   6'''



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
            #print(f'Recompensa:{reward}, Estado:{self.state}')

    

        ############### Recompensa baseada em soma linha + coluna ################
        if premio == False and buraco == False:
            reward = int(self.state[0]+self.state[1])
            #print(f'Recompensa:{reward} , Estado:{self.state}')


        
        return self.state,reward, terminated, truncated, {} ### Retornando o estado se terminou True/False 




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



# =====================================
# Q-TABLE
# =====================================

# exemplo:
# grid 4x4
# cada estado será:
# (linha, coluna)

num_actions = env.action_space.n

# dicionário ao invés de matriz
# porque estados são tuplas agora

q_table = {}

# =====================================
# HIPERPARÂMETROS
# =====================================

learning_rate = 0.2 # taxa de aprendizagem
discount_factor = 0.95

epsilon = 0.99
epsilon_decay = 0.9
epsilon_min = 0.01


num_episodes = 64

# =====================================
# TREINAMENTO
# =====================================

rewards_por_episodio = []
for episodio in range(num_episodes):

    
    state, info = env.reset()

    print('Aqui deve ser a origem ->',state) # Apenas para garantir que o agente inicie o episódio em [0,0]
    
    done = False
    
    total_reward = 0

    while not done:

        # =====================================
        # TRANSFORMA ARRAY EM TUPLA
        # =====================================

        # exemplo:
        # [2, 3] -> (2, 3)

        state_key = tuple(state)

        # =====================================
        # SE ESTADO NÃO EXISTE NA Q-TABLE
        # =====================================

        if state_key not in q_table:

            q_table[state_key] = np.zeros(num_actions)  # q_table[(2,3)] = [0, 0, 0, 0], com o tempo a tabela é preenchida com os valores das recompensas

        # =====================================
        # EPSILON-GREEDY
        # =====================================

        if np.random.random() < epsilon:   ### np.random.random gera números aleatórios decimais  entre 0 e 1 

            action = env.action_space.sample()

        else:

            action = np.argmax(q_table[state_key]) # retorna o índice da melhor ação / esetimativa de recompensa da ação naquele estado

        # =====================================
        # EXECUTA AÇÃO
        # =====================================

        next_state,reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
       
        # =====================================
        # TRANSFORMA NEXT_STATE EM TUPLA
        # =====================================

        next_state_key = tuple(next_state)
        


        # =============================================================================================
        # PUNIÇÕES PARA EVITAR QUE O AGENTE FIQUE PARADO OU QUE VOLTE A ESTADOS MAIS DISTANTE DO PRÊMIO:
        # =============================================================================================
        
        dist_atual = abs(state_key[0]-3) + abs(state_key[1] - 3)
       
        dist_proxima = abs(next_state_key[0] - 3) + abs(next_state_key[1]-3)

        if dist_proxima>= dist_atual:
            reward -= 5



        

        # =====================================
        # SE NEXT_STATE NÃO EXISTE NA TABELA
        # =====================================

        if next_state_key not in q_table:

            q_table[next_state_key] = np.zeros(num_actions)

        # =====================================
        # Q-LEARNING UPDATE
        # =====================================

        old_value = q_table[state_key][action]

        next_max = np.max(q_table[next_state_key])

        new_value = old_value + learning_rate * (
            reward
            + discount_factor * next_max
            - old_value
        )

        q_table[state_key][action] = new_value   # está atualizando o valor para aquele estado com aquela ação

        # =====================================
        # AVANÇA ESTADO
        # =====================================

        
        state = next_state

        total_reward += reward

    

        # render
        env.render()

    rewards_por_episodio.append(total_reward)
    # =====================================
    # DIMINUI EXPLORAÇÃO
    # =====================================

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    
    # progresso
    if episodio >=0:

        print(f'\nEPISÓDIO {episodio}')
        print(f'Recompensa total: {total_reward}')
        print(f'Epsilon: {epsilon:.3f}')




plt.plot(rewards_por_episodio)

plt.xlabel("Episódio")
plt.ylabel("Recompensa Total")

plt.title("Curva de Aprendizagem")

plt.show()


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

