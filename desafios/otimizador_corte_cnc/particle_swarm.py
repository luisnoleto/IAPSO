from common.layout_display import LayoutDisplayMixin
import numpy as np
import random
import copy
import time
from tqdm import tqdm  # Para barra de progresso
import matplotlib.pyplot as plt

# Verificar disponibilidade de GPU
try:
    import cupy as cp
    has_gpu = True
    print("GPU acceleration enabled (CuPy found)")
except ImportError:
    has_gpu = False
    print("GPU acceleration disabled (CuPy not found). Using NumPy instead.")
    cp = np

class ParticleSwarm(LayoutDisplayMixin):
    def __init__(self, num_particles, num_iterations, dim, sheet_width, sheet_height, recortes_disponiveis, 
                 use_gpu=False, show_progress=True, visualization_interval=10):
        """
        Initializes the Particle Swarm optimizer.
        :param num_particles: Number of particles.
        :param num_iterations: Number of iterations to run.
        :param dim: Dimensionality of the problem.
        :param sheet_width: Width of the cutting sheet.
        :param sheet_height: Height of the cutting sheet.
        :param recortes_disponiveis: List of available parts (JSON structure).
        :param use_gpu: Whether to use GPU acceleration if available.
        :param show_progress: Whether to show progress bar during optimization.
        :param visualization_interval: How often to update visualization (iterations).
        """
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dim = dim
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.recortes_disponiveis = recortes_disponiveis
        self.initial_layout = recortes_disponiveis
        self.particles = []  # Lista de partículas
        self.velocities = []  # Lista de velocidades
        self.best_positions = []  # Melhores posições individuais
        self.best_fitness_values = []  # Fitness das melhores posições individuais
        self.global_best = None  # Melhor posição global
        self.global_best_fitness = float('-inf')  # Fitness da melhor posição global
        self.optimized_layout = None
        self.fitness_history = []  # Histórico de fitness para plotagem
        
        # Novos parâmetros
        self.use_gpu = use_gpu and has_gpu  # Usar GPU apenas se disponível e solicitado
        self.show_progress = show_progress
        self.visualization_interval = visualization_interval
        
        # Parâmetros do PSO (ajustados para melhor convergência)
        self.w = 0.7  # Inércia (aumentada para melhor exploração)
        self.w_damp = 0.99  # Fator de amortecimento da inércia
        self.c1 = 1.5  # Coeficiente cognitivo 
        self.c2 = 2.0  # Coeficiente social (aumentado para dar mais peso ao melhor global)
        
        print(f"Particle Swarm Optimization Initialized with {num_particles} particles and {num_iterations} iterations")
        if self.use_gpu:
            print("Using GPU acceleration")

    def initialize_particles(self):
        # Initialize particle positions and velocities
        self.particles = []
        self.velocities = []
        self.best_positions = []
        self.best_fitness_values = []
        self.fitness_history = []
        
        for _ in range(self.num_particles):
            # Criar uma cópia profunda dos recortes para cada partícula
            particle = []
            for recorte in self.recortes_disponiveis:
                # Criar cópia do recorte com novas posições x,y
                novo_recorte = copy.deepcopy(recorte)
                # Obter largura e altura conforme o tipo do recorte
                if recorte['tipo'] == 'circular':
                    width = height = recorte['r'] * 2
                elif recorte['tipo'] == 'triangular':
                    width = recorte['b']
                    height = recorte['h']
                else:  # retangular ou diamante
                    width = recorte['largura']
                    height = recorte['altura']
                
                # Gerar posições aleatórias dentro dos limites da chapa
                novo_recorte['x'] = random.uniform(0, self.sheet_width - width)
                novo_recorte['y'] = random.uniform(0, self.sheet_height - height)
                
                # Adicionar rotação aleatória se o tipo permitir
                if recorte['tipo'] != 'circular':
                    novo_recorte['rotacao'] = random.uniform(0, 360)
                
                particle.append(novo_recorte)
            
            # Gerar velocidades aleatórias
            velocity = []
            for recorte in particle:
                v = {'vx': random.uniform(-5, 5), 'vy': random.uniform(-5, 5)}
                if recorte['tipo'] != 'circular':
                    v['vrot'] = random.uniform(-10, 10)
                velocity.append(v)
            
            # Adicionar partícula e velocidade às listas
            self.particles.append(particle)
            self.velocities.append(velocity)
            
            # Calcular fitness da partícula
            fitness = self.fitness_function(particle)
            
            # Inicializar melhor posição da partícula
            self.best_positions.append(copy.deepcopy(particle))
            self.best_fitness_values.append(fitness)
            
            # Atualizar melhor global se necessário
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = copy.deepcopy(particle)

    def fitness_function(self, particle):
        """Avalia a qualidade da disposição dos recortes na chapa."""
        # Calcula a área total utilizada pelos recortes
        used_area = 0
        for recorte in particle:
            if recorte['tipo'] == 'circular':
                used_area += np.pi * recorte['r'] ** 2
            elif recorte['tipo'] == 'triangular':
                used_area += 0.5 * recorte['b'] * recorte['h']
            else:  # retangular ou diamante
                used_area += recorte['largura'] * recorte['altura']
        
        total_area = self.sheet_width * self.sheet_height
        utilization_score = used_area / total_area  # Quanto mais perto de 1, melhor o aproveitamento

        # Distribuição espacial - recompensa espaçamento entre os recortes
        distribution_score = 0
        for i in range(len(particle)):
            for j in range(i + 1, len(particle)):
                # Calcular distância entre os centros dos recortes
                r1_center_x, r1_center_y = self.get_center(particle[i])
                r2_center_x, r2_center_y = self.get_center(particle[j])
                distance = np.sqrt((r1_center_x - r2_center_x)**2 + (r1_center_y - r2_center_y)**2)
                
                # Recompensar distância adequada (não muito próxima nem muito distante)
                optimal_distance = 30  # Valor ajustável
                distribution_score += min(1, distance / optimal_distance) * 0.02

        # Penalização por sobreposições (AUMENTADA DRASTICAMENTE)
        overlap_penalty = 0
        for i in range(len(particle)):
            for j in range(i + 1, len(particle)):
                if self.is_overlapping(particle[i], particle[j]):
                    overlap_penalty += 1
        
        # Penalização por sair dos limites da chapa (AUMENTADA)
        boundary_penalty = 0
        for recorte in particle:
            if self.is_out_of_bounds(recorte):
                boundary_penalty += 1
        
        # Cálculo do fitness final com pesos ajustados
        fitness = (utilization_score * 0.4) + (distribution_score * 0.1) - (overlap_penalty * 2.0) - (boundary_penalty * 1.5)
        
        # Não limitar o fitness a zero permite uma melhor diferenciação entre layouts ruins
        return fitness

    def get_center(self, recorte):
        """Retorna as coordenadas centrais do recorte"""
        if recorte['tipo'] == 'circular':
            center_x = recorte['x'] + recorte['r']
            center_y = recorte['y'] + recorte['r']
        elif recorte['tipo'] == 'triangular':
            center_x = recorte['x'] + recorte['b']/2
            center_y = recorte['y'] + recorte['h']/2
        else:  # retangular ou diamante
            center_x = recorte['x'] + recorte['largura']/2
            center_y = recorte['y'] + recorte['altura']/2
        return center_x, center_y

    def is_overlapping(self, recorte1, recorte2):
        """Verifica se dois recortes estão sobrepostos (simplificado para bounding boxes)."""
        # Otimizado para melhor desempenho
        
        # Obtém as coordenadas dos cantos dos bounding boxes
        r1_x1, r1_y1 = recorte1['x'], recorte1['y']
        if recorte1['tipo'] == 'circular':
            r1_x2, r1_y2 = r1_x1 + 2*recorte1['r'], r1_y1 + 2*recorte1['r']
        elif recorte1['tipo'] == 'triangular':
            r1_x2, r1_y2 = r1_x1 + recorte1['b'], r1_y1 + recorte1['h']
        else:  # retangular ou diamante
            r1_x2, r1_y2 = r1_x1 + recorte1['largura'], r1_y1 + recorte1['altura']
            
        r2_x1, r2_y1 = recorte2['x'], recorte2['y']
        if recorte2['tipo'] == 'circular':
            r2_x2, r2_y2 = r2_x1 + 2*recorte2['r'], r2_y1 + 2*recorte2['r']
        elif recorte2['tipo'] == 'triangular':
            r2_x2, r2_y2 = r2_x1 + recorte2['b'], r2_y1 + recorte2['h']
        else:  # retangular ou diamante
            r2_x2, r2_y2 = r2_x1 + recorte2['largura'], r2_y1 + recorte2['altura']
        
        # Verificação rápida usando "early return"
        if r1_x2 <= r2_x1 or r2_x2 <= r1_x1 or r1_y2 <= r2_y1 or r2_y2 <= r1_y1:
            return False
        return True
    
    def is_out_of_bounds(self, recorte):
        """Verifica se o recorte está fora dos limites da chapa."""
        x, y = recorte['x'], recorte['y']
        
        # Definir largura e altura baseado no tipo
        if recorte['tipo'] == 'circular':
            width = height = recorte['r'] * 2
        elif recorte['tipo'] == 'triangular':
            width = recorte['b']
            height = recorte['h']
        else:  # retangular ou diamante
            width = recorte['largura']
            height = recorte['altura']
            
        # Verificar se está fora dos limites
        return (x < 0 or y < 0 or 
                x + width > self.sheet_width or 
                y + height > self.sheet_height)

    def evaluate_particles(self):
        """Avalia todas as partículas e atualiza melhores posições."""
        # Usar paralelismo com GPU se disponível
        if self.use_gpu:
            # Implementação simplificada - numa implementação completa, 
            # moveríamos cálculos intensivos para GPU usando CuPy ou CUDA
            pass
        
        best_particle_idx = -1
        best_fitness_current = float('-inf')
        
        for i, particle in enumerate(self.particles):
            # Calcular fitness da partícula atual
            fitness = self.fitness_function(particle)
            
            # Acompanhar melhor partícula da iteração atual
            if fitness > best_fitness_current:
                best_fitness_current = fitness
                best_particle_idx = i
            
            # Atualizar melhor posição pessoal se o fitness atual for melhor
            if fitness > self.best_fitness_values[i]:
                self.best_fitness_values[i] = fitness
                self.best_positions[i] = copy.deepcopy(particle)
                
                # Atualizar melhor posição global se necessário
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = copy.deepcopy(particle)
        
        return best_particle_idx, best_fitness_current

    def update_velocity(self):
        """Atualiza a velocidade de cada partícula."""
        for i in range(self.num_particles):
            # Verificar se esta partícula está presa (fitness muito baixo)
            if self.best_fitness_values[i] < -3.0:
                # Chance de reinicializar a partícula (escape de mínimos locais)
                if random.random() < 0.2:  # 20% de chance
                    print(f"Reinicializando partícula {i} que está presa com fitness {self.best_fitness_values[i]}")
                    particle = []
                    for recorte in self.recortes_disponiveis:
                        novo_recorte = copy.deepcopy(recorte)
                        if recorte['tipo'] == 'circular':
                            width = height = recorte['r'] * 2
                        elif recorte['tipo'] == 'triangular':
                            width = recorte['b']
                            height = recorte['h']
                        else:
                            width = recorte['largura']
                            height = recorte['altura']
                        
                        novo_recorte['x'] = random.uniform(0, self.sheet_width - width)
                        novo_recorte['y'] = random.uniform(0, self.sheet_height - height)
                        
                        if recorte['tipo'] != 'circular':
                            novo_recorte['rotacao'] = random.uniform(0, 360)
                        
                        particle.append(novo_recorte)
                    
                    self.particles[i] = particle
                    continue
            
            for j in range(len(self.particles[i])):
                # Componentes de velocidade: inércia + cognitivo + social
                
                # Fator de inércia (continuar na direção atual)
                self.velocities[i][j]['vx'] = self.w * self.velocities[i][j]['vx']
                self.velocities[i][j]['vy'] = self.w * self.velocities[i][j]['vy']
                
                # Componente cognitivo (atração para melhor posição pessoal)
                r1 = random.random()
                self.velocities[i][j]['vx'] += self.c1 * r1 * (self.best_positions[i][j]['x'] - self.particles[i][j]['x'])
                self.velocities[i][j]['vy'] += self.c1 * r1 * (self.best_positions[i][j]['y'] - self.particles[i][j]['y'])
                
                # Componente social (atração para melhor posição global)
                r2 = random.random()
                self.velocities[i][j]['vx'] += self.c2 * r2 * (self.global_best[j]['x'] - self.particles[i][j]['x'])
                self.velocities[i][j]['vy'] += self.c2 * r2 * (self.global_best[j]['y'] - self.particles[i][j]['y'])
                
                # Limitar velocidades máximas para evitar movimentos muito bruscos
                max_velocity = 10.0
                self.velocities[i][j]['vx'] = max(-max_velocity, min(max_velocity, self.velocities[i][j]['vx']))
                self.velocities[i][j]['vy'] = max(-max_velocity, min(max_velocity, self.velocities[i][j]['vy']))
                
                # Atualizar velocidade de rotação, se aplicável
                if 'rotacao' in self.particles[i][j] and self.particles[i][j]['tipo'] != 'circular':
                    if 'vrot' not in self.velocities[i][j]:
                        self.velocities[i][j]['vrot'] = random.uniform(-5, 5)
                    
                    self.velocities[i][j]['vrot'] = self.w * self.velocities[i][j]['vrot']
                    
                    r1_rot = random.random()
                    r2_rot = random.random()
                    
                    # Calcular diferenças de ângulo (considerando periodicidade)
                    best_pos_diff = (self.best_positions[i][j]['rotacao'] - self.particles[i][j]['rotacao']) % 360
                    if best_pos_diff > 180:
                        best_pos_diff -= 360
                        
                    global_best_diff = (self.global_best[j]['rotacao'] - self.particles[i][j]['rotacao']) % 360
                    if global_best_diff > 180:
                        global_best_diff -= 360
                    
                    self.velocities[i][j]['vrot'] += self.c1 * r1_rot * best_pos_diff
                    self.velocities[i][j]['vrot'] += self.c2 * r2_rot * global_best_diff
                    
                    # Limitar velocidade de rotação
                    self.velocities[i][j]['vrot'] = max(-30, min(30, self.velocities[i][j]['vrot']))

    def update_position(self):
        """Atualiza a posição de cada partícula baseado na velocidade."""
        for i in range(self.num_particles):
            for j in range(len(self.particles[i])):
                # Atualizar posição x e y
                self.particles[i][j]['x'] += self.velocities[i][j]['vx']
                self.particles[i][j]['y'] += self.velocities[i][j]['vy']
                
                # Atualizar rotação, se aplicável
                if 'rotacao' in self.particles[i][j] and 'vrot' in self.velocities[i][j]:
                    self.particles[i][j]['rotacao'] = (self.particles[i][j]['rotacao'] + self.velocities[i][j]['vrot']) % 360
                
                # Garantir que os recortes permaneçam dentro dos limites da chapa
                self.enforce_boundaries(self.particles[i][j])
    
    def enforce_boundaries(self, recorte):
        """Garante que o recorte permaneça dentro dos limites da chapa."""
        # Definir largura e altura baseado no tipo
        if recorte['tipo'] == 'circular':
            width = height = recorte['r'] * 2
        elif recorte['tipo'] == 'triangular':
            width = recorte['b']
            height = recorte['h']
        else:  # retangular ou diamante
            width = recorte['largura']
            height = recorte['altura']
            
        # Limitar posição x
        recorte['x'] = max(0, min(self.sheet_width - width, recorte['x']))
        
        # Limitar posição y
        recorte['y'] = max(0, min(self.sheet_height - height, recorte['y']))

    def get_best_solution(self):
        """Retorna a melhor solução encontrada."""
        return self.global_best
    
    def plot_progress(self):
        """Plota o progresso do fitness ao longo das iterações."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Evolução do Fitness')
        plt.xlabel('Iteração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        plt.show()

    def run(self):
        """
        Executa o loop principal do algoritmo Particle Swarm.
        Este método deve retornar o layout otimizado (estrutura JSON).
        """
        # Inicializar partículas
        self.initialize_particles()
        
        print("Starting optimization...")
        start_time = time.time()
        
        # Ajustar parâmetros iniciais
        self.w = 0.9  # Inércia inicial mais alta para exploração
        self.c1 = 1.2  # Fator cognitivo (pessoal)
        self.c2 = 1.8  # Fator social (global)
        
        # Estratégia de resfriamento da inércia
        w_start = self.w
        w_end = 0.4
        
        # Loop principal do PSO com barra de progresso
        iterator = tqdm(range(self.num_iterations)) if self.show_progress else range(self.num_iterations)
        
        best_fitness_without_improvement = 0
        last_improvement = 0
        
        for iteration in iterator:
            # Avaliar partículas
            best_idx, best_fitness = self.evaluate_particles()
            
            # Registrar melhor fitness para histórico
            self.fitness_history.append(self.global_best_fitness)
            
            # Verificar se houve melhoria
            if len(self.fitness_history) > 1 and self.global_best_fitness > self.fitness_history[-2]:
                last_improvement = iteration
            
            # Se não houver melhoria por muitas iterações, perturbar as partículas
            if iteration - last_improvement > 100:
                print(f"\nEstagnação detectada na iteração {iteration}. Perturbando partículas...")
                self.perturb_particles()
                last_improvement = iteration
            
            # Atualizar velocidades
            self.update_velocity()
            
            # Atualizar posições
            self.update_position()
            
            # Atualizar inércia com esquema de resfriamento linear
            self.w = w_start - (w_start - w_end) * (iteration / self.num_iterations)
            
            # Exibir progresso
            if self.show_progress:
                # Atualiza a descrição da barra de progresso
                iterator.set_description(f"Fitness: {self.global_best_fitness:.4f}")
            elif (iteration + 1) % 10 == 0:
                print(f"Iteração {iteration + 1}/{self.num_iterations}: Melhor fitness = {self.global_best_fitness:.4f}")
            
            # Visualização intermediária
            if (iteration + 1) % self.visualization_interval == 0:
                print(f"\nIteração {iteration + 1}: Melhor fitness = {self.global_best_fitness:.4f}")
                # Mostrar visualização intermediária para debugar
                if self.global_best_fitness > 0:  # Só mostrar layouts bons
                    self.display_layout(self.global_best, title=f"Iteration {iteration+1} - Best Layout")
        
        # Exibir estatísticas finais
        elapsed_time = time.time() - start_time
        print(f"\nOtimização completa em {elapsed_time:.2f} segundos")
        print(f"Melhor fitness: {self.global_best_fitness:.6f}")
        
        # Retornar melhor solução encontrada
        self.optimized_layout = self.global_best
        return self.optimized_layout

    def perturb_particles(self):
        """Perturba as partículas para escapar de mínimos locais"""
        for i in range(self.num_particles):
            # Perturbar apenas algumas partículas
            if random.random() < 0.3:  # 30% das partículas
                continue
                
            for j in range(len(self.particles[i])):
                # Adicionar perturbação às posições
                self.particles[i][j]['x'] += random.uniform(-20, 20)
                self.particles[i][j]['y'] += random.uniform(-20, 20)
                
                # Perturbar rotação se aplicável
                if 'rotacao' in self.particles[i][j]:
                    self.particles[i][j]['rotacao'] = (self.particles[i][j]['rotacao'] + random.uniform(-30, 30)) % 360
                
                # Garantir que continuem dentro dos limites
                self.enforce_boundaries(self.particles[i][j])

    def optimize_and_display(self):
        """
        Mostra o layout inicial, executa a otimização e depois mostra o layout otimizado.
        """
        # Mostrar layout inicial
        print("Displaying initial layout...")
        self.display_layout(self.initial_layout, title="Initial Layout - Particle Swarm")
        
        # Executar a otimização
        print("Running optimization...")
        self.optimized_layout = self.run()
        
        # Mostrar layout otimizado
        print("Displaying optimized layout...")
        self.display_layout(self.optimized_layout, title="Optimized Layout - Particle Swarm")
        
        # Plotar gráfico de progresso
        self.plot_progress()
        
        return self.optimized_layout