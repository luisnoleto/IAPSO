from common.layout_display import LayoutDisplayMixin
import numpy as np
import random
import copy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class ParticleSwarm(LayoutDisplayMixin):
    def __init__(self, num_particles, num_iterations, dim, sheet_width, sheet_height, recortes_disponiveis,
                 use_gpu=False, show_progress=True, visualization_interval=10):
        """
        Inicializa o otimizador Particle Swarm.
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
        
        # Parâmetros simplificados
        self.show_progress = show_progress
        
        # Parâmetros PSO - SIMPLIFICADOS E ESTÁVEIS
        self.w = 0.7  # Inércia - valor estável
        self.c1 = 1.5  # Componente cognitivo (best pessoal)
        self.c2 = 1.5  # Componente social (best global)
        
        # Contador para estagnação
        self.stagnation_counter = 0
        
        print(f"Particle Swarm Optimization iniciado com {num_particles} partículas e {num_iterations} iterações")

    def initialize_particles(self):
        """Inicializa partículas com posições aleatórias e velocidades."""
        self.particles = []
        self.velocities = []
        self.best_positions = []
        self.best_fitness_values = []
        self.fitness_history = []
        
        for _ in range(self.num_particles):
            # Criar uma cópia profunda dos recortes para cada partícula
            particle = []
            for recorte in self.recortes_disponiveis:
                # Criar cópia do recorte com novas posições
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
            
            # Gerar velocidades aleatórias moderadas
            velocity = []
            for recorte in particle:
                v = {'vx': random.uniform(-2, 2), 'vy': random.uniform(-2, 2)}
                if recorte['tipo'] != 'circular':
                    v['vrot'] = random.uniform(-5, 5)
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
        """
        Avalia a qualidade da disposição dos recortes na chapa.
        """
        # 1. Área utilizada (positivo)
        used_area = 0
        for recorte in particle:
            if recorte['tipo'] == 'circular':
                used_area += np.pi * recorte['r'] ** 2
            elif recorte['tipo'] == 'triangular':
                used_area += 0.5 * recorte['b'] * recorte['h']
            else:  # retangular ou diamante
                used_area += recorte['largura'] * recorte['altura']
        
        total_area = self.sheet_width * self.sheet_height
        utilization_score = used_area / total_area  # Entre 0 e 1
        
        # 2. Distribuição - recompensa espaço entre peças (positivo)
        distribution_score = 0
        for i in range(len(particle)):
            for j in range(i + 1, len(particle)):
                r1_center_x, r1_center_y = self.get_center(particle[i])
                r2_center_x, r2_center_y = self.get_center(particle[j])
                distance = np.sqrt((r1_center_x - r2_center_x)**2 + (r1_center_y - r2_center_y)**2)
                
                # Recompensa por distância adequada entre peças
                min_distance = 5  # Distância mínima desejada
                if distance > min_distance:
                    distribution_score += 0.1
        
        # Normalizar score de distribuição para um valor entre 0 e 1
        if len(particle) > 1:
            distribution_score /= (len(particle) * (len(particle) - 1) / 2)
        
        # 3. Penalidade por sobreposição (negativo) - MENOS AGRESSIVA
        overlap_penalty = 0
        for i in range(len(particle)):
            for j in range(i + 1, len(particle)):
                if self.is_overlapping(particle[i], particle[j]):
                    r1_center_x, r1_center_y = self.get_center(particle[i])
                    r2_center_x, r2_center_y = self.get_center(particle[j])
                    distance = np.sqrt((r1_center_x - r2_center_x)**2 + (r1_center_y - r2_center_y)**2)
                    
                    # Penalidade proporcional à gravidade da sobreposição
                    overlap_penalty += 0.5  # Penalidade mais suave
        
        # 4. Penalidade por sair dos limites (negativo)
        boundary_penalty = 0
        for recorte in particle:
            if self.is_out_of_bounds(recorte):
                boundary_penalty += 0.5
        
        # Cálculo final do fitness com pesos mais equilibrados
        fitness = (utilization_score * 1.0) + (distribution_score * 1.8) - (overlap_penalty * 1.2) - (boundary_penalty * 1.0)
        
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

    def update_velocity(self):
        """Atualiza a velocidade de cada partícula."""
        for i in range(self.num_particles):
            for j in range(len(self.particles[i])):
                # Componentes de velocidade simplificados
                
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
                
                # Limitar velocidades para evitar movimentos muito bruscos
                max_velocity = 5.0  # Valor reduzido para movimentos mais suaves
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
                    self.velocities[i][j]['vrot'] = max(-20, min(20, self.velocities[i][j]['vrot']))

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
    
    def perturb_particles(self, intensity=0.5):
        """
        Perturba as partículas para aumentar exploração.
        :param intensity: Intensidade da perturbação (0-1)
        """
        intensity = intensity * 50  # Aumentar intensidade para perturbações mais significativas
        print(f"Aplicando perturbação com intensidade {intensity:.2f}")
        
        # Preservar a melhor partícula
        best_index = self.best_fitness_values.index(max(self.best_fitness_values))
        
        # Calcular número de partículas a perturbar baseado na intensidade
        num_to_perturb = int(self.num_particles * min(0.9, intensity))
        
        # Selecionar índices de partículas para perturbar (exceto a melhor)
        indices = list(range(self.num_particles))
        indices.remove(best_index)
        random.shuffle(indices)
        indices_to_perturb = indices[:num_to_perturb]
        
        for i in indices_to_perturb:
            for j in range(len(self.particles[i])):
                # Deslocamento proporcional à intensidade (muito mais agressivo)
                max_shift = intensity * 90  # Aumentado para 50 (antes era 20)
                
                # Aplicar perturbação diretamente, não baseada em probabilidade
                self.particles[i][j]['x'] += random.uniform(-max_shift, max_shift)
                self.particles[i][j]['y'] += random.uniform(-max_shift, max_shift)
                
                # Perturbar rotação de forma mais significativa
                if 'rotacao' in self.particles[i][j]:
                    angle_shift = intensity * 20  # Aumentado para 60 graus (antes era 30)
                    self.particles[i][j]['rotacao'] = random.uniform(0, 360)  # Rotação completamente nova
                
                # Garantir limites
                self.enforce_boundaries(self.particles[i][j])
    

    def run(self):
        """
        Executa o algoritmo PSO simplificado.
        """
        # Inicialização
        self.initialize_particles()
        
        print("Iniciando otimização...")
        start_time = time.time()
        
        # Definir parâmetros de controle de estagnação
        best_fitness_so_far = float('-inf')
        stagnation_counter = 0
        stagnation_threshold = 50  # Número de iterações sem melhoria para considerar estagnação
        
        # Loop principal - SIMPLIFICADO
        iterator = tqdm(range(self.num_iterations), desc="Otimização PSO")
        for iteration in iterator:
            # 1. Avaliar partículas
            for i, particle in enumerate(self.particles):
                fitness = self.fitness_function(particle)
                
                # Atualizar melhor posição pessoal
                if fitness > self.best_fitness_values[i]:
                    self.best_fitness_values[i] = fitness
                    self.best_positions[i] = copy.deepcopy(particle)
                    
                    # Atualizar melhor global
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best = copy.deepcopy(particle)
            
            # Registrar histórico
            self.fitness_history.append(self.global_best_fitness)
            
            # 2. Verificar estagnação
            if self.global_best_fitness > best_fitness_so_far:
                best_fitness_so_far = self.global_best_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Aplicar perturbação se estagnado
            if stagnation_counter >= stagnation_threshold:
                intensity = min(0.1 + (stagnation_counter - stagnation_threshold) / 100, 0.8)
                self.perturb_particles(intensity)
                stagnation_counter = 0  # Resetar contador
            
            # 3. Atualizar velocidades e posições
            self.update_velocity()
            self.update_position()
            
            # Atualizar descrição da barra de progresso
            if self.show_progress:
                iterator.set_description(f"Fitness: {self.global_best_fitness:.4f}")
        
        # Refinamento final - focar na melhor solução
        print("Refinamento final...")
        self.w = 0.5  # Menor inércia
        self.c1 = 1.0  # Menos exploração pessoal
        self.c2 = 2.0  # Mais convergência global
        
        for _ in range(self.num_iterations // 10):
            # Avaliar e atualizar
            for i, particle in enumerate(self.particles):
                fitness = self.fitness_function(particle)
                
                # Verificar se não há sobreposição antes de aceitar como melhor
                has_overlap = any(self.is_overlapping(particle[i1], particle[i2]) 
                                for i1 in range(len(particle)) 
                                for i2 in range(i1+1, len(particle)))
                
                if not has_overlap and fitness > self.best_fitness_values[i]:
                    self.best_fitness_values[i] = fitness
                    self.best_positions[i] = copy.deepcopy(particle)
                    
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best = copy.deepcopy(particle)
            
            self.update_velocity()
            self.update_position()
        
        elapsed_time = time.time() - start_time
        print(f"\nOtimização completa em {elapsed_time:.2f} segundos")
        print(f"Melhor fitness: {self.global_best_fitness:.6f}")
        
        # Retornar melhor solução
        self.optimized_layout = self.global_best
        return self.optimized_layout
    
    def plot_progress(self):
        """Plota o progresso do fitness ao longo das iterações."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Evolução do Fitness')
        plt.xlabel('Iteração')
        plt.ylabel('Melhor Fitness')
        plt.grid(True)
        plt.show()
    
    def optimize_and_display(self):
        """Mostra o layout inicial, executa a otimização e depois mostra o layout otimizado."""
        # Mostrar layout inicial
        print("Exibindo layout inicial...")
        self.display_layout(self.initial_layout, title="Layout Inicial - Particle Swarm")
        
        # Executar a otimização
        print("Executando otimização...")
        self.optimized_layout = self.run()
        
        # Mostrar layout otimizado
        print("Exibindo layout otimizado...")
        self.display_layout(self.optimized_layout, title="Layout Otimizado - Particle Swarm")
        
        # Plotar gráfico de progresso
        self.plot_progress()
        
        return self.optimized_layout