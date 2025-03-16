from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Any

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        print("Algoritmo Genético para Otimização do Corte de Chapa com Maximização de Espaço Contínuo")
        self.TAM_POP = TAM_POP
        self.initial_layout = self.pre_ordenar_recortes(recortes_disponiveis)
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.POP = []
        self.aptidao = []
        self.numero_geracoes = numero_geracoes
        self.melhor_aptidoes = []
        self.optimized_layout = None
        
        # Parâmetros otimizados
        self.taxa_cruzamento = 0.70
        self.taxa_mutacao = 0.20
        self.elitismo = True
        self.tam_torneio = 3
        
        self.melhor_individuo = None
        self.melhor_fitness = float('-inf')
        
        self.initialize_population()

    def pre_ordenar_recortes(self, recortes):
        """Pré-ordena os recortes para otimizar o posicionamento.
        Este método classifica os recortes antes da execução do algoritmo genético, atribuindo uma área a cada recorte e ordenando-os por um critério de prioridade.
        Os retângulos têm a maior prioridade (peso = 6), seguidos pelos diamantes (peso = 3) e depois pelos círculos (peso = 1).
        Isso ajuda a tentar alocar os formatos de maneira mais eficiente desde o início.
        O uso de copy.deepcopy() garante que a lista original de recortes não seja alterada acidentalmente.
        """
        for i, recorte in enumerate(recortes):
            if recorte['tipo'] == 'retangular':
                area = recorte['largura'] * recorte['altura']
            elif recorte['tipo'] == 'circular':
                area = math.pi * recorte['r'] ** 2
            elif recorte['tipo'] == 'diamante':
                area = recorte['largura'] * recorte['altura']
            else:
                area = 0
            recorte['area'] = area
            recorte['idx_original'] = i
        
        def chave_ordenacao(recorte):
            tipo_peso = {'retangular': 6, 'diamante': 3, 'circular': 1}.get(recorte['tipo'], 0)
            return (tipo_peso, recorte['area'])
        
        return [copy.deepcopy(r) for r in sorted(recortes, key=chave_ordenacao, reverse=True)]

    def initialize_population(self):
        """Este método inicializa a população do algoritmo genético.
            Cada indivíduo na população é uma permutação dos índices dos recortes.
            O método cria TAM_POP indivíduos aleatórios embaralhando a lista de índices dos recortes.
            A aptidão de cada indivíduo é inicializada como zero.
            """
        n_recortes = len(self.initial_layout)
        indices_base = list(range(n_recortes))
        
        for _ in range(self.TAM_POP):
            individuo = indices_base.copy()
            random.shuffle(individuo)
            self.POP.append(individuo)
        
        self.aptidao = [0] * self.TAM_POP

    def decode_chromosome(self, chromosome):
        """Este método traduz um cromossomo (sequência de índices de recortes) para um layout real.
        Utiliza uma abordagem de "caixa livre" para alocar os recortes na chapa.
        Calcula o espaço livre restante à direita da chapa, que pode ser usado para medir a eficiência do arranjo.
        A lógica de melhor ajuste minimiza o desperdício ao selecionar o espaço adequado para cada peça.
        """
        free_rects = [(0, 0, self.sheet_width, self.sheet_height)]
        layout_result = []
        max_x_usado = 0
        
        for idx in chromosome:
            recorte = copy.deepcopy(self.initial_layout[idx])
            best_fit = None
            best_rect_idx = -1
            best_rotation = 0
            min_waste = float('inf')
            
            width, height = self.get_dimensions(recorte)
            
            for i, rect in enumerate(free_rects):
                rx, ry, rw, rh = rect
                if width <= rw and height <= rh:
                    waste = (rw - width) + (rh - height)
                    if waste < min_waste:
                        min_waste = waste
                        best_fit = (rx, ry, width, height)
                        best_rect_idx = i
                        best_rotation = 0
            
            if best_fit is not None:
                rx, ry, width, height = best_fit
                recorte['x'] = rx
                recorte['y'] = ry
                recorte['rotacao'] = best_rotation
                layout_result.append(recorte)
                max_x_usado = max(max_x_usado, rx + width)
                self.update_free_rectangles(free_rects, best_rect_idx, rx, ry, width, height)
                self.merge_free_rectangles(free_rects)
        
        espaco_livre_direita = (self.sheet_width - max_x_usado) * self.sheet_height
        
        return {
            'layout': layout_result,
            'n_posicionados': len(layout_result),
            'n_total': len(self.initial_layout),
            'max_x_usado': max_x_usado,
            'espaco_livre_direita': espaco_livre_direita
        }
    
    def update_free_rectangles(self, free_rects, rect_idx, x, y, width, height):
        """
        Este método atualiza a lista de espaços livres na chapa após um recorte ser posicionado.
        Remove o retângulo original onde o recorte foi colocado.
        Divide esse retângulo em possíveis áreas livres restantes.
        Elimina espaços livres muito pequenos (min_size = 15.0).
        """
        rx, ry, rw, rh = free_rects[rect_idx]
        del free_rects[rect_idx]
        
        if x + width < rx + rw:
            free_rects.append((x + width, ry, (rx + rw) - (x + width), rh))
        if y + height < ry + rh:
            free_rects.append((rx, y + height, rw, (ry + rh) - (y + height)))
        if x > rx:
            free_rects.append((rx, ry, x - rx, rh))
        if y > ry:
            free_rects.append((rx, ry, rw, y - ry))
        
        min_size = 15.0
        i = 0
        while i < len(free_rects):
            if free_rects[i][2] < min_size or free_rects[i][3] < min_size:
                del free_rects[i]
            else:
                i += 1

    def merge_free_rectangles(self, free_rects):
        """"
        Este método tenta combinar áreas livres sobrepostas ou redundantes.
        Percorre a lista de retângulos livres e remove aqueles que são completamente cobertos por outros.
        """	
        i = 0
        while i < len(free_rects):
            j = i + 1
            while j < len(free_rects):
                r1, r2 = free_rects[i], free_rects[j]
                if (r1[0] <= r2[0] and r1[1] <= r2[1] and 
                    r1[0] + r1[2] >= r2[0] + r2[2] and 
                    r1[1] + r1[3] >= r2[1] + r2[3]):
                    free_rects.pop(j)
                elif (r2[0] <= r1[0] and r2[1] <= r1[1] and 
                      r2[0] + r2[2] >= r1[0] + r1[2] and 
                      r2[1] + r2[3] >= r1[1] + r1[3]):
                    free_rects.pop(i)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1

    def get_dimensions(self, recorte):
        """	Este método retorna as dimensões de um recorte, adicionando uma margem de segurança.
        Círculos têm largura e altura iguais ao diâmetro.
        Diamantes e retângulos mantêm suas dimensões.
        """	
        tipo = recorte['tipo']
        margin = 1.0
        
        if tipo == 'circular':
            diametro = 2 * recorte['r'] + margin
            return diametro, diametro
        elif tipo == 'diamante':
            return recorte['largura'] + margin, recorte['altura'] + margin
        else:  # retangular
            return recorte['largura'] + margin, recorte['altura'] + margin

    def detect_overlaps(self, layout):
        """Este método verifica se há sobreposição entre recortes no layout.
        Percorre pares de recortes e verifica se seus retângulos se sobrepõem.
        Retorna o número total de sobreposições.
        """
        num_overlaps = 0
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                piece1, piece2 = layout[i], layout[j]
                x1, y1 = piece1['x'], piece1['y']
                w1, h1 = self.get_dimensions(piece1)
                x2, y2 = piece2['x'], piece2['y']
                w2, h2 = self.get_dimensions(piece2)
                
                if not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1):
                    num_overlaps += 1
        
        return num_overlaps

    def evaluate(self):
        """"Este método avalia cada indivíduo da população atribuindo um valor de aptidão (fitness).
        Penaliza layouts com sobreposição e peças não posicionadas.
        Maximiza a compactação e minimiza o espaço livre desperdiçado.
        O fitness considera: Taxa de posicionamento dos recortes, espaço livre na chapa, compactação horizontal, eficiência do tipo de recorte e penalidade por sobreposição.
        """
        for i in range(self.TAM_POP):
            decoded = self.decode_chromosome(self.POP[i])
            layout = decoded['layout']
            n_posicionados = decoded['n_posicionados']
            n_total = decoded['n_total']
            taxa_posicionamento = n_posicionados / n_total
            
            espaco_livre_direita = decoded['espaco_livre_direita']
            area_total = self.sheet_width * self.sheet_height
            taxa_espaco_livre = espaco_livre_direita / area_total
            
            # Componente de compactação
            compactacao = 0
            if layout:
                taxa_compactacao_x = 1.0 - (decoded['max_x_usado'] / self.sheet_width)
                compactacao = taxa_compactacao_x * 0.5
            
            eficiencia_tipo = self.calcular_eficiencia_tipo(layout)
            num_overlaps = self.detect_overlaps(layout)
            overlap_penalty = num_overlaps * 1000.0
            
            # Fitness final com pesos otimizados
            fitness = (
                (taxa_posicionamento * 150) + 
                (taxa_espaco_livre * 80) +
                (compactacao * 20) + 
                (eficiencia_tipo * 30) -
                overlap_penalty
            )
            
            # Penalidade para peças não posicionadas
            if n_posicionados < n_total:
                fitness -= (n_total - n_posicionados) * 200
            
            self.aptidao[i] = fitness
            
            # Atualiza o melhor indivíduo
            if fitness > self.melhor_fitness and num_overlaps == 0:
                self.melhor_fitness = fitness
                self.melhor_individuo = self.POP[i].copy()
                self.optimized_layout = layout

    def calcular_eficiencia_tipo(self, layout):
        """"Este método deveria calcular a eficiência do layout com base nos tipos de recortes utilizados.
        No código enviado, ele ainda não está implementado.
        Pode ser usado para favorecer layouts que maximizem determinados tipos de recortes.
        """
        if not layout:
            return 0
        
        score = 0
        for peca in layout:
            if peca['tipo'] == 'retangular':
                distancia_borda = min(peca['x'], peca['y'])
                if distancia_borda < 20:
                    score += 2
            elif peca['tipo'] == 'diamante':
                centro_x, centro_y = self.sheet_width / 2, self.sheet_height / 2
                distancia_centro = math.sqrt((peca['x'] - centro_x)**2 + (peca['y'] - centro_y)**2)
                if distancia_centro > self.sheet_width / 4:
                    score -= 1
        
        return score / len(layout)

    def selection_tournament(self):
        selected = []
        for _ in range(self.TAM_POP):
            competitors = random.sample(range(self.TAM_POP), self.tam_torneio)
            best = max(competitors, key=lambda idx: self.aptidao[idx])
            selected.append(self.POP[best].copy())
        return selected

    def crossover_order(self, parent1, parent2):
        if random.random() > self.taxa_cruzamento:
            return parent1.copy()
            
        size = len(parent1)
        p1, p2 = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        for i in range(p1, p2 + 1):
            child[i] = parent1[i]
        
        j = p2 + 1
        for i in range(p2 + 1, p2 + 1 + size):
            idx = i % size
            val = parent2[idx]
            if val not in child:
                child[j % size] = val
                j += 1
        
        return child

    def mutation_swap(self, individual):
        if random.random() <= self.taxa_mutacao:
            size = len(individual)
            i, j = random.sample(range(size), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def genetic_operators(self):
        selected = self.selection_tournament()
        new_pop = []
        
        if self.elitismo and self.melhor_individuo is not None:
            new_pop.append(self.melhor_individuo.copy())
        
        while len(new_pop) < self.TAM_POP:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = self.crossover_order(parent1, parent2)
            child = self.mutation_swap(child)
            new_pop.append(child)
        
        self.POP = new_pop[:self.TAM_POP]

    def run(self):
        progress_bar = tqdm(total=self.numero_geracoes, desc="Otimização GA", ncols=100)
        
        for gen in range(self.numero_geracoes):
            self.evaluate()
            best_fitness = max(self.aptidao)
            self.melhor_aptidoes.append(best_fitness)
            
            progress_bar.set_description(f"GA | Fitness: {best_fitness:.2f}")
            progress_bar.update(1)
            
            if gen % 5 == 0:
                print(f"Geração {gen}: Melhor Fitness = {best_fitness:.2f}")
                if self.optimized_layout:
                    print(f"Recortes posicionados: {len(self.optimized_layout)}/{len(self.initial_layout)}")
            
            self.genetic_operators()
        
        progress_bar.close()
        self.evaluate()
        
        if self.melhor_individuo:
            decoded = self.decode_chromosome(self.melhor_individuo)
            self.optimized_layout = decoded['layout']
            espaco_livre_total = decoded['espaco_livre_direita'] + len(self.optimized_layout) * 1.0
            
            print(f"\nResultado Final após {self.numero_geracoes} gerações:")
            print(f"Melhor Fitness: {self.melhor_fitness:.2f}")
            print(f"Recortes posicionados: {len(self.optimized_layout)}/{len(self.initial_layout)}")
            print(f"Espaço livre total: {espaco_livre_total:.2f} unidades")
            print(f"Componente de compactação: {decoded['max_x_usado']:.2f}/{self.sheet_width}")
            print(f"Componente de eficiência: {self.calcular_eficiencia_tipo(self.optimized_layout):.2f}")
            print(f"Colisões: {self.detect_overlaps(self.optimized_layout)}")
            print(f"Taxa de posicionamento: {decoded['n_posicionados']}/{decoded['n_total']}")
            print(f"Espaço livre à direita: {decoded['espaco_livre_direita']:.2f} unidades")
 
        
        return self.optimized_layout

    def optimize_and_display(self):
        self.display_layout(self.initial_layout, title="Layout Inicial - Algoritmo Genético")
        self.optimized_layout = self.run()
        self.display_layout(self.optimized_layout, title="Layout Otimizado - Algoritmo Genético")
        self.plot_fitness_evolution()
        return self.optimized_layout
        
    def plot_fitness_evolution(self):
        """
        Exibe um gráfico mostrando a evolução do fitness ao longo das gerações.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.melhor_aptidoes)), self.melhor_aptidoes, 'b-', linewidth=2)
        plt.title('Evolução do Fitness - Algoritmo Genético')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.grid(True)
        
        # Destaque para o valor final
        plt.scatter(len(self.melhor_aptidoes)-1, self.melhor_aptidoes[-1], 
                   color='red', s=100, zorder=5)
        plt.annotate(f'Final: {self.melhor_aptidoes[-1]:.2f}', 
                    (len(self.melhor_aptidoes)-1, self.melhor_aptidoes[-1]),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=12, color='darkred')
        
        plt.tight_layout()
        plt.show()