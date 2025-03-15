from common.layout_display import LayoutDisplayMixin
import random
import copy
import math
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Any
# try:
#     import cupy as np
#     print("Usando CuPy com GPU (CUDA 12)!")
# except ImportError:
#     import numpy as np
#     print("Usando NumPy (CPU)")

class GeneticAlgorithm(LayoutDisplayMixin):
    def __init__(self, TAM_POP, recortes_disponiveis, sheet_width, sheet_height, numero_geracoes=100):
        print("Algoritmo Genético para Otimização do Corte de Chapa com Maximização de Espaço Contínuo")
        self.TAM_POP = TAM_POP
        
        # Pré-ordenação dos recortes por tipo e área para melhor otimização do espaço
        self.initial_layout = self.pre_ordenar_recortes(recortes_disponiveis)
        
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.POP = []  # População: cada indivíduo é uma permutação de índices
        self.POP_AUX = []
        self.aptidao = []
        self.numero_geracoes = numero_geracoes
        self.melhor_aptidoes = []
        self.optimized_layout = None  # To be set after optimization
        
        # Parâmetros do GA - mais robustos às variações
        self.taxa_cruzamento = 0.70  # Arredondado para evitar sensibilidade extrema
        self.taxa_mutacao = 0.18     # Arredondado para evitar sensibilidade extrema
        self.elitismo = True
        self.tam_torneio = 3
        
        # Melhores resultados
        self.melhor_individuo = None
        self.melhor_fitness = float('-inf')
        self.historico_fitness = []
        
        self.initialize_population()

    def pre_ordenar_recortes(self, recortes):
        """
        Pré-ordena os recortes para otimizar o posicionamento.
        Dá prioridade aos retângulos grandes, depois aos diamantes e por fim aos círculos.
        """
        # Calcula a área de cada recorte
        for i, recorte in enumerate(recortes):
            if recorte['tipo'] == 'retangular':
                area = recorte['largura'] * recorte['altura']
            elif recorte['tipo'] == 'circular':
                area = 3.14159 * recorte['r'] ** 2
            elif recorte['tipo'] == 'diamante':
                area = recorte['largura'] * recorte['altura']  # Aproximação
            else:
                area = 0
            recorte['area'] = area
            recorte['idx_original'] = i  # Mantém o índice original
        
        # Aplica um peso ao tipo de recorte (retangular tem prioridade)
        def chave_ordenacao(recorte):
            tipo_peso = {'retangular': 6, 'diamante': 3, 'circular': 1}.get(recorte['tipo'], 0)
            return (tipo_peso, recorte['area'])
        
        # Retorna cópias para não afetar o original
        return [copy.deepcopy(r) for r in sorted(recortes, key=chave_ordenacao, reverse=True)]

    def initialize_population(self):
        """
        Inicializa a população como permutações dos índices dos recortes.
        Cada indivíduo é uma ordem diferente de posicionamento dos recortes.
        """
        n_recortes = len(self.initial_layout)
        indices_base = list(range(n_recortes))
        
        # Criar população inicial com permutações aleatórias
        for _ in range(self.TAM_POP):
            # Gera uma permutação dos índices
            individuo = indices_base.copy()
            random.shuffle(individuo)
            self.POP.append(individuo)
        
        # Inicializa vetor de aptidão
        self.aptidao = [0] * self.TAM_POP

    def decode_chromosome(self, chromosome):
        """
        Decodifica um cromossomo (permutação) em um layout usando retângulos livres.
        Retorna o layout gerado e informações para avaliação do fitness.
        """
        # Inicia com um único retângulo livre do tamanho da chapa
        free_rects = [(0, 0, self.sheet_width, self.sheet_height)]
        layout_result = []
        max_x_usado = 0
        
        # Posiciona cada recorte de acordo com a ordem do cromossomo
        for idx in chromosome:
            recorte = copy.deepcopy(self.initial_layout[idx])
            
            # Limitar rotações para evitar sobreposições causadas por ângulos estranhos
            rotacoes_possiveis = [0]
            if recorte['tipo'] in ['retangular', 'diamante']:
                rotacoes_possiveis = [0, 90]  # Limitar a 4 rotações possíveis
            
            best_fit = None
            best_rect_idx = -1
            best_rotation = 0
            min_waste = float('inf')
            
            # Para cada rotação possível
            for rotacao in rotacoes_possiveis:
                recorte_temp = copy.deepcopy(recorte)
                recorte_temp['rotacao'] = rotacao
                
                # Obtém dimensões do recorte considerando a rotação
                width, height = self.get_dimensions(recorte_temp)
                
                # Tenta encaixar em cada retângulo livre
                for i, rect in enumerate(free_rects):
                    rx, ry, rw, rh = rect
                    
                    # Verificação mais estrita - garante que cabe completamente
                    if width <= rw and height <= rh:
                        # Preferência para retângulos à esquerda e com menos desperdício
                        waste = (rw - width) + (rh - height)
                        
                        if waste < min_waste:
                            min_waste = waste
                            best_fit = (rx, ry, width, height)
                            best_rect_idx = i
                            best_rotation = rotacao
            
            # Se encontrou um encaixe
            if best_fit is not None:
                rx, ry, width, height = best_fit
                
                # Atualiza o recorte com a posição e rotação encontradas
                recorte['x'] = rx
                recorte['y'] = ry
                recorte['rotacao'] = best_rotation
                layout_result.append(recorte)
                
                # Atualiza a coordenada x máxima usada
                max_x_usado = max(max_x_usado, rx + width)
                
                # Atualiza os retângulos livres - versão mais robusta
                self.update_free_rectangles_improved(free_rects, best_rect_idx, rx, ry, width, height)
                # Força mesclar retângulos livres restantes
                self.merge_free_rectangles(free_rects)
        
        # Calcula a área livre contínua à direita
        espaco_livre_direita = (self.sheet_width - max_x_usado) * self.sheet_height
        
        return {
            'layout': layout_result,
            'n_posicionados': len(layout_result),
            'n_total': len(self.initial_layout),
            'max_x_usado': max_x_usado,
            'espaco_livre_direita': espaco_livre_direita
        }
    
    def update_free_rectangles_improved(self, free_rects, rect_idx, x, y, width, height):
        """
        Versão melhorada para atualizar retângulos livres para evitar sobreposições.
        """
        rx, ry, rw, rh = free_rects[rect_idx]
        
        # Remove o retângulo usado da lista
        del free_rects[rect_idx]
        
        # Dividir o espaço em até 4 novos retângulos (direita, abaixo, esquerda, acima)
        
        # 1. Retângulo à direita do recorte
        if x + width < rx + rw:
            free_rects.append((x + width, ry, (rx + rw) - (x + width), rh))
        
        # 2. Retângulo abaixo do recorte
        if y + height < ry + rh:
            free_rects.append((rx, y + height, rw, (ry + rh) - (y + height)))
        
        # 3. Retângulo à esquerda do recorte (se aplicável)
        if x > rx:
            free_rects.append((rx, ry, x - rx, rh))
        
        # 4. Retângulo acima do recorte (se aplicável)
        if y > ry:
            free_rects.append((rx, ry, rw, y - ry))
        
        # Eliminar retângulos muito pequenos (reduz fragmentação)
        min_size = 15.0  # Aumentado de 10.0 para 15.0 para evitar sobreposições
        i = 0
        while i < len(free_rects):
            if free_rects[i][2] < min_size or free_rects[i][3] < min_size:
                del free_rects[i]
            else:
                i += 1

    def merge_free_rectangles(self, free_rects):
        """
        Mescla retângulos livres sobrepostos para reduzir fragmentação.
        Implementação simplificada que remove retângulos contidos em outros.
        """
        i = 0
        while i < len(free_rects):
            j = i + 1
            while j < len(free_rects):
                r1 = free_rects[i]
                r2 = free_rects[j]
                
                # Se r2 está contido em r1, remove r2
                if (r1[0] <= r2[0] and r1[1] <= r2[1] and 
                    r1[0] + r1[2] >= r2[0] + r2[2] and 
                    r1[1] + r1[3] >= r2[1] + r2[3]):
                    free_rects.pop(j)
                # Se r1 está contido em r2, remove r1
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
        """
        Retorna as dimensões (largura, altura) de um recorte considerando sua rotação.
        Adiciona uma pequena margem de segurança para evitar sobreposições.
        """
        tipo = recorte['tipo']
        rotacao = recorte.get('rotacao', 0)
        margin = 1.0  # Margem de segurança de 1 unidade
        
        if tipo == 'circular':
            diametro = 2 * recorte['r'] + margin
            return diametro, diametro
            
        elif tipo == 'diamante':
            # Diamante é um caso específico que pode ser rotacionado
            largura = recorte['largura'] + margin
            altura = recorte['altura'] + margin
            if rotacao in [90]:
                return altura, largura
            return largura, altura
            
        else:  # retangular
            largura = recorte['largura'] + margin
            altura = recorte['altura'] + margin
            if rotacao in [90]:
                return altura, largura
            return largura, altura


    def detect_overlaps(self, layout):
        """
        Verifica se há sobreposições entre peças no layout.
        Retorna o número de sobreposições detectadas.
        """
        num_overlaps = 0
        
        # Para cada par de peças no layout
        for i in range(len(layout)):
            for j in range(i + 1, len(layout)):
                piece1 = layout[i]
                piece2 = layout[j]
                
                # Obter coordenadas e dimensões das peças
                x1, y1 = piece1['x'], piece1['y']
                w1, h1 = self.get_dimensions(piece1)
                
                x2, y2 = piece2['x'], piece2['y']
                w2, h2 = self.get_dimensions(piece2)
                
                # Verificar sobreposição de retângulos
                # Não há sobreposição se uma está à esquerda/direita/acima/abaixo da outra
                if not (x1 + w1 <= x2 or x2 + w2 <= x1 or 
                        y1 + h1 <= y2 or y2 + h2 <= y1):
                    num_overlaps += 1
        
        return num_overlaps

    def evaluate(self):
        """
        Avalia a aptidão de cada indivíduo na população, considerando múltiplas áreas livres contínuas.
        """
        for i in range(self.TAM_POP):
            decoded = self.decode_chromosome(self.POP[i])
            
            # Extrair dados do layout decodificado
            layout = decoded['layout']
            n_posicionados = decoded['n_posicionados']
            n_total = decoded['n_total']
            taxa_posicionamento = n_posicionados / n_total
            
            # Cálculo de áreas livres contínuas
            espaco_livre_direita = decoded['espaco_livre_direita']
            area_total = self.sheet_width * self.sheet_height
            taxa_espaco_livre = espaco_livre_direita / area_total
            
            # Componente de compactação (preferência para peças à esquerda)
            compactacao = 0
            if layout:
                # Quanto menor o x máximo usado, melhor a compactação horizontal
                taxa_compactacao_x = 1.0 - (decoded['max_x_usado'] / self.sheet_width)
                compactacao = taxa_compactacao_x * 0.5
            
            # Cálculo da eficiência por tipo (penaliza layouts onde peças grandes ocupam áreas ineficientes)
            eficiencia_tipo = self.calcular_eficiencia_tipo(layout)
            
            # Detectar sobreposições e aplicar penalidade
            num_overlaps = self.detect_overlaps(layout)
            overlap_penalty = num_overlaps * 1000.0  # Penalidade EXTREMAMENTE alta para sobreposição
            
            # Fitness final com todos os componentes
            fitness = (
                (taxa_posicionamento * 150) + 
                (taxa_espaco_livre * 80) +  # Aumentado o peso do espaço livre
                (compactacao * 20) + 
                (eficiencia_tipo * 30) -    # Novo componente de eficiência por tipo
                overlap_penalty
            )
            
            # Penalização para peças não posicionadas
            if n_posicionados < n_total:
                fitness -= (n_total - n_posicionados) * 200
            
            self.aptidao[i] = fitness
            
            # Atualiza o melhor indivíduo se necessário
            if fitness > self.melhor_fitness:
                # Só atualiza o melhor se não houver sobreposições
                if num_overlaps == 0:
                    self.melhor_fitness = fitness
                    self.melhor_individuo = self.POP[i].copy()
                    self.optimized_layout = layout

    def calcular_eficiencia_tipo(self, layout):
        """
        Calcula um score de eficiência baseado em como os diferentes tipos de peças estão posicionados.
        Penaliza losangos que ocupam áreas que seriam melhor ocupadas por retângulos.
        """
        if not layout:
            return 0
        
        score = 0
        for peca in layout:
            # Retângulos próximos às bordas recebem bônus
            if peca['tipo'] == 'retangular':
                distancia_borda = min(peca['x'], peca['y'])
                if distancia_borda < 20:  # Se estiver próximo à borda
                    score += 2
            
            # Losangos no meio da chapa recebem bônus
            elif peca['tipo'] == 'diamante':
                centro_x = self.sheet_width / 2
                centro_y = self.sheet_height / 2
                distancia_centro = math.sqrt((peca['x'] - centro_x)**2 + (peca['y'] - centro_y)**2)
                if distancia_centro > self.sheet_width / 4:  # Se estiver afastado do centro
                    score -= 1  # Penalidade para losangos longe do centro
        
        # Normaliza o score
        return score / len(layout)

    def selection_tournament(self):
        """Seleção por torneio."""
        selected = []
        for _ in range(self.TAM_POP):
            # Seleciona indivíduos aleatórios para o torneio
            competitors = random.sample(range(self.TAM_POP), self.tam_torneio)
            # Encontra o melhor competidor
            best = max(competitors, key=lambda idx: self.aptidao[idx])
            selected.append(self.POP[best].copy())
        return selected

    def crossover_order(self, parent1, parent2):
        """
        Cruzamento que preserva a ordem dos elementos (Order Crossover - OX).
        Adequado para problemas de permutação.
        """
        if random.random() > self.taxa_cruzamento:
            return parent1.copy()
            
        size = len(parent1)
        # Seleciona dois pontos de corte
        p1, p2 = sorted(random.sample(range(size), 2))
        
        # Cria filho com segmento do primeiro pai
        child = [None] * size
        for i in range(p1, p2 + 1):
            child[i] = parent1[i]
        
        # Preenche o restante com elementos do segundo pai, mantendo a ordem
        j = p2 + 1
        for i in range(p2 + 1, p2 + 1 + size):
            idx = i % size
            val = parent2[idx]
            if val not in child:
                child[j % size] = val
                j += 1
        
        return child

    def mutation_swap(self, individual):
        """
        Mutação por troca (swap mutation).
        Troca dois genes aleatórios de posição.
        """
        if random.random() <= self.taxa_mutacao:
            size = len(individual)
            i, j = random.sample(range(size), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def genetic_operators(self):
        """Aplica os operadores genéticos para criar a nova população."""
        # Seleção
        selected = self.selection_tournament()
        
        # Nova população
        new_pop = []
        
        # Elitismo: mantém o melhor indivíduo
        if self.elitismo and self.melhor_individuo is not None:
            new_pop.append(self.melhor_individuo.copy())
        
        # Cruzamento e Mutação
        while len(new_pop) < self.TAM_POP:
            # Seleciona pais
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            # Aplica cruzamento
            child = self.crossover_order(parent1, parent2)
            
            # Aplica mutação
            child = self.mutation_swap(child)
            
            new_pop.append(child)
        
        # Garante que a população não exceda o tamanho definido
        self.POP = new_pop[:self.TAM_POP]

    def run(self):
        """Executa o algoritmo genético por um número definido de gerações."""
        # Barra de progresso
        progress_bar = tqdm(total=self.numero_geracoes, desc="Otimização GA", ncols=100)
        
        for gen in range(self.numero_geracoes):
            # Avalia a população
            self.evaluate()
            
            # Registra o melhor fitness desta geração
            best_fitness = max(self.aptidao)
            self.melhor_aptidoes.append(best_fitness)
            
            # Atualiza a descrição da barra de progresso
            progress_bar.set_description(f"GA | Fitness: {best_fitness:.2f}")
            progress_bar.update(1)
            
            # Mostra progresso a cada 5 gerações (em vez de 10)
            if gen % 5 == 0:
                print(f"Geração {gen}: Melhor Fitness = {best_fitness:.2f}")
                if self.optimized_layout:
                    print(f"Recortes posicionados: {len(self.optimized_layout)}/{len(self.initial_layout)}")
            
            # Aplica operadores genéticos para criar a próxima geração
            self.genetic_operators()
        
        progress_bar.close()
        
        # Avalia a população final
        self.evaluate()
        
        # Decodifica o melhor indivíduo para gerar o layout final
        if self.melhor_individuo:
            decoded = self.decode_chromosome(self.melhor_individuo)
            self.optimized_layout = decoded['layout']
            
            # Imprime resultado final
            print(f"\nResultado Final após {self.numero_geracoes} gerações:")
            print(f"Melhor Fitness: {self.melhor_fitness:.2f}")
            print(f"Recortes posicionados: {len(self.optimized_layout)}/{len(self.initial_layout)}")
            espaco_livre = decoded['espaco_livre_direita']
            print(f"Espaço livre contínuo: {espaco_livre:.2f} ({100*espaco_livre/(self.sheet_width*self.sheet_height):.1f}%)")
        
        return self.optimized_layout

    def optimize_and_display(self):
        """
        Exibe o layout inicial, executa o algoritmo genético e exibe o layout otimizado.
        """
        # Exibe layout inicial
        self.display_layout(self.initial_layout, title="Layout Inicial - Algoritmo Genético")
        
        # Executa a otimização
        self.optimized_layout = self.run()
        
        # Exibe layout otimizado
        self.display_layout(self.optimized_layout, title="Layout Otimizado - Algoritmo Genético")
        
        return self.optimized_layout