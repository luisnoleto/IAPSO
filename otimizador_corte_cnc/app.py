from genetic_algorithm import GeneticAlgorithm

def main():
    # Define sheet dimensions
    sheet_width = 200
    sheet_height = 100

    # Define available parts (recortes_disponiveis) as a JSON-like structure.
    # Examples of one of each type of part:
    # recortes_disponiveis = [
    #     {"tipo": "retangular", "largura": 20, "altura": 10, "x": 0, "y": 0, "rotacao": 0},
    #     {"tipo": "circular", "r": 10, "x": 0, "y": 0},
    #     {"tipo": "triangular", "b": 25, "h": 20, "x": 0, "y": 0, "rotacao": 10},
    #     {"tipo": "diamante", "largura": 30, "altura": 20, "x": 0, "y": 0, "rotacao": 0}
    # ]

    recortes_disponiveis = [
        {"tipo": "retangular", "largura": 29, "altura": 29, "x": 1, "y": 1, "rotacao": 0},
        {"tipo": "retangular", "largura": 29, "altura": 29, "x": 31, "y": 1, "rotacao": 0},
        {"tipo": "retangular", "largura": 29, "altura": 29, "x": 1, "y": 31, "rotacao": 0},
        {"tipo": "retangular", "largura": 29, "altura": 29, "x": 1, "y": 69, "rotacao": 0},
        {"tipo": "retangular", "largura": 139, "altura": 29, "x": 60, "y": 70, "rotacao": 0},
        {"tipo": "retangular", "largura": 60, "altura": 8, "x": 66, "y": 52, "rotacao": 0},
        {"tipo": "retangular", "largura": 44, "altura": 4, "x": 117, "y": 39, "rotacao": 0},
        {"tipo": "diamante", "largura": 29, "altura": 48, "x": 32, "y": 31, "rotacao": 0},
        {"tipo": "diamante", "largura": 29, "altura": 48, "x": 62, "y": 2, "rotacao": 0},
        {"tipo": "diamante", "largura": 29, "altura": 48, "x": 94, "y": 2, "rotacao": 0},
        {"tipo": "circular", "r": 16, "x": 124, "y": 2},
        {"tipo": "circular", "r": 16, "x": 158, "y": 2}
    ]

    # Instantiate and run Genetic Algorithm.
    ga_optimizer = GeneticAlgorithm(TAM_POP=100, recortes_disponiveis=recortes_disponiveis,
                                    sheet_width=sheet_width, sheet_height=sheet_height, numero_geracoes=300)
    print("Running Genetic Algorithm...")
    ga_optimized_layout = ga_optimizer.optimize_and_display()


    # Optionally, print out the results (optimized layouts)
   

    print("Genetic Algorithm Optimized Layout:")
    for item in ga_optimized_layout or []:
        print(item)


if __name__ == "__main__":
    main()