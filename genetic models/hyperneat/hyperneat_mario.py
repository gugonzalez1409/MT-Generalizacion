import neat
import numpy as np
from pureples.hyperneat.hyperneat import create_phenotype_network


def create_ff_network_mario(cppn, substrate):
    
    input_coords = substrate.input_coordinates
    output_coords = substrate.output_coordinates
    
    input_nodes = list(range(len(input_coords)))
    output_nodes = list(range(len(input_coords), len(input_coords) + len(output_coords)))
    
    node_evals = []
    
    # Usar una función de activación estándar para la red fenotipo
    activation_functions = neat.activations.ActivationFunctionSet()
    activation_func = activation_functions.get('sigmoid')

    for i, out_coord in enumerate(output_coords):
        links = []
        for j, in_coord in enumerate(input_coords):
            # Input para la CPPN: (x_in, y_in, x_out, y_out, bias)
            cppn_input = [in_coord[0], in_coord[1], out_coord[0], out_coord[1], 1.0]
            weight = cppn.activate(cppn_input)[0]
            
            # Solo añadir la conexión si el peso es significativo
            if abs(weight) > 0.2:
                links.append((input_nodes[j], weight * 5.0)) # Escalamos el peso

        # Añadir la evaluación del nodo de salida
        # (node_id, activation_func, aggregation_func, bias, response, links)
        node_evals.append((output_nodes[i], activation_func, sum, 0.0, 1.0, links))

    return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)


def ini_pop(state, stats, config, output):

    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop

def select_action_from_paper(outputs, agent_pos, input_shape):

    height, width = input_shape

    agent_y = min(max(agent_pos[0], 0), height - 1)
    agent_x = min(max(agent_pos[1], 0), width - 1)

    agent_idx = agent_y * width + agent_x
    
    activations = { 0: outputs[agent_idx] }

    if agent_x + 1 < width:
        right_idx = agent_y * width + (agent_x + 1)
        activations[1] = outputs[right_idx] # Acción 1: right

    # Evaluar vecino izquierdo
    if agent_x - 1 >= 0:
        left_idx = agent_y * width + (agent_x - 1)
        activations[6] = outputs[left_idx] # Acción 6: left

    # Evaluar vecino de arriba (mapeado a 'A' para saltar)
    if agent_y - 1 >= 0:
        up_idx = (agent_y - 1) * width + agent_x
        activations[5] = outputs[up_idx] # Acción 5: A (salto)
    
    # La acción con la activación más alta es la que se elige.
    best_action_index = max(activations, key=activations.get)

    return best_action_index

def run_hyper_mario(gens, env, max_steps, config, substrate, activations, input_shape, max_trials=0, output=True):

    trials = 1

    def eval_fitness(genomes, config):
        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            net = create_ff_network_mario(cppn, substrate)

            fitnesses = []
            for _ in range(trials):
                ob = env.reset()
                net.reset()
                total_reward = 0

                for step in range(max_steps):
                    env.render()
                    agent_pos_arr = np.where(ob == 2)
                    if len(agent_pos_arr[0]) > 0:
                        agent_pos = (agent_pos_arr[0][0], agent_pos_arr[1][0])
                    else:
                        agent_pos = (input_shape[0] - 1, input_shape[1] // 2)

                    # Aplanar la observación para la red
                    ob_flat = ob.flatten()
                    
                    o = net.activate(ob_flat)

                    # Seleccionar acción usando la lógica del paper
                    action = select_action_from_paper(o, agent_pos, input_shape)

                    ob, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)
            g.fitness = np.array(fitnesses).mean()

    # El resto de la función para gestionar la población y las estadísticas no cambia.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)