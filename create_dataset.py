import os
import math
import argparse
import numpy as np
import random
from utils.data_utils import save_dataset, set_seed
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from fcmeans import FCM
from tqdm import tqdm


def generate_tsp_data(dataset_size, tsp_size):
    return np.random.uniform(size=(dataset_size, tsp_size, 2)).tolist()


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_op_data(dataset_size, op_size, max_length, prize_type='const', num_agents=1, num_depots=1, cluster='kmc'):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'unif':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    elif prize_type == 'coop' or prize_type == 'nocoop':

        # K-Means clustering
        if cluster == 'km':
            km = KMeans(n_clusters=num_agents)
            labels = np.zeros((dataset_size, op_size))
            for i in tqdm(range(dataset_size)):
                labels[i] = km.fit_predict(loc[i])

        # K-Means constrained clustering
        elif cluster == 'kmc':
            kmc = KMeansConstrained(n_clusters=num_agents, size_min=op_size // num_agents,
                                    size_max=op_size // num_agents + 1, random_state=0, max_iter=1)
            labels = np.zeros((dataset_size, op_size))
            for i in tqdm(range(dataset_size)):
                labels[i] = kmc.fit_predict(loc[i])

        # Fuzzy C-Means clustering
        else:
            assert cluster == 'fcm', 'Cluster method not listed: K-Means(km), K-Means const (kmc), Fuzzy C-Means (fcm)'
            labels = np.zeros((dataset_size, op_size))
            for i in tqdm(range(dataset_size)):
                fcm = FCM(n_clusters=num_agents)
                fcm.fit(loc[i])
                labels[i] = fcm.predict(loc[i])

        prize = np.ones(labels.shape)
        prize[labels != np.random.randint(low=0, high=num_agents, size=1)[0]] = 0.5 if prize_type == 'coop' else 0
    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.

    # Calculate 20% of the length of the list
    twenty_percent = int(len(prize) * 0.2)

    # Randomly select 10% of the elements
    indices_to_negate = random.sample(range(len(prize)), twenty_percent)

    # Negate the selected elements
    for index in indices_to_negate:
        prize[index] = -100

    # End depot is different from start depot
    if num_depots == 2:
        depot2 = np.random.uniform(size=(dataset_size, 2))
        return list(zip(
            depot.tolist(),
            loc.tolist(),
            prize.tolist(),
            np.full(dataset_size, max_length).tolist(),  # Capacity, same for whole dataset
            depot2.tolist()
        ))

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        prize.tolist(),
        np.full(dataset_size, max_length).tolist()  # Capacity, same for whole dataset
    ))


def generate_pctsp_data(dataset_size, pctsp_size, penalty_factor=3):
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, pctsp_size, 2))

    # For the penalty to make sense it should be not too large (in which case all nodes will be visited) nor too small
    # so we want the objective term to be approximately equal to the length of the tour, which we estimate with half
    # of the nodes by half of the tour length (which is very rough but similar to op)
    # This means that the sum of penalties for all nodes will be approximately equal to the tour length (on average)
    # The expected total (uniform) penalty of half of the nodes (since approx half will be visited by the constraint)
    # is (n / 2) / 2 = n / 4 so divide by this means multiply by 4 / n,
    # However instead of 4 we use penalty_factor (3 works well) so we can make them larger or smaller
    MAX_LENGTHS = {
        20: 2.,
        50: 3.,
        100: 4.
    }
    penalty_max = MAX_LENGTHS[pctsp_size] * (penalty_factor) / float(pctsp_size)
    penalty = np.random.uniform(size=(dataset_size, pctsp_size)) * penalty_max

    # Take uniform prizes
    # Now expectation is 0.5 so expected total prize is n / 2, we want to force to visit approximately half of the nodes
    # so the constraint will be that total prize >= (n / 2) / 2 = n / 4
    # equivalently, we divide all prizes by n / 4 and the total prize should be >= 1
    deterministic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * 4 / float(pctsp_size)

    # In the deterministic setting, the stochastic_prize is not used and the deterministic prize is known
    # In the stochastic setting, the deterministic prize is the expected prize and is known up front but the
    # stochastic prize is only revealed once the node is visited
    # Stochastic prize is between (0, 2 * expected_prize) such that E(stochastic prize) = E(deterministic_prize)
    stochastic_prize = np.random.uniform(size=(dataset_size, pctsp_size)) * deterministic_prize * 2

    return list(zip(
        depot.tolist(),
        loc.tolist(),
        penalty.tolist(),
        deterministic_prize.tolist(),
        stochastic_prize.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset (test, validation...)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument('--problem', default='op', help="The problem to solve. Options: op, tsp, pctsp, vrp, top")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument('--data_dist', type=str, default='all',
                        help="Distributions to generate for problem: const, dist, unif or all.")
    parser.add_argument('--cluster', type=str, nargs='+', default=['km', 'kmc', 'fcm'], help="For OP with coop/nocoop"
                        "data dist, choose cluster method: K-Means(km), K-Means constrained(kmc), Fuzzy C-Means(fcm)")
    parser.add_argument('--max_length', type=float, nargs='+', default=[2, 3, 4],
                        help="Normalized time limit to solve the problem")
    parser.add_argument('--num_agents', type=int, nargs='+', default=[1], help="Number of agents (only for OP-MP-TN)")
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")
    opts = parser.parse_args()
    set_seed(opts.seed)
    assert opts.problem == 'op' and len(opts.graph_sizes) == len(opts.max_length),\
        "Number of graph sizes and max lengths must be the same for the OP problem"

    distributions = ['const', 'unif', 'dist', 'coop', 'nocoop'] if opts.data_dist == 'all' else [opts.data_dist]
    for distribution in distributions:
        print('Distribution = {}'.format(distribution))
        for num_agents in opts.num_agents:
            print('Number of agents = {}'.format(num_agents))
            for cluster in opts.cluster:
                print('Cluster method = {}'.format(cluster))
                for i, graph_size in enumerate(opts.graph_sizes):
                    print('Graph size = {}'.format(graph_size))

                    # Directory and filename
                    if opts.problem == 'op':
                        data_dir = os.path.join(
                            opts.data_dir,
                            opts.problem,
                            str(opts.num_depots) + 'depots',
                            str(num_agents) + 'agents',
                            distribution,
                            cluster,
                            str(graph_size)
                        )
                    else:
                        data_dir = os.path.join(
                            opts.data_dir,
                            opts.problem,
                            str(graph_size)
                        )
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir, exist_ok=True)
                    length_str = int(opts.max_length[i]) if opts.max_length[i].is_integer() else opts.max_length[i]
                    filename = os.path.join(data_dir, "{}_seed{}_L{}.pkl".format(opts.name, opts.seed, length_str))

                    # Generate data
                    if opts.problem == 'op':
                        dataset = generate_op_data(opts.dataset_size, graph_size, prize_type=distribution,
                                                   max_length=opts.max_length[i], num_depots=opts.num_depots,
                                                   num_agents=num_agents, cluster=cluster)
                    elif opts.problem == 'tsp':
                        dataset = generate_tsp_data(opts.dataset_size, graph_size)
                    elif opts.problem == 'pctsp':
                        dataset = generate_pctsp_data(opts.dataset_size, graph_size)
                    else:
                        assert opts.problem == 'vrp', 'Problem not in list: [op, tsp, vrp, pctsp]'
                        dataset = generate_vrp_data(opts.dataset_size, graph_size)

                    # Save dataset
                    if graph_size > 100 and opts.dataset_size > 1e6:
                        step, count, c = 1, 0, 0
                        while count < opts.dataset_size:
                            save_dataset(
                                dataset[int(count):int(min(count + step, len(dataset)))],
                                os.path.join(filename.replace('.pkl', ''), str(c).zfill(9))
                            )
                            count += step
                            c += 1
                    else:
                        save_dataset(dataset, filename)
                    del dataset
            if not opts.problem == 'op':
                break
        if not opts.problem == 'op':
            break
    print('Finished')
