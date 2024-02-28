import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from fcmeans import FCM
from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from k_means_constrained import KMeansConstrained

from problems.op.state_op import StateOP
from utils.beam_search import beam_search
from utils.data_utils import load_dataset


class OP(object):
    NAME = 'op'  # Orienteering problem

    @staticmethod
    def get_costs(dataset, pi, num_depots=1):

        # Index of end depot
        end_ids = 0 if num_depots == 1 else dataset['loc'].shape[1] + 1

        # Check if tour consists in going to the end depot
        if pi.size(-1) == 1:  # In case all tours directly return to depot, prevent further problems
            assert (pi == end_ids).all(), "If all length 1 tours, they should be equal to the end depot index"
            # Return
            return torch.zeros(pi.size(0), dtype=torch.float, device=pi.device), None

        # Check that tours are valid, i.e. contain 0 to n - 1
        sorted_pi = pi.data.sort(1)[0]

        # Make sure each node visited once at most (except for end depot)
        assert ((sorted_pi[:, 1:] == end_ids) | (sorted_pi[:, 1:] > sorted_pi[:, :-1])).all(), "Duplicates"

        prize_with_depot = torch.cat((torch.zeros_like(dataset['prize'][:, :1]), dataset['prize']), 1)
        if num_depots > 1:
            prize_with_depot = torch.cat((prize_with_depot, torch.zeros_like(dataset['prize'][:, :1])), 1)
        p = prize_with_depot.gather(1, pi)

        # Gather dataset in order of tour
        loc_with_depot = torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
        if num_depots > 1:
            loc_with_depot = torch.cat((loc_with_depot, dataset['depot2'][:, None, :]), 1)
        d = loc_with_depot.gather(1, pi[..., None].expand(*pi.size(), loc_with_depot.size(-1)))

        end_depot = 'depot' if num_depots == 1 else 'depot2'
        length = (
                (d[:, 1:] - d[:, :-1]).norm(p=2, dim=-1).sum(1)  # Prevent error if len 1 seq
                + (d[:, 0] - dataset['depot']).norm(p=2, dim=-1)  # Depot to first
                + (d[:, -1] - dataset[end_depot]).norm(p=2, dim=-1)  # Last to depot, will be 0 if depot is last
        )
        assert (length <= dataset['max_length'] + 1e-5).all(), \
            "Max length exceeded by {}".format((length - dataset['max_length']).max())

        # We want to maximize total prize but code minimizes so return negative
        if 'oracle' in dataset:
            return -10 * p.sum(-1) / dataset['oracle'], None
        else:
            return -p.sum(-1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return OPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateOP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = OP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)

def generate_instance(size, prize_type, num_agents=1, num_depots=1, max_length=2, cluster='kmc', test_coop=False):
    loc = torch.FloatTensor(size, 2).uniform_(0, 1)
    depot = torch.FloatTensor(2).uniform_(0, 1)

    # Initialize obstacle tensor
    obstacle = torch.zeros((size,))
    num_obstacles = int(0.2 * size)
    obstacles_indices = np.random.choice(size, size=num_obstacles, replace=False)
    for idx in obstacles_indices:
        obstacle[idx] = 1
    
    # Methods taken from Fischetti et al. 1998
    if prize_type == 'const':
        prize = torch.ones(size)
    elif prize_type == 'unif':
        prize = (1 + torch.randint(0, 100, size=(size,))) / 100.
    elif prize_type == 'coop' or prize_type == 'nocoop':

        # K-Means clustering
        if cluster == 'km':
            km = KMeans(n_clusters=num_agents)
            labels = km.fit_predict(loc)

        # K-Means constrained clustering
        elif cluster == 'kmc':
            kmc = KMeansConstrained(n_clusters=num_agents, size_min=size // num_agents,
                                    size_max=size // num_agents + 1, random_state=0, max_iter=1)
            labels = kmc.fit_predict(loc)

        # Fuzzy C-Means clustering
        else:
            assert cluster == 'fcm', 'Cluster methods are: K-Means(km), K-Means const (kmc), Fuzzy C-Means (fcm)'
            fcm = FCM(n_clusters=num_agents)
            fcm.fit(loc)
            labels = fcm.predict(loc)

        # Return data
        if test_coop:
            agents = {}
            for i in range(num_agents):
                prize = torch.ones(labels.shape)
                prize[labels != i] = 0.5 if prize_type == 'coop' else 0
                agents[i] = {'loc': loc, 'prize': prize, 'depot': depot, 'obstacle': obstacle,
                             'max_length': torch.tensor(max_length)}
            return agents
        else:
            prize = torch.ones(labels.shape)
            prize[labels != np.random.randint(low=0, high=num_agents, size=1)[0]] = .5 if prize_type == 'coop' else 0

    else:  # Based on distance to depot
        assert prize_type == 'dist'
        prize_ = (depot[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    # Output dataset
    dictionary = {'loc': loc, 'prize': prize, 'depot': depot, 'obstacle': obstacle, 'max_length': torch.tensor(max_length)}

    # End depot is different from start depot
    if num_depots == 2:
        dictionary['depot2'] = torch.FloatTensor(2).uniform_(0, 1)

    return dictionary


class OPDataset(Dataset):

    def __init__(self, filename=None, size=20, num_samples=1000000, offset=0, distribution='coop', test_coop=False,
                 num_agents=1, num_depots=1, max_length=2, **kwargs):
        super(OPDataset, self).__init__()
        assert distribution is not None, "Data distribution must be specified for OP"
        # Currently the distribution can only vary in the type of the prize
        prize_type = distribution

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl' or os.path.isdir(filename)

            print('Loading dataset...')
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            if num_depots == 1:
                self.data = [
                    {
                        'loc': torch.FloatTensor(loc),
                        'prize': torch.FloatTensor(prize),
                        'depot': torch.FloatTensor(depot),
                        'obstacle': torch.FloatTensor(obstacle),
                        'max_length': torch.tensor(max_length)
                    }
                    for depot, loc, prize, obstacle, max_length in tqdm(data[offset:offset + num_samples])
                ]
            else:
                assert num_depots == 2, 'Number of depots has to be either 1 or 2.'
                self.data = [
                    {
                        'loc': torch.FloatTensor(loc),
                        'prize': torch.FloatTensor(prize),
                        'depot': torch.FloatTensor(depot),
                        'max_length': torch.tensor(max_length),
                        'obstacle': torch.FloatTensor(obstacle),
                        'depot2': torch.FloatTensor(depot2)
                    }
                    for depot, loc, prize, obstacle, max_length, depot2 in tqdm(data[offset:offset + num_samples])
                ]
        else:
            print('Generating dataset...')
            self.data = [
                generate_instance(
                    size,
                    prize_type,
                    num_agents=num_agents,
                    num_depots=num_depots,
                    max_length=max_length,
                    test_coop=test_coop
                ) for _ in tqdm(range(num_samples))
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class OPDatasetLarge(Dataset):

    def __init__(self, filename=None, distribution='coop', num_depots=1, **kwargs):
        super(OPDatasetLarge, self).__init__()
        self.num_depots = num_depots
        assert distribution is not None, "Data distribution must be specified for OP"
        # Currently the distribution can only vary in the type of the prize

        assert os.path.splitext(filename)[1] == '.pkl' or os.path.isdir(filename)
        assert os.path.isdir(filename)
        self.filename = filename

        print('Loading dataset...')
        self.size = len(os.listdir(filename))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = load_dataset(os.path.join(self.filename, str(idx).zfill(9)))
        if self.num_depots == 1:
            return {
                        'loc': torch.FloatTensor(data[1]),
                        'prize': torch.FloatTensor(data[2]),
                        'depot': torch.FloatTensor(data[0]),
                        'max_length': torch.tensor(data[3]),
                        'obstacle': torch.FloatTensor(data[4])
                    }
        else:
            return {
                        'loc': torch.FloatTensor(data[1]),
                        'prize': torch.FloatTensor(data[2]),
                        'depot': torch.FloatTensor(data[0]),
                        'max_length': torch.tensor(data[3]),
                        'obstacle': torch.FloatTensor(data[4]),
                        'depot2': torch.FloatTensor(data[5])
                    }
