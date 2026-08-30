import math
import numpy as np
import torch

class GNode:
    def __init__(self, state):
        self.state = state
        self.children = {}
        self.N = 0 #count visits
        self.W = 0.0 #accumulate value from player-to-move perspective
        self.is_expanded = False
        self.legal = None
        self.logits = None #store policy logits for legal actions
        self.value = 0.0 #store value from player-to-move perspective
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0 #mean value

#evaluate network
def _evaluate(game, net, state):
    x = torch.tensor(game.encode(state)).unsqueeze(0) #encode board with batch dimension
    with torch.no_grad():
        logits, value = net(x)
    logits = logits[0].numpy() #remove batch dimension and convert to NumPy
    legal = game.legal_moves(state)
    return legal, logits[legal], float(value.item())

def _expand(game, net, node): #evaluate and initialize node
    legal, logits, value = _evaluate(game, net, node.state)
    node.legal = legal
    node.logits = logits
    node.value = value
    for a in legal:
        node.children[a] = GNode(game.apply(node.state, a)) #child state after playing move
    node.is_expanded = True #mark node expanded
    return value

#compute stable softmax
def _softmax(x):
    z = x - x.max() #shift values for numerical stability
    e = np.exp(z)
    return e / e.sum() #normalize probability distribution

def _sigma(q, max_visit, c_visit=50.0, c_scale=1.0):
    return (c_visit + max_visit) * c_scale * q #scale action values with search visits

def _completed_q(q_values, visit_counts, priors, node_value):
    total = visit_counts.sum()
    if total == 0:
        v_mix = node_value #fall back to network value
    else:
        visited = visit_counts > 0
        weighted_q = (priors[visited] * q_values[visited]).sum() / priors[visited].sum() #average visited values by prior
        v_mix = (node_value + total * weighted_q) / (1.0 + total) #blend network and searched values
    return np.where(visit_counts > 0, q_values, v_mix) #use mixed value for unvisited actions

def _improved_policy(logits, completed_q, max_visit): #form search-improved distribution
    return _softmax(logits + _sigma(completed_q, max_visit)) #combine policy logits and searched values

def _child_stats(node):
    legal = node.legal
    N = np.array([node.children[a].N for a in legal], dtype=np.float64) #visit counts
    Q = np.array([-node.children[a].Q if node.children[a].N > 0 else 0.0 for a in legal], dtype=np.float64) #from current player's perspective
    return N, Q

#select interior nodes deterministically
def _nonroot_select(node):
    legal = node.legal
    N, raw_q = _child_stats(node)
    priors = _softmax(node.logits)
    cq = _completed_q(raw_q, N, priors, node.value)
    max_visit = N.max() if N.size else 0.0 #handle empty visit array
    pi = _improved_policy(node.logits, cq, max_visit)
    total = N.sum()
    score = pi - N / (1.0 + total) #favor moves whose visit share trails policy
    return legal[int(np.argmax(score))]

#one simulation
def _simulate(game, net, root, first_action):
    path = [root]
    node = root.children[first_action] #forced first action
    path.append(node)
    while node.is_expanded and not game.is_terminal(node.state):
        a = _nonroot_select(node) #deterministic interior selection
        node = node.children[a]
        path.append(node)
    leaf = node
    if game.is_terminal(leaf.state):
        value = game.outcome(leaf.state) #use terminal result from leaf perspective
    else:
        value = _expand(game, net, leaf) #expand at most once per simulation
    for nd in reversed(path):
        nd.N += 1 #visited once more
        nd.W += value #accumulate value
        value = -value #invert perspective while backing up

def run_gumbel_mcts(game, net, root_state, n_sims=150, m_considered=16, add_noise=True, rng=None):
    root = GNode(root_state)
    _expand(game, net, root) #evaluate root
    legal = root.legal
    logits = root.logits
    n_legal = len(legal)

    if n_legal == 1: #if only one legal move
        return legal[0], {legal[0]: 1.0}

    if add_noise and rng is not None:
        u = rng.uniform(size=n_legal) #sample uniform noise source
        g = -np.log(-np.log(u + 1e-12) + 1e-12) #Gumbel(0,1) noise
    else:
        g = np.zeros(n_legal)

    m = min(m_considered, n_legal) #limit candidate actions
    candidates = list(np.argsort(-(g + logits))[:m]) #rank perturbed logits and keep top actions
    num_phases = max(1, int(math.ceil(math.log2(m)))) #sequential-halving phase count

    remaining = candidates
    while len(remaining) > 1:
        sims_per = max(1, n_sims // (num_phases * len(remaining))) #allocate simulations per candidate
        for idx in remaining:
            for _ in range(sims_per):
                _simulate(game, net, root, legal[idx]) #forced-root-action sim
        N, raw_q = _child_stats(root) #update stats
        max_visit = N.max() if N.size else 0.0
        score = {idx: g[idx] + logits[idx] + _sigma(raw_q[idx], max_visit) for idx in remaining} #rank candidates
        remaining = sorted(remaining, key=lambda i: score[i], reverse=True)[:max(1, len(remaining) // 2)] #keep top half

    survivor = legal[remaining[0]] #self-play move

    #build improved policy at root
    N, raw_q = _child_stats(root)
    priors = _softmax(logits)
    cq = _completed_q(raw_q, N, priors, root.value)
    max_visit = N.max() if N.size else 0.0
    pi = _improved_policy(logits, cq, max_visit)
    pi_dict = {legal[i]: float(pi[i]) for i in range(n_legal)} #map actions to probabilities

    return survivor, pi_dict
