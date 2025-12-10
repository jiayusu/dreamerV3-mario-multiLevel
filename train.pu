# train.py
"""

"""

import math, random, time, copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_super_mario_bros
from gym.wrappers import ResizeObservation, GrayScaleObservation, FrameStack
from nes_py.wrappers import JoypadSpace
import multiprocessing as mp
import wandb
import os

# ---------------------------
# Configuration
# ---------------------------
LEVELS = ["1-1", "1-2"]
LEVEL_TO_ID = {level: i for i, level in enumerate(LEVELS)}
N_LEVELS = len(LEVELS)
LEVEL_EMB_DIM = 8

CUSTOM_MOVEMENT = [
    ['right'],
    ['right', 'A'],
    ['A'],
    ['right', 'B'],
    ['noop']
]
N_ACTIONS = len(CUSTOM_MOVEMENT)

TOTAL_STEPS = 500_000
NUM_COLLECTORS = 4
COLLECT_STEPS = 1000
BATCH_SIZE = 32
SEQ_LEN = 50
LATENT_DIM = 32
RNN_HIDDEN = 256
LR_WM = 3e-4
LR_AC = 5e-4
GAMMA = 0.995
LAMBDA = 0.95
ENT_COEF = 0.01
PPO_EPS = 0.2
KL_BALANCE = 0.8
KL_FREE = 0.5
KL_SCALE = 1.0
REWARD_SCALE = 0.1
MCTS_SIMS = 30
MCTS_HORIZON = 5
C_PUCT = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Env & Utils
# ---------------------------
def make_env(level="1-1"):
    env_name = f"SuperMarioBros-{level}-v0"
    env = gym_super_mario_bros.make(env_name)
    env = JoypadSpace(env, CUSTOM_MOVEMENT)
    env = ResizeObservation(env, (64, 64))
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, 4)
    return env

def obs_to_tensor(obs, device=DEVICE):
    x = np.array(obs).astype(np.float32) / 255.0
    return torch.from_numpy(x).unsqueeze(0).to(device)

def scale_reward(r):
    return np.sign(r) * np.log(1 + abs(r)) * REWARD_SCALE

# ---------------------------
# Networks
# ---------------------------
class Encoder(nn.Module):
    def __init__():
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(256 * 2 * 2, 256)

    def forward(self, x):
        return self.fc(self.conv(x))

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(RNN_HIDDEN + LATENT_DIM, 256 * 2 * 2)
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, h, z):
        x = torch.cat([h, z], -1)
        x = self.fc(x).view(-1, 256, 2, 2)
        return self.conv(x)

class RSSM(nn.Module):
    def __init__(self, action_dim, hidden_dim=RNN_HIDDEN, latent_dim=LATENT_DIM, action_emb_dim=32):
        super().__init__()
        self.rnn = nn.GRUCell(action_emb_dim + latent_dim, hidden_dim)  # ✅ FIXED DIMENSION
        self.action_emb = nn.Embedding(action_dim, action_emb_dim)
        self.fc_prior = nn.Linear(hidden_dim, 2 * latent_dim)
        self.fc_posterior = nn.Linear(hidden_dim + 256, 2 * latent_dim)

    def prior(self, h):
        stats = self.fc_prior(h)
        return self._gaussian(stats)

    def posterior(self, h, embed):
        stats = self.fc_posterior(torch.cat([h, embed], -1))
        return self._gaussian(stats)

    def _gaussian(self, stats):
        mean, std = torch.chunk(stats, 2, -1)
        std = F.softplus(std) + 0.1
        return mean, std

    def imagine(self, actions, h, z):
        T, B = actions.shape
        h_list, z_list = [], []
        for t in range(T):
            # ✅ FIXED: ensure long dtype and correct device
            a_emb = self.action_emb(actions[t].long().to(h.device))
            rnn_input = torch.cat([a_emb, z], -1)
            h = self.rnn(rnn_input, h)
            prior_mean, prior_std = self.prior(h)
            z = prior_mean + prior_std * torch.randn_like(prior_std)
            h_list.append(h)
            z_list.append(z)
        return torch.stack(h_list), torch.stack(z_list)

class WorldModel(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.encoder = Encoder()
        self.rssm = RSSM(action_dim)
        self.decoder = Decoder()
        self.reward_model = nn.Sequential(
            nn.Linear(RNN_HIDDEN + LATENT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, actions):
        T, B = actions.shape
        embeds = torch.stack([self.encoder(obs[t]) for t in range(T)])
        h = torch.zeros(B, RNN_HIDDEN, device=obs.device)
        z = torch.zeros(B, LATENT_DIM, device=obs.device)
        h_list, z_list = [], []
        prior_means, prior_stds = [], []
        post_means, post_stds = [], []

        for t in range(T):
            a_emb = self.rssm.action_emb(actions[t].long().to(h.device))  # ✅ FIXED
            rnn_input = torch.cat([a_emb, z], -1)
            h = self.rssm.rnn(rnn_input, h)
            prior_mean, prior_std = self.rssm.prior(h)
            embed = embeds[t]
            post_mean, post_std = self.rssm.posterior(h, embed)
            z = post_mean + post_std * torch.randn_like(post_std)
            h_list.append(h)
            z_list.append(z)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(post_mean)
            post_stds.append(post_std)

        h_seq = torch.stack(h_list)
        z_seq = torch.stack(z_list)
        prior = (torch.stack(prior_means), torch.stack(prior_stds))
        post = (torch.stack(post_means), torch.stack(post_stds))
        recons = torch.stack([self.decoder(h_seq[t], z_seq[t]) for t in range(T)])
        rewards = self.reward_model(torch.cat([h_seq, z_seq], -1))
        return recons, rewards, prior, post

    def encode_step(self, obs, action, h, z):
        embed = self.encoder(obs)
        # ✅ FIXED: ensure long dtype and correct device
        a_emb = self.rssm.action_emb(action.long().to(h.device))
        rnn_input = torch.cat([a_emb, z], -1)
        h = self.rnn(rnn_input, h)
        post_mean, post_std = self.rssm.posterior(h, embed)
        z = post_mean + post_std * torch.randn_like(post_std)
        return h, z

class Actor(nn.Module):
    def __init__(self, action_dim, n_levels=N_LEVELS, level_emb_dim=LEVEL_EMB_DIM):
        super().__init__()
        self.level_emb = nn.Embedding(n_levels, level_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(RNN_HIDDEN + LATENT_DIM + level_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, h, z, level_id):
        emb = self.level_emb(level_id)
        x = torch.cat([h, z, emb], -1)
        logits = self.fc(x)
        return F.softmax(logits, dim=-1), logits

class Critic(nn.Module):
    def __init__(self, n_levels=N_LEVELS, level_emb_dim=LEVEL_EMB_DIM):
        super().__init__()
        self.level_emb = nn.Embedding(n_levels, level_emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(RNN_HIDDEN + LATENT_DIM + level_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, h, z, level_id):
        emb = self.level_emb(level_id)
        x = torch.cat([h, z, emb], -1)
        return self.fc(x).squeeze(-1)

# ---------------------------
# MCTS with Level ID
# ---------------------------
class MCTSNode:
    __slots__ = ['h', 'z', 'prior', 'children', 'N', 'W', 'Q', 'reward']
    def __init__(self, h, z, prior):
        self.h = h
        self.z = z
        self.prior = prior
        self.children = {}
        self.N = 0.0
        self.W = 0.0
        self.Q = 0.0
        self.reward = 0.0

class MCTS:
    def __init__(self, wm, actor, n_actions, sims, horizon, c_puct, device, level_id):
        self.wm = wm
        self.actor = actor
        self.n_actions = n_actions
        self.sims = sims
        self.horizon = horizon
        self.c_puct = c_puct
        self.device = device
        self.level_id = level_id

    def run(self, h, z):
        level_id_tensor = torch.tensor([self.level_id], device=self.device)
        with torch.no_grad():
            _, root_logits = self.actor(h, z, level_id_tensor)
            root_probs = F.softmax(root_logits, dim=-1).cpu().numpy().ravel()

        root = MCTSNode(h, z, prior=1.0)
        root.children = {a: MCTSNode(None, None, float(root_probs[a])) for a in range(self.n_actions)}

        for _ in range(self.sims):
            path = []
            node = root
            depth = 0
            while depth < self.horizon:
                unexpanded = [a for a, child in node.children.items() if child.h is None]
                if unexpanded:
                    action = random.choice(unexpanded)
                    a_tensor = torch.tensor([action], device=self.device)
                    next_h, next_z = self.wm.rssm.imagine(a_tensor.unsqueeze(0), node.h, node.z)
                    reward = self.wm.reward_model(torch.cat([next_h[0], next_z[0]], -1)).item()
                    child = node.children[action]
                    child.h = next_h[0]
                    child.z = next_z[0]
                    child.reward = reward
                    _, logits_c = self.actor(next_h[0], next_z[0], level_id_tensor)
                    probs_c = F.softmax(logits_c, dim=-1).cpu().numpy().ravel()
                    child.children = {a2: MCTSNode(None, None, float(probs_c[a2])) for a2 in range(self.n_actions)}
                    path.append(child)
                    break
                else:
                    best_score = -1e9
                    best_action = None
                    for a, child in node.children.items():
                        u = self.c_puct * child.prior * math.sqrt(node.N + 1e-8) / (1 + child.N)
                        score = child.Q + u
                        if score > best_score:
                            best_score = score
                            best_action = a
                    node = node.children[best_action]
                    path.append(node)
                    depth += 1

            if path:
                leaf = path[-1]
                value = self.wm.reward_model(torch.cat([leaf.h, leaf.z], -1)).item()
                for n in reversed(path):
                    value = n.reward + GAMMA * value
                    n.N += 1
                    n.W += value
                    n.Q = n.W / (n.N + 1e-8)

        visits = np.array([root.children[a].N for a in range(self.n_actions)], dtype=np.float32)
        pi = visits / visits.sum() if visits.sum() > 0 else np.ones(self.n_actions) / self.n_actions
        root_value = sum(child.W for child in root.children.values()) / max(sum(child.N for child in root.children.values()), 1)
        return pi, root_value

# ---------------------------
# Sequence PER
# ---------------------------
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self): return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

class SequencePER:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = 0.4
        self.epsilon = 1e-6

    def _get_priority(self, error): return (abs(error) + self.epsilon) ** self.alpha

    def add(self, error, data): self.tree.add(self._get_priority(error), data)

    def sample(self, n, seq_len):
        batch, idxs, weights = [], [], []
        segment = self.tree.total() / n
        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            idx, p, episode = self.tree.get(random.uniform(a, b))
            if len(episode) <= seq_len:
                seq = episode
            else:
                start = random.randint(0, len(episode) - seq_len)
                seq = episode[start:start + seq_len]
            batch.append(seq)
            idxs.append(idx)
            weights.append(p)
        prob = np.array(weights) / self.tree.total()
        isw = np.power(self.tree.n_entries * prob, -self.beta)
        isw /= isw.max()
        return batch, idxs, isw

    def update(self, idx, error): self.tree.update(idx, self._get_priority(error))

    def __len__(self): return self.tree.n_entries

# ---------------------------
# GAE
# ---------------------------
def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = 0
        else:
            next_val = values[t + 1] * (1 - dones[t])
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

# ---------------------------
# Collector with Level ID (FULLY FIXED)
# ---------------------------
def collector_process(wm_state, actor_state, steps_queue, data_queue, process_id):
    wm = WorldModel(N_ACTIONS)
    actor = Actor(N_ACTIONS, n_levels=N_LEVELS, level_emb_dim=LEVEL_EMB_DIM)  # ✅ Explicit
    wm.load_state_dict(wm_state)
    actor.load_state_dict(actor_state)
    wm.eval()
    actor.eval()
    wm.to("cpu")
    actor.to("cpu")

    steps = 0
    while steps < COLLECT_STEPS:
        level = random.choice(LEVELS)
        level_id = LEVEL_TO_ID[level]
        env = make_env(level)
        obs = env.reset()
        # ✅ Explicit CPU tensors
        h = torch.zeros(1, RNN_HIDDEN, device="cpu")
        z = torch.zeros(1, LATENT_DIM, device="cpu")
        episode = []
        done = False

        # ✅ Inner loop checks steps to prevent deadlock
        while not done and steps < COLLECT_STEPS:
            obs_t = obs_to_tensor(obs, device="cpu")
            with torch.no_grad():
                level_id_tensor = torch.tensor([level_id], device="cpu")
                probs, _ = actor(h, z, level_id_tensor)
                action = torch.multinomial(probs, 1).item()
                # ✅ Explicit CPU action tensor
                action_tensor = torch.tensor([action], dtype=torch.long, device="cpu")
                next_h, next_z = wm.encode_step(obs_t, action_tensor, h, z)
            next_obs, r, done, _ = env.step(action)
            r = scale_reward(r)
            episode.append((obs, action, r, done, h.numpy(), z.numpy(), level_id))
            obs = next_obs
            h, z = next_h, next_z
            steps += 1
            if done:
                data_queue.put(episode)
                episode = []
                break
        env.close()

    steps_queue.put(steps)

# ---------------------------
# KL Loss
# ---------------------------
def kl_balance_loss(prior, post, alpha=KL_BALANCE):
    prior_mean, prior_std = prior
    post_mean, post_std = post
    kl = torch.distributions.kl_divergence(
        torch.distributions.Normal(post_mean, post_std),
        torch.distributions.Normal(prior_mean, prior_std)
    ).sum(-1)
    kl = torch.maximum(kl, torch.tensor(KL_FREE, device=kl.device))
    return KL_SCALE * (alpha * kl.detach() + (1 - alpha) * kl)

# ---------------------------
# Training with W&B
# ---------------------------
def train():
    wandb.init(
        project="muzero-mario-level",
        config={
            "levels": LEVELS,
            "total_steps": TOTAL_STEPS,
            "n_actions": N_ACTIONS,
            "latent_dim": LATENT_DIM,
            "rnn_hidden": RNN_HIDDEN,
            "mcts_sims": MCTS_SIMS,
        }
    )

    # ✅ Explicit initialization
    wm = WorldModel(N_ACTIONS).to(DEVICE)
    actor = Actor(N_ACTIONS, n_levels=N_LEVELS, level_emb_dim=LEVEL_EMB_DIM).to(DEVICE)
    critic = Critic(n_levels=N_LEVELS, level_emb_dim=LEVEL_EMB_DIM).to(DEVICE)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=LR_WM)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=LR_AC)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=LR_AC)

    buffer = SequencePER(capacity=1000)
    total_steps = 0

    steps_queue = mp.Queue()
    data_queue = mp.Queue()
    processes = []
    for i in range(NUM_COLLECTORS):
        p = mp.Process(target=collector_process, args=(
            wm.state_dict(), actor.state_dict(), steps_queue, data_queue, i))
        p.start()
        processes.append(p)

    print("🚀 Training started...")
    last_log = 0
    level_returns = {level: deque(maxlen=10) for level in LEVELS}

    while total_steps < TOTAL_STEPS:
        while not data_queue.empty():
            episode = data_queue.get()
            if not episode:
                continue
            level_id = episode[0][6]
            level = LEVELS[level_id]
            total_r = sum(t[2] for t in episode)
            level_returns[level].append(total_r)
            priority = abs(total_r) + 1.0
            buffer.add(priority, episode)
        while not steps_queue.empty():
            total_steps += steps_queue.get()

        if len(buffer) < BATCH_SIZE:
            time.sleep(0.1)
            continue

        seq_batch, idxs, isw = buffer.sample(BATCH_SIZE, SEQ_LEN)
        isw = torch.tensor(isw, dtype=torch.float32, device=DEVICE)

        obs_seq, action_seq, reward_seq, level_ids_seq = [], [], [], []
        for seq in seq_batch:
            obs_seq.append(torch.stack([obs_to_tensor(t[0], DEVICE) for t in seq]))
            action_seq.append(torch.tensor([t[1] for t in seq], device=DEVICE))
            reward_seq.append(torch.tensor([t[2] for t in seq], device=DEVICE))
            level_ids_seq.append(torch.tensor([t[6] for t in seq], device=DEVICE))
        obs_seq = torch.stack(obs_seq).transpose(0, 1)
        action_seq = torch.stack(action_seq).transpose(0, 1)
        reward_seq = torch.stack(reward_seq).transpose(0, 1)
        level_ids_seq = torch.stack(level_ids_seq).transpose(0, 1)

        recons, rewards_pred, prior, post = wm(obs_seq, action_seq)
        loss_recon = F.mse_loss(recons, obs_seq)
        loss_reward = F.mse_loss(rewards_pred, reward_seq)
        loss_kl = kl_balance_loss(prior, post).mean()
        wm_loss = loss_recon + loss_reward + loss_kl

        opt_wm.zero_grad()
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(), 10.0)
        opt_wm.step()

        with torch.no_grad():
            _, _, _, post = wm(obs_seq, action_seq)
            post_mean, post_std = post
            z_seq = post_mean + post_std * torch.randn_like(post_std)
            h_seq = torch.zeros(SEQ_LEN, BATCH_SIZE, RNN_HIDDEN, device=DEVICE)
            h = torch.zeros(BATCH_SIZE, RNN_HIDDEN, device=DEVICE)
            for t in range(SEQ_LEN):
                a_emb = wm.rssm.action_emb(action_seq[t].long())  # ✅ Safe
                h = wm.rssm.rnn(torch.cat([a_emb, z_seq[t]], -1), h)
                h_seq[t] = h

        h_flat = h_seq.view(-1, RNN_HIDDEN)
        z_flat = z_seq.view(-1, LATENT_DIM)
        actions_flat = action_seq.view(-1)
        rewards_flat = reward_seq.view(-1).cpu().numpy()
        dones_flat = np.zeros_like(rewards_flat)
        level_ids_flat = level_ids_seq.view(-1)

        values = critic(h_flat, z_flat, level_ids_flat).cpu().numpy()
        probs, logits = actor(h_flat, z_flat, level_ids_flat)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_flat).cpu().numpy()

        advantages_list, returns_list = [], []
        for b in range(BATCH_SIZE):
            start = b * SEQ_LEN
            end = start + SEQ_LEN
            adv, ret = compute_gae(rewards_flat[start:end], values[start:end], dones_flat[start:end])
            advantages_list.append(adv)
            returns_list.append(ret)
        advantages = np.concatenate(advantages_list)
        returns = np.concatenate(returns_list)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
        log_probs = torch.tensor(log_probs, dtype=torch.float32, device=DEVICE)

        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - PPO_EPS, 1 + PPO_EPS) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        entropy = dist.entropy().mean()
        actor_loss -= ENT_COEF * entropy

        critic_loss = F.mse_loss(critic(h_flat, z_flat, level_ids_flat), returns)

        distill_loss = 0
        for b in range(min(2, BATCH_SIZE)):
            h0 = h_seq[0, b:b+1]
            z0 = z_seq[0, b:b+1]
            level0 = level_ids_seq[0, b].item()
            mcts = MCTS(wm, actor, N_ACTIONS, MCTS_SIMS, MCTS_HORIZON, C_PUCT, DEVICE, level0)
            pi_mcts, _ = mcts.run(h0, z0)
            pi_mcts = torch.tensor(pi_mcts, dtype=torch.float32, device=DEVICE)
            _, logits0 = actor(h0, z0, torch.tensor([level0], device=DEVICE))
            pi_actor = F.softmax(logits0, dim=-1).squeeze(0)
            distill_loss += F.kl_div(pi_actor.log(), pi_mcts, reduction='batchmean')
        distill_loss /= min(2, BATCH_SIZE)
        actor_loss += 0.1 * distill_loss

        opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        opt_actor.step()

        opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        opt_critic.step()

        with torch.no_grad():
            new_priorities = (critic_loss.item() + 1e-6) * np.ones(len(idxs))
        for i, idx in enumerate(idxs):
            buffer.update(idx, new_priorities[i])

        if total_steps - last_log > 5000:
            log_dict = {
                "step": total_steps,
                "wm_loss": wm_loss.item(),
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "distill_loss": distill_loss.item(),
                "buffer_size": len(buffer),
            }
            for level in LEVELS:
                if level_returns[level]:
                    log_dict[f"return/{level}"] = np.mean(level_returns[level])
            wandb.log(log_dict)
            last_log = total_steps

    # Unified save
    torch.save({
        "wm": wm.state_dict(),
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "config": {
            "levels": LEVELS,
            "n_actions": N_ACTIONS,
            "latent_dim": LATENT_DIM,
            "rnn_hidden": RNN_HIDDEN,
            "level_emb_dim": LEVEL_EMB_DIM
        }
    }, "dreamerdreamer_mario_final.pt")

    for p in processes:
        p.terminate()
        p.join()
    wandb.finish()
    print("✅ Training finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    train()
