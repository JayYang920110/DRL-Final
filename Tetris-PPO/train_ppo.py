import argparse
import os
import random
from distutils.util import strtobool
import time
#tools
from torch.utils.tensorboard import SummaryWriter
# torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import numpy as np
#PPO env
from src.tetris import Tetris
import os
os.environ["MUJOCO_GL"] = "egl"


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp-name', type= str, default=os.path.basename(__file__).rstrip(".py"),
            help="the name of this experiment")
    parser.add_argument('--learning-rate', type= float, default=1e-3,
            help="lr of the optimizer")  
    parser.add_argument('--seed', type= int, default=1,
            help="seed of rand num")        
    parser.add_argument('--total-timesteps', type= int, default=1000000,
            help="total time steps of the experiment")  
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")    
    # GPU args
    parser.add_argument('--torch-deterministic', type= lambda x: bool(strtobool(x)), default=True,
            nargs='?', const=True, help="if toggled, torch.backend.cudnn.deterministic=False")      
    parser.add_argument('--cuda', type= lambda x: bool(strtobool(x)), default=True,
            nargs='?', const=True, help="if toggled, cuda will not be enabled by default") 
    
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--from-pretrained", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the model will be loaded from a pretrained model")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

#global variables
args=parse_args()

# env wrapper
def make_env(width,height,block_size):
    def thunk():
        env = Tetris(width=width, height=height, block_size=block_size)
        return env
    return thunk

#Agent setting
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            # print(f"weight:{weight}, bias:{bias}")
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            NoisyLinear(200, 512),
            nn.Tanh(),
            NoisyLinear(512, 256),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            NoisyLinear(200, 512),
            nn.Tanh(),
            NoisyLinear(512, 256),
            nn.Tanh(),
            layer_init(nn.Linear(256, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 34)),
        )

    def get_value(self, x):
        x=x.view(-1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        B=x.size()[0]
        # print(B)
        x_flat = x.view(B,-1)         # [B, 200]
        logits = self.actor(x_flat)     # [B, 34]
        
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # print(f"action:{action}")
        return action, probs.log_prob(action), probs.entropy(), self.critic(x_flat)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


#PPO　Agent
class ppo_buffer():
    def __init__(self, device):
        self.obs = torch.zeros((args.num_steps,) + (20,10)).to(device)
        self.actions = torch.zeros((args.num_steps,) + (1,)).to(device)
        self.logprobs = torch.zeros((args.num_steps, ),dtype=torch.float32).to(device)
        self.rewards = torch.zeros((args.num_steps, ),dtype=torch.float32).to(device)
        self.dones = torch.zeros((args.num_steps, )).to(device)
        self.values = torch.zeros((args.num_steps, ),dtype=torch.float32).to(device)


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.mean = torch.zeros(shape).to(device)
        self.var = torch.ones(shape).to(device)
        self.count = epsilon

    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False) 
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = M2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self):
        return torch.sqrt(self.var + 1e-8)  # 加 epsilon 避免除零



class PPO_agent(ppo_buffer):   
    def __init__(self,env,device,lr,writer):
        super(PPO_agent, self).__init__(device)
        self.model=Agent().to(device)
        self.writer=writer
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr,eps=1e-5)
        self.lr=lr
        self.lr_frac= 1.0
        # print(f"debug:{type(envs.reset())}")
        self.next_obs = env.reset()
        self.next_done = torch.zeros(args.num_envs).to(device)
        #background setting
            #static variables
        self.global_step = 0
        self.start_time = time.time()
        self.num_updates = args.total_timesteps // args.batch_size
        self.checkpoint=0
        self.rms = RunningMeanStd(shape=(1,))  
        
    def GAE(self):
        with torch.no_grad():
            # mean_reward = self.rewards.mean()
            # std_reward = self.rewards.std() + 1e-8  # 避免除以0
            # self.rewards = (self.rewards - mean_reward) / std_reward
            next_value = self.model.get_value(self.next_obs).reshape(1, -1)
            if args.gae: # GAE
                advantages = torch.zeros_like(self.rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    if self.dones[t]:
                        lastgaelam = 0
                    delta = self.rewards[t] + args.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else: # normal advantage function
                returns = torch.zeros_like(self.rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - self.next_done.float()
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        next_return = returns[t + 1]
                    if self.dones[t]:
                        lastgaelam = 0
                    returns[t] = self.rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - self.values
        return returns, advantages
    def train(self):
        # collect data
        for update in range(1,self.num_updates+1):

            if args.anneal_lr:
                self.lr_frac = 1.0 - (update-1.0) / self.num_updates
                if self.lr_frac < 0.66 and self.checkpoint==0:
                    save_dir = f'./trained/PPO/tetris_{output_index}'
                    os.makedirs(save_dir, exist_ok=True) 
                    torch.save(agent.model.state_dict(), os.path.join(save_dir, 'stage1.pth'))
                    self.checkpoint+=1
                elif self.lr_frac < 0.33 and self.checkpoint==1:
                    torch.save(agent.model.state_dict(), f'./trained/PPO/tetris_{output_index}/stage2.pth')
                    self.checkpoint+=1
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * self.lr_frac
            for step in range(args.num_steps):
                self.global_step += 1*args.num_envs 
                with torch.no_grad():
                    obs = env.get_ppo_state()
                    self.obs[step] = torch.tensor(obs,dtype=torch.float32).to(device)
                    self.model.reset_noise()
                    action_idx, logprob, _, value = self.model.get_action_and_value(self.obs[step].unsqueeze(0))
                    next_steps = env.get_next_states()
                    next_actions, next_states = zip(*next_steps.items())
                    # print(next_actions)
                    if action_idx>=len(next_actions):
                        action = next_actions[0]
                    else:
                        action = next_actions[action_idx]
                    # print(action)
                    self.values[step] = value.flatten()
                    
                self.actions[step] = action_idx
                self.logprobs[step] = logprob
                # TRY NOT TO MODIFY: execute the game and log data.
                reward, done = env.step(action, render=False)
                self.rewards[step] = torch.tensor(reward).to(device).view(-1)
                self.next_obs, self.next_done = torch.tensor(env.get_current_board_state(),dtype=torch.float32).to(device), torch.tensor(done).to(device)
                if done:
                    env.reset()
            # Reward Scaling
            self.rms.update(self.rewards)
            self.rewards = (self.rewards - self.rms.mean) / self.rms.std

            returns,advantages=self.GAE()
            # flatten the batch
            b_obs = self.obs.reshape((-1,) + (20,10))
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + (1,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)
            # print(f"value:{torch.max(b_values)}, return: {torch.max(b_returns)}")
            #print(f"b_returns:{b_returns},b_values:{b_values}, b_advantages:{b_advantages}")
            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = self.model.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds].squeeze(1))
                    # print(f"newlogprob:{newlogprob},{newlogprob.shape}, b_logprobs[mb_inds]:{b_logprobs[mb_inds]},{b_logprobs[mb_inds].shape}")
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    # print(f"ratio:{ratio}")
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    # Policy loss
                    # print(f"ratio:{ratio}, b_advantages[mb_inds]:{b_advantages[mb_inds]}")
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:                        
                        v_loss_unclipped = (newvalue-b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped-b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    # print(f"epoch:{epoch}, step:{step}, pg_loss:{pg_loss.item()}, v_loss:{v_loss.item()}, entropy_loss:{entropy_loss.item()}, approx_kl:{approx_kl.item()}")
                    loss = pg_loss - args.ent_coef * entropy_loss +  args.vf_coef * v_loss 

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
                    self.optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break 
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            writer.add_scalar("charts/value", b_values.mean().item(), self.global_step)
            writer.add_scalar("charts/return", b_returns.mean().item(), self.global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)          
            

if __name__=="__main__":
    output_index = 4
    print(args)
    run_name=f"tetrus_{args.seed}_{int(time.time())}"
    # whether build wandb 
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # Tensorboard construction
    writer = SummaryWriter(f"./runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )        
    #test
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    #build envs
    env = make_env(args.width,args.height,args.block_size)()

    agent=PPO_agent(env,device,lr=args.learning_rate,writer=writer)
    if args.from_pretrained:
        agent.model.load_state_dict(torch.load(f'./trained/PPO/tetris_{output_index}.pth', map_location=device))
    agent.train()
    torch.save(agent.model.state_dict(), f'./trained/PPO/tetris_{output_index}/stage3.pth')
    
