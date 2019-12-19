import torch
from torch.distributions import Categorical
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from .utils import QNet, Policy


class SAC_Discrete:
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"
    def __init__(self, state_size, action_size, hidden_size, device):

        self.action_size = action_size
        self.state_size = state_size
        self.device = device

        self.critic_local = QNet(state_size, action_size, hidden_size, device)
        self.critic_local_2 = QNet(state_size, action_size, hidden_size, device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=3e-4, eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(), lr=3e-4, eps=1e-4)
        self.critic_target = QNet(state_size, action_size, hidden_size, device)
        self.critic_target_2 = QNet(state_size, action_size, hidden_size, device)

        SAC_Discrete.copy_model_over(self.critic_local, self.critic_target)
        SAC_Discrete.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.actor_local = Policy(state_size, action_size, hidden_size, device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=3e-4, eps=1e-4)
        self.automatic_entropy_tuning = True
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=3e-4, eps=1e-4)

        self.add_extra_noise = False

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    @staticmethod
    def create_actor_distribution(action_types, actor_output, action_size):
        """Creates a distribution that the actor can then use to randomly draw actions"""
        if action_types == "DISCRETE":
            assert actor_output.size()[1] == action_size, "Actor output the wrong size"
            action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
        else:
            raise ValueError('what?!')
            # assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
            # means = actor_output[:, :action_size].squeeze(0)
            # stds = actor_output[:, action_size:].squeeze(0)
            # if len(means.shape) == 2: means = means.squeeze(-1)
            # if len(stds.shape) == 2: stds = stds.squeeze(-1)
            # if len(stds.shape) > 1 or len(means.shape) > 1:
            #     raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
            # action_distribution = torch.normal.Normal(means.squeeze(0), torch.abs(stds))
        return action_distribution

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        state = torch.tensor(state)
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities).unsqueeze(0)
        action_distribution = SAC_Discrete.create_actor_distribution("DISCRETE", action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)

            # print(f'qf1_next_target shape : {qf1_next_target.size()}')

            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * 0.99 * min_qf_next_target

            # print(f'next_q_value shape : {next_q_value.size()}')

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())

        # print(f'qf1 shape : {qf1.size()}')

        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi

        # my fix
        policy_loss = torch.sum(action_probabilities * inside_term, dim=1)

        policy_loss = policy_loss.mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)
        return qf1_loss.data.numpy(), qf2_loss.data.numpy(), -1, policy_loss.data.numpy(), self.alpha.data.numpy()

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        """Updates the parameters for the actor, both critics and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1, 1)
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2, 1)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss, 1)

        self.soft_update_of_target_network(self.critic_local, self.critic_target, 0.95)
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2, 0.95)
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """Takes an optimisation step by calculating gradients given the loss and then updating the parameters"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # reset gradients to 0
        loss.backward(retain_graph=retain_graph)  # this calculates the gradients
        # self.logger.info("Loss -- {}".format(loss.item()))
        # if self.debug_mode: self.log_gradient_and_weight_information(network, optimizer)
        if clipping_norm is not None:
            for net in network:
                # clip gradients to help stabilise training
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)
        optimizer.step()  # this applies the gradients

    def batch_action(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        action, (_, _), max_probability_action = self.produce_action_and_action_info(state)
        if not use_gumbel:
            action = max_probability_action.data.numpy()
        else:
            action = action.data.numpy()

        if need_argmax:
            return action

        onehot_actions = np.eye(self.action_size)
        onehot_actions = onehot_actions[action]
        return onehot_actions

    def get_single_action(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        # state: [state_size, ]
        # return [action_size, ]
        action = self.get_batch_actions(np.array([state]), need_argmax, use_gumbel, temperature)
        return action[0]

    def save(self, folder):
        pass

    def load(self, folder):
        pass

    def update_step(self, batch):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch
        return self.update(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)

    def update_V_target(self, tau):
        SAC_Discrete.copy_model_over(self.critic_local, self.critic_target)
        SAC_Discrete.copy_model_over(self.critic_local_2, self.critic_target_2)
