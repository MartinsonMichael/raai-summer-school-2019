from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from .utils_continues import QNet, Policy


class SAC_Continues:
    """Soft Actor-Critic model based on the 2018 paper https://arxiv.org/abs/1812.05905 and on this github implementation
      https://github.com/pranz24/pytorch-soft-actor-critic. It is an actor-critic algorithm where the agent is also trained
      to maximise the entropy of their actions as well as their cumulative reward"""
    agent_name = "SAC"

    def __init__(self, state_size, action_size, hidden_size, device):

        if isinstance(action_size, tuple):
            if action_size.__len__() > 1:
                raise ValueError('action shape doesnt understood')
            action_size = action_size[0]
        elif not isinstance(action_size, int):
            raise ValueError('action shape doesnt understood')

        self.action_size = action_size
        self.state_size = state_size
        self.device = device

        self.critic_local = QNet(state_size, action_size, hidden_size, device)
        self.critic_local_2 = QNet(state_size, action_size, hidden_size, device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=3e-4, eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(), lr=3e-4, eps=1e-4)
        self.critic_target = QNet(state_size, action_size, hidden_size, device)
        self.critic_target_2 = QNet(state_size, action_size, hidden_size, device)

        SAC_Continues.copy_model_over(self.critic_local, self.critic_target)
        SAC_Continues.copy_model_over(self.critic_local_2, self.critic_target_2)

        self.actor_local = Policy(state_size, action_size, hidden_size, device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=3e-4, eps=1e-4)

        self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=3e-4, eps=1e-4)

    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the log probability of the action, and the tanh of the mean action"""
        actor_output = self.actor_local(state)
        mean, log_std = actor_output[:, :self.action_size], actor_output[:, self.action_size:]
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Runs a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_all_parameters(qf1_loss, qf2_loss, policy_loss, alpha_loss)
        return qf1_loss.data.numpy(), qf2_loss.data.numpy(), -1, policy_loss.data.numpy(), self.alpha.data.numpy()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch, next_state_action)
            qf2_next_target = self.critic_target_2(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1.0 - mask_batch) * 0.95 * min_qf_next_target
        qf1 = self.critic_local(state_batch, action_batch)
        qf2 = self.critic_local_2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, log_pi, _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch, action)
        qf2_pi = self.critic_local_2(state_batch, action)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        return policy_loss, log_pi

    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

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

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def update_all_parameters(self, critic_loss_1, critic_loss_2, actor_loss, alpha_loss):
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1, 1)
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2, 1)
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss, 1)

        self.soft_update_of_target_network(self.critic_local, self.critic_target, 0.95)
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2, 0.95)
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def batch_action(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        action, _, max_probability_action = self.produce_action_and_action_info(torch.from_numpy(state.astype(np.float32)))
        if not use_gumbel:
            action = max_probability_action.data.numpy()
        else:
            action = action.data.numpy()

        return action

    def single_action(self, state, need_argmax=False, use_gumbel=True, temperature=0.5):
        # state: [state_size, ]
        # return [action_size, ]
        action = self.get_batch_actions(np.array([state]), need_argmax, use_gumbel, temperature)
        return action[0]

    def save(self, folder):
        import os
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.critic_target.state_dict(), os.path.join(folder, 'q1_target'))
        torch.save(self.critic_target_2.state_dict(), os.path.join(folder, 'q2_target'))

        torch.save(self.critic_local.state_dict(), os.path.join(folder, 'q1'))
        torch.save(self.critic_local_2.state_dict(), os.path.join(folder, 'q2'))

        torch.save(self.actor_local.state_dict(), os.path.join(folder, 'policy'))

    def load(self, folder):
        import os
        if not os.path.exists(folder):
            raise ValueError(f'there is no such folder: {folder}')
        self.critic_local.load_state_dict(torch.load(os.path.join(folder, 'q1')))
        self.critic_local_2.load_state_dict(torch.load(os.path.join(folder, 'q2')))

        self.critic_target.load_state_dict(torch.load(os.path.join(folder, 'q1_target')))
        self.critic_target_2.load_state_dict(torch.load(os.path.join(folder, 'q2_target')))

        self.actor_local.load_state_dict(torch.load(os.path.join(folder, 'policy')))

    def update_step(self, batch):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = batch
        return self.update(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)

    def hard_target_update(self):
        SAC_Continues.copy_model_over(self.critic_local, self.critic_target)
        SAC_Continues.copy_model_over(self.critic_local_2, self.critic_target_2)
