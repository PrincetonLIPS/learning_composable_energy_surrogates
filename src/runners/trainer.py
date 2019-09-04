"""Handles training surrogates given a data dir"""
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable

import ast
import pdb
from copy import deepcopy
import glob
import os
import collections
import matplotlib.pyplot as plt

import io

from .. import arguments
from .. import fa_combined as fa
# from ..runners.online_trainer import OnlineTrainer
from ..logging.tensorboard_logger import Logger as TFLogger
from ..maps.spline_function_space_map import SplineFunctionSpaceMap
from ..energy_model.fenics_energy_model import FenicsEnergyModel
from ..nets.feed_forward_net import FeedForwardNet
from ..nets.ring_net import RingNet
from ..geometry.polar import Polarizer, SemiPolarizer
from ..geometry.remove_rigid_body import RigidRemover

from ..util.timer import Timer
from ..util.jacobian import compute_jacobian
from ..util.rigid import apply_random_rotation, apply_random_translation
from ..viz.plotting import plot_boundary, plot_vectors
"""Example namedtuple to store data
    x: vector of boundary displacements
    p: vector of params (eg c1, c2 for metamaterial), might be None
    v: vector used to find Hvp
    f: Energy of PDE when solved with boundary condition x
    J: Jacobian df/dx
    Hvp: Hessian vector product (d2f/dx2)*v
"""

Example = collections.namedtuple('Example', 'x p v f J Hvp')


def log(message, *args):
    message = str(message)
    for arg in args:
        message = message + str(arg)
    print(message)
    with open('log.txt', 'w+') as logfile:
        logfile.write(message)


def rmse(y, y_, loss_scale=torch.Tensor([1.0])):
    """Root mean squared error"""
    y_ = y_.to(y.device)
    assert y.size() == y_.size()
    loss_scale = loss_scale.to(y.device)
    if len(y.size()) > 1:
        loss_scale = loss_scale.view(-1,
                                     *[1 for _ in range(len(y.size()) - 1)])
        loss_scale = loss_scale * torch.prod(
            torch.Tensor([s for s in y.shape[1:]]).to(y.device))
    return torch.sqrt(torch.mean((y - y_)**2 * loss_scale))


def error_percent(y, y_):
    """Mean of abs(err) / abs(true_val)"""
    y_ = y_.view(y_.size(0), -1).cpu()
    y = y.view(y_.size()).cpu()
    return torch.mean(
        torch.norm(y - y_, dim=1) / (torch.norm(y, dim=1) + 1e-7))


def similarity(y, y_):
    """Cosine similarity between vectors"""
    y = y.view(y.size(0), -1).cpu()
    y_ = y_.view(y_.size(0), -1).cpu()
    assert y.size(0) == y_.size(0)
    assert y.size(1) == y_.size(1)
    return torch.mean(
        torch.sum(y * y_, dim=1) /
        (torch.norm(y, dim=1) * torch.norm(y_, dim=1)))


class Trainer(object):
    def __init__(self,
                 args,
                 data_dir,
                 surrogate,
                 tflogger=None,
                 sobolev_J=True,
                 sobolev_Hvp=True,
                 pde=None):
        self.args = args
        self.pde = pde
        self.sobolev_J = sobolev_J
        self.sobolev_Hvp = sobolev_Hvp
        if self.sobolev_Hvp:
            assert self.sobolev_J, "Should use J if using Hvp"
        self.surrogate = surrogate
        self.tflogger = tflogger
        self.semipolarizer = SemiPolarizer(self.surrogate.fsm)
        self.init_optimizer()

    def init_optimizer(self, args):
        # Create optimizer if surrogate is trainable
        if hasattr(self.surrogate, 'parameters'):
            if self.args.optimizer == 'adam' or self.args.optimizer == 'amsgrad':
                self.optimizer = torch.optim.AdamW(
                    (p
                     for p in self.surrogate.parameters() if p.requires_grad),
                    args.lr,
                    weight_decay=args.wd,
                    betas=ast.literal_eval(args.adam_betas),
                    amsgrad=(self.args.optimizer == 'amsgrad'))
            elif self.args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(
                    (p
                     for p in self.surrogate.parameters() if p.requires_grad),
                    args.lr,
                    momentum=0.9,
                    weight_decay=args.wd)
            elif self.args.optimizer == 'radam':
                from ..util.radam import RAdam
                self.optimizer = RAdam(
                    (p
                     for p in self.surrogate.parameters() if p.requires_grad),
                    args.lr,
                    weight_decay=args.wd,
                    betas=ast.literal_eval(args.adam_betas))
            else:
                raise Exception("Unknown optimizer")
            if self.args.fix_batch:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    patience=10 if self.args.fix_batch else 20 *
                    len(self.train_data) // self.args.batch_size,
                    verbose=self.args.verbose,
                    factor=1. / np.sqrt(np.sqrt(np.sqrt(2))))
            else:
                '''
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    gamma=0.1,
                    milestones=[1e2,5e2,2e3,1e4,1e5])
                '''
                self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                    self.optimizer,
                    base_lr=args.lr * 1e-3,
                    max_lr=3 * args.lr,
                    step_size_up=1000,
                    step_size_down=None,
                    mode='triangular',
                    gamma=0.995,
                    scale_fn=None,
                    scale_mode='cycle',
                    cycle_momentum=False,
                    base_momentum=0.8,
                    max_momentum=0.9,
                    last_epoch=-1)
        else:
            self.optimizer = None

    def forward(self, x, p, v):
        if self.sobolev_Hvp:
            raise Exception("Currently deprecated")
        else:
            f, J = self.surrogate.f_J(x, p)
            return f, J, None

    def train_step(self, step):
        """Do a single step of Sobolev training. Log stats to tensorboard."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        x, p, v, f, J, Hvp = self.get_train_batch()
        with Timer() as timer:
            fhat, Jhat, Hvphat = self.forward(x, p, v)

        if self.evaluator is not None and self.args.deploy_loss_weight != 0.0:
            deploy_loss = self.evaluator.differentiable_loss(self.surrogate)
        else:
            deploy_loss = None
        self.tflogger.log_scalar('batch_forward_time', timer.interval, step)
        total_loss = self.stats(step,
                                x,
                                f,
                                J,
                                Hvp,
                                fhat,
                                Jhat,
                                Hvphat,
                                deploy_loss=deploy_loss)

        if self.optimizer:
            total_loss.backward()
            if self.args.verbose:
                log([
                    getattr(p.grad, 'data',
                            torch.Tensor([0.0])).norm().cpu().numpy().sum()
                    for p in self.surrogate.net.parameters()
                ])
            if self.args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.surrogate.net.parameters(),
                                               self.args.clip_grad_norm)
            self.optimizer.step()
            if self.args.fix_batch:
                self.scheduler.step(total_loss)
            else:
                self.scheduler.step()
            if self.args.verbose:
                log("lr: {}".format(self.optimizer.param_groups[0]['lr']))

        return total_loss.item()

    def val_step(self, step):
        """Do a single validation step. Log stats to tensorboard."""
        x, p, v, f, J, Hvp = self.get_val_batch()
        fhat, Jhat, Hvphat = self.forward(x, p, v)
        if (self.evaluator is not None and
                (step - 1) % self.args.deploy_every == 0):
            errors_V, zero_errors_V, initial_errors_V = self.evaluator.eval_n(
                self.surrogate, step, self.args.deploy_n)
        else:
            errors_V = None
            zero_errors_V = None
            initial_errors_V = None
        return self.stats(step,
                          x,
                          f,
                          J,
                          Hvp,
                          fhat,
                          Jhat,
                          Hvphat,
                          phase='val',
                          errors_V=errors_V,
                          zero_errors_V=zero_errors_V,
                          initial_errors_V=initial_errors_V).item()

    def visualize(self, step, dataset, dataset_name):
        inds = np.random.randint(0, len(dataset), size=4)
        data = [dataset[i] for i in inds]
        examples = [d[0] for d in data]
        x, p, v, f, J, Hvp = self.get_batch(examples=examples)
        fhat, Jhat = self.surrogate.f_J(x, p)

        x, f, J, fhat, Jhat = x.cpu(), f.cpu(), J.cpu(), fhat.cpu(), Jhat.cpu()

        cuda = self.surrogate.fsm.cuda
        self.surrogate.fsm.cuda = False

        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        axes = [ax for axs in axes for ax in axs]
        RCs = self.surrogate.fsm.ring_coords.cpu().detach().numpy()
        rigid_remover = RigidRemover(self.surrogate.fsm)
        for i, ax in enumerate(axes):
            locs = RCs + self.surrogate.fsm.to_ring(x[i]).detach().numpy()
            plot_boundary(lambda x: (0, 0),
                          1000,
                          label='reference, f={:.3e}'.format(f[i].item()),
                          ax=ax,
                          color='k')
            #plot_boundary(self.surrogate.fsm.get_query_fn(true_u[i]),
            #              1000, ax=ax, label='true_u, f={:.3e}'.format(f[i].item()),
            #              linestyle='--')
            plot_boundary(self.surrogate.fsm.get_query_fn(x[i]),
                          1000,
                          ax=ax,
                          label='ub, fhat={:.3e}'.format(fhat[i].item()),
                          linestyle='-',
                          color='darkorange')
            plot_boundary(self.surrogate.fsm.get_query_fn(
                rigid_remover(x[i].unsqueeze(0)).squeeze(0)),
                          1000,
                          ax=ax,
                          label='rigid removed',
                          linestyle='--',
                          color='blue')
            if J is not None and Jhat is not None:
                J_ = self.surrogate.fsm.to_ring(J[i])
                Jhat_ = self.surrogate.fsm.to_ring(Jhat[i])
                normalizer = np.mean(
                    np.nan_to_num([
                        J_.norm(dim=1).detach().numpy(),
                        Jhat_.norm(dim=1).detach().numpy()
                    ]))
                plot_vectors(locs,
                             J_.detach().numpy(),
                             ax=ax,
                             label='J',
                             color='darkgreen',
                             normalizer=normalizer,
                             scale=1.)
                plot_vectors(locs,
                             Jhat_.detach().numpy(),
                             ax=ax,
                             color='darkorchid',
                             label='Jhat',
                             normalizer=normalizer,
                             scale=1.)
                plot_vectors(locs, (J_ - Jhat_).detach().numpy(),
                             ax=ax,
                             color='red',
                             label='residual J-Jhat',
                             normalizer=normalizer,
                             scale=1.)

            ax.legend()
        fig.canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        self.tflogger.log_images('{} displacements'.format(dataset_name),
                                 [buf], step)

        if J is not None and Jhat is not None:
            fig, axes = plt.subplots(2, 2, figsize=(8, 8))
            axes = [ax for axs in axes for ax in axs]
            for i, ax in enumerate(axes):
                J_ = self.surrogate.fsm.to_ring(J[i])
                Jhat_ = self.surrogate.fsm.to_ring(Jhat[i])
                normalizer = 10 * np.mean(
                    np.nan_to_num([
                        J_.norm(dim=1).detach().numpy(),
                        Jhat_.norm(dim=1).detach().numpy()
                    ]))
                plot_boundary(lambda x: (0, 0), 1000, label='reference', ax=ax)
                plot_boundary(self.surrogate.fsm.get_query_fn(J[i]),
                              100,
                              ax=ax,
                              label='true_J, f={:.3e}'.format(f[i].item()),
                              linestyle='--',
                              normalizer=normalizer)
                plot_boundary(self.surrogate.fsm.get_query_fn(Jhat[i]),
                              100,
                              ax=ax,
                              label='surrogate_J, fhat={:.3e}'.format(
                                  fhat[i].item()),
                              linestyle='-.',
                              normalizer=normalizer)
                ax.legend()
            fig.canvas.draw()
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            self.tflogger.log_images('{} Jacobians'.format(dataset_name),
                                     [buf], step)
        self.surrogate.fsm.cuda = cuda

    def stats(self,
              step,
              x,
              f,
              J,
              Hvp,
              fhat,
              Jhat,
              Hvphat,
              errors_V=None,
              zero_errors_V=None,
              deploy_loss=None,
              initial_errors_V=None,
              phase='train'):
        """Take ground truth and predictions. Log stats and return loss."""

        if self.args.quadratic_loss_scale:
            loss_scale = 1. / torch.mean(x**2, dim=1)
        else:
            loss_scale = torch.Tensor([1.0])
        f_loss = rmse(f / train_f_std, fhat / train_f_std,
                      loss_scale)
        J_loss = rmse(J / train_J_std, Jhat / train_J_std, loss_scale)
        H_loss = rmse(Hvp, Hvphat, loss_scale) if self.sobolev_Hvp else 0.

        total_loss = (f_loss + self.args.J_weight * J_loss +
                      self.args.H_weight * H_loss)

        if deploy_loss is not None:
            total_loss = ((1. - self.args.deploy_loss_weight) * total_loss +
                          self.args.deploy_loss_weight * deploy_loss.cpu())

        f_pce = error_percent(f, fhat)
        J_pce = error_percent(J, Jhat)
        H_pce = error_percent(Hvp, Hvphat) if self.sobolev_Hvp else 0.

        J_sim = similarity(J, Jhat)
        H_sim = similarity(Hvp, Hvphat) if self.sobolev_Hvp else 0.

        if self.args.verbose and phase == 'train':
            log("\n")
            log("{} step {}: total_loss {:.3e}, f_loss {:.3e}, J_loss {:.3e}, H_loss {:.3e}, "
                "f_pce {:.3e}, J_pce {:.3e}, H_pce {:.3e}, J_sim {:.3e}, H_sim {:.3e}"
                .format(phase, step, float(total_loss), float(f_loss),
                        float(J_loss), float(H_loss),
                        float(f_pce), float(J_pce), float(H_pce), float(J_sim),
                        float(H_sim)))
            log("f_mean: {}, fhat_mean: {}, J_mean: {}, Jhat_mean: {}".format(
                torch.mean(f).item(),
                torch.mean(fhat).item(),
                torch.mean(J).item(),
                torch.mean(Jhat).item()))

        if self.tflogger is not None:
            self.tflogger.log_scalar('train_set_size', len(self.train_data),
                                     step)
            self.tflogger.log_scalar('val_set_size', len(self.val_data), step)
            if self.evaluator is not None:
                self.tflogger.log_scalar('deploy_set_size',
                                         len(self.evaluator.val_data), step)
            if hasattr(self.surrogate, 'net') and hasattr(
                    self.surrogate.net, 'parameters'):
                self.tflogger.log_scalar(
                    'param_norm_sum',
                    sum([
                        p.norm().data.cpu().numpy().sum()
                        for p in self.surrogate.net.parameters()
                    ]), step)
            self.tflogger.log_scalar('total_loss_' + phase, total_loss, step)
            self.tflogger.log_scalar('f_loss_' + phase, f_loss, step)
            self.tflogger.log_scalar('f_pce_' + phase, f_pce, step)

            self.tflogger.log_scalar('f_mean_' + phase, f.mean().item(), step)
            self.tflogger.log_scalar('f_std_' + phase, f.std().item(), step)
            self.tflogger.log_scalar('fhat_mean_' + phase,
                                     fhat.mean().item(), step)
            self.tflogger.log_scalar('fhat_std_' + phase,
                                     fhat.std().item(), step)

            if self.sobolev_J:
                self.tflogger.log_scalar('J_loss_' + phase, J_loss, step)
                self.tflogger.log_scalar('J_pce_' + phase, J_pce, step)
                self.tflogger.log_scalar('J_sim_' + phase, J_sim, step)

                self.tflogger.log_scalar('J_mean_' + phase,
                                         J.mean().item(), step)
                self.tflogger.log_scalar('J_std_mean_' + phase,
                                         J.std(dim=1).mean().item(), step)
                self.tflogger.log_scalar('Jhat_mean_' + phase,
                                         Jhat.mean().item(), step)
                self.tflogger.log_scalar('Jhat_std_mean_' + phase,
                                         Jhat.std(dim=1).mean().item(), step)

            if self.sobolev_Hvp:
                self.tflogger.log_scalar('Hvp_loss_' + phase, H_loss, step)
                self.tflogger.log_scalar('Hvp_pce_' + phase, H_pce, step)
                self.tflogger.log_scalar('H_sim_' + phase, H_sim, step)

            if errors_V is not None:
                self.tflogger.log_scalar('Errors_V_avg', np.mean(errors_V),
                                         step)
                self.tflogger.log_scalar('Errors_V_std', np.std(errors_V),
                                         step)
                self.tflogger.log_scalar('Solution_norms_avg',
                                         np.mean(zero_errors_V), step)
                self.tflogger.log_scalar('Solution_norms_std',
                                         np.std(zero_errors_V), step)
                self.tflogger.log_scalar('Errors_initial_V_avg',
                                         np.mean(initial_errors_V), step)
                self.tflogger.log_scalar('Errors_initial_V_std',
                                         np.std(initial_errors_V), step)

        return total_loss

    def get_train_batch(self):
        return self.get_batch(self.train_data, self.args.batch_size)

    def get_val_batch(self):
        return self.get_batch(self.val_data, self.args.batch_size)

    def get_batch(self, dataset=None, batch_size=None, examples=None):
        """Get a random batch of examples from dataset"""
        # fsm_cuda = self.surrogate.fsm.cuda
        # self.surrogate.fsm.cuda = False
        if examples is None:
            assert dataset is not None and batch_size is not None
            if len(dataset) < batch_size:
                raise Exception(
                    "Dataset size {} too small for batch size {}".format(
                        len(dataset), batch_size))
            if self.args.fix_batch:  # Train on only one batch
                inds = np.arange(batch_size)
            else:
                inds = np.random.choice(len(dataset),
                                        batch_size,
                                        replace=False)
            examples = [dataset[i] for i in inds]

        x = torch.stack([e.x for e in examples])

        if examples[0].p is None:
            p = None
        else:
            p = Variable(self.surrogate.fsm._cuda(
                torch.Tensor(np.stack([e.p for e in examples]))),
                         requires_grad=True)
        f = Variable(
            self.surrogate.fsm._cuda(
                torch.Tensor(np.stack([e.f for e in examples]))))

        if all(e.J is not None for e in examples):
            J = self.surrogate.fsm.to_ring(
                torch.stack([e.J for e in examples]))
        else:
            J = None
        if all(e.Hvp is not None and e.v is not None for e in examples):
            raise Exception("Currently deprecated")
        else:
            Hvp = None
            v = None

        if not isinstance(self.surrogate.net, RingNet):
            x = self.surrogate.fsm.to_torch(x)
            J = self.surrogate.fsm.to_torch(J)
        else:
            x = self.surrogate.fsm.proc_torch(x)
            J = self.surrogate.fsm.proc_torch(J)

        x = Variable(x.data, requires_grad=True)
        J = Variable(J.data, requires_grad=True)

        return x, p, v, f, J, Hvp
