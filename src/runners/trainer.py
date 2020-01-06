"""Handles training surrogates given a data dir"""
import numpy as np
import torch
from torch.utils.data import DataLoader

import pdb
import ast
import matplotlib.pyplot as plt
import math

import io
from ..geometry.remove_rigid_body import RigidRemover

from ..util.timer import Timer
from ..viz.plotting import plot_boundary, plot_vectors


def _cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


class Trainer(object):
    def __init__(self, args, surrogate, train_data, val_data, tflogger=None, pde=None):
        self.args = args
        self.pde = pde
        self.surrogate = surrogate
        self.tflogger = tflogger
        self.train_data = train_data
        self.val_data = val_data
        self.train_loader = DataLoader(
            self.train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True
        )
        self.init_optimizer()
        self.train_f_std = _cuda(torch.Tensor([[1.0]]))
        self.train_J_std = _cuda(torch.Tensor([[1.0]]))

    def init_optimizer(self):
        # Create optimizer if surrogate is trainable
        if hasattr(self.surrogate, "parameters"):
            if self.args.optimizer == "adam" or self.args.optimizer == "amsgrad":
                self.optimizer = torch.optim.AdamW(
                    (p for p in self.surrogate.parameters() if p.requires_grad),
                    self.args.lr,
                    weight_decay=self.args.wd,
                    amsgrad=(self.args.optimizer == "amsgrad"),
                )
            elif self.args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(
                    (p for p in self.surrogate.parameters() if p.requires_grad),
                    self.args.lr,
                    momentum=0.9,
                    weight_decay=self.args.wd,
                )
            elif self.args.optimizer == "radam":
                from ..util.radam import RAdam

                self.optimizer = RAdam(
                    (p for p in self.surrogate.parameters() if p.requires_grad),
                    self.args.lr,
                    weight_decay=self.args.wd,
                    betas=ast.literal_eval(self.args.adam_betas),
                )
            else:
                raise Exception("Unknown optimizer")
            '''
            if self.args.fix_batch:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    patience=10
                    if self.args.fix_batch
                    else 20 * len(self.train_data) // self.args.batch_size,
                    verbose=self.args.verbose,
                    factor=1.0 / np.sqrt(np.sqrt(np.sqrt(2))),
                )
            else:
                """
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer,
                    gamma=0.1,
                    milestones=[1e2,5e2,2e3,1e4,1e5])
                """
            '''
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.args.lr * 1e-3,
                max_lr=3 * self.args.lr,
                step_size_up=int(math.ceil(len(self.train_loader) / 2)),
                step_size_down=int(math.floor(len(self.train_loader) / 2)),
                mode="triangular",
                gamma=0.995,
                scale_fn=None,
                scale_mode="cycle",
                cycle_momentum=False,
                base_momentum=0.8,
                max_momentum=0.9,
                last_epoch=-1,
            )
        else:
            self.optimizer = None

    def train_step(self, step, batch):
        """Do a single step of Sobolev training. Log stats to tensorboard."""
        if self.optimizer:
            self.optimizer.zero_grad()
        u, p, f, J, H = batch

        with Timer() as timer:
            u, p, f, J, H = _cuda(u), _cuda(p), _cuda(f), _cuda(J), _cuda(H)

        self.tflogger.log_scalar("batch_cuda_time", timer.interval, step)

        with Timer() as timer:
            if self.args.hess:
                vectors = torch.randn(*J.size())
                fhat, Jhat, Hvphat = self.surrogate.f_J_Hvp(u, p,
                                                            vectors=vectors)
                Hvp = (vectors.view(*J.size(), 1)*H).sum(dim=1)
            else:
                fhat, Jhat = self.surrogate.f_J(u, p)
                Hvphat = torch.zeros_like(Jhat)
                Hvp = torch.zeros_like(Jhat)
        # pdb.set_trace()

        self.tflogger.log_scalar("batch_forward_time", timer.interval, step)

        with Timer() as timer:
            f_loss, f_pce, J_loss, J_sim, H_loss, H_sim, total_loss = self.stats(
                step, u, f, J, Hvp, fhat, Jhat, Hvphat
            )
        self.tflogger.log_scalar("stats_forward_time", timer.interval, step)

        with Timer() as timer:
            if self.optimizer:
                total_loss.backward()
                if self.args.verbose:
                    log(
                        [
                            getattr(p.grad, "data", torch.Tensor([0.0]))
                            .norm()
                            .cpu()
                            .numpy()
                            .sum()
                            for p in self.surrogate.net.parameters()
                        ]
                    )
                if self.args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.surrogate.net.parameters(), self.args.clip_grad_norm
                    )
                self.optimizer.step()
                self.scheduler.step()
                if self.args.verbose:
                    log("lr: {}".format(self.optimizer.param_groups[0]["lr"]))
        self.tflogger.log_scalar("backward_time", timer.interval, step)

        return (
            f_loss.item(),
            f_pce.item(),
            J_loss.item(),
            J_sim.item(),
            H_loss.item(),
            H_sim.item(),
            total_loss.item(),
        )

    def val_step(self, step):
        """Do a single validation step. Log stats to tensorboard."""
        for i, batch in enumerate(self.val_loader):
            u, p, f, J, H = batch
            u, p, f, J, H = _cuda(u), _cuda(p), _cuda(f), _cuda(J), _cuda(H)
            if self.args.hess:
                vectors = torch.randn(*J.size())
                fhat, Jhat, Hvphat = self.surrogate.f_J_Hvp(u, p,
                                                            vectors=vectors)
                Hvp = (vectors.view(*J.size(), 1)*H).sum(dim=1)
            else:
                fhat, Jhat = self.surrogate.f_J(u, p)
                Hvphat = torch.zeros_like(Jhat)
                Hvp = torch.zeros_like(Jhat)

            u_ = torch.cat([u_, u.data], dim=0) if i > 0 else u.data
            f_ = torch.cat([f_, f.data], dim=0) if i > 0 else f.data
            J_ = torch.cat([J_, J.data], dim=0) if i > 0 else J.data
            Hvp_ = torch.cat([Hvp_, Hvp.data], dim=0) if i > 0 else Hvp.data
            fhat_ = torch.cat([fhat_, fhat.data], dim=0) if i > 0 else fhat.data
            Jhat_ = torch.cat([Jhat_, Jhat.data], dim=0) if i > 0 else Jhat.data
            Hvphat_ = torch.cat([Hvphat_, Hvphat.data], dim=0) if i > 0 else Hvphat.data


        return list(
            r.item() for r in self.stats(step, u_, f_, J_, Hvp_, fhat_, Jhat_, Hvphat_, phase="val")
        )

    def visualize(self, step, batch, dataset_name):
        u, p, f, J, H = batch
        u, p, f, J = u[:16], p[:16], f[:16], J[:16]
        fhat, Jhat = self.surrogate.f_J(u, p)

        assert len(u) <= 16
        assert len(u) == len(p)
        assert len(u) == len(f)
        assert len(u) == len(J)
        assert len(u) == len(fhat)
        assert len(u) == len(Jhat)

        u, f, J, fhat, Jhat = u.cpu(), f.cpu(), J.cpu(), fhat.cpu(), Jhat.cpu()

        cuda = self.surrogate.fsm.cuda
        self.surrogate.fsm.cuda = False

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = [ax for axs in axes for ax in axs]
        RCs = self.surrogate.fsm.ring_coords.cpu().detach().numpy()
        rigid_remover = RigidRemover(self.surrogate.fsm)
        for i in range(len(u)):
            ax = axes[i]
            locs = RCs + self.surrogate.fsm.to_ring(u[i]).detach().numpy()
            plot_boundary(
                lambda x: (0, 0),
                1000,
                label="reference, f={:.3e}".format(f[i].item()),
                ax=ax,
                color="k",
            )
            plot_boundary(
                self.surrogate.fsm.get_query_fn(u[i]),
                1000,
                ax=ax,
                label="ub, fhat={:.3e}".format(fhat[i].item()),
                linestyle="-",
                color="darkorange",
            )
            plot_boundary(
                self.surrogate.fsm.get_query_fn(
                    rigid_remover(u[i].unsqueeze(0)).squeeze(0)
                ),
                1000,
                ax=ax,
                label="rigid removed",
                linestyle="--",
                color="blue",
            )
            if J is not None and Jhat is not None:
                J_ = self.surrogate.fsm.to_ring(J[i])
                Jhat_ = self.surrogate.fsm.to_ring(Jhat[i])
                normalizer = np.mean(
                    np.nan_to_num(
                        [
                            J_.norm(dim=1).detach().numpy(),
                            Jhat_.norm(dim=1).detach().numpy(),
                        ]
                    )
                )
                plot_vectors(
                    locs,
                    J_.detach().numpy(),
                    ax=ax,
                    label="J",
                    color="darkgreen",
                    normalizer=normalizer,
                    scale=1.0,
                )
                plot_vectors(
                    locs,
                    Jhat_.detach().numpy(),
                    ax=ax,
                    color="darkorchid",
                    label="Jhat",
                    normalizer=normalizer,
                    scale=1.0,
                )
                plot_vectors(
                    locs,
                    (J_ - Jhat_).detach().numpy(),
                    ax=ax,
                    color="red",
                    label="residual J-Jhat",
                    normalizer=normalizer,
                    scale=1.0,
                )

            ax.legend()
        fig.canvas.draw()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()
        self.tflogger.log_images("{} displacements".format(dataset_name), [buf], step)

        if J is not None and Jhat is not None:
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            axes = [ax for axs in axes for ax in axs]
            for i in range(len(J)):
                ax = axes[i]
                J_ = self.surrogate.fsm.to_ring(J[i])
                Jhat_ = self.surrogate.fsm.to_ring(Jhat[i])
                normalizer = 10 * np.mean(
                    np.nan_to_num(
                        [
                            J_.norm(dim=1).detach().numpy(),
                            Jhat_.norm(dim=1).detach().numpy(),
                        ]
                    )
                )
                plot_boundary(lambda x: (0, 0), 1000, label="reference", ax=ax)
                plot_boundary(
                    self.surrogate.fsm.get_query_fn(J[i]),
                    100,
                    ax=ax,
                    label="true_J, f={:.3e}".format(f[i].item()),
                    linestyle="--",
                    normalizer=normalizer,
                )
                plot_boundary(
                    self.surrogate.fsm.get_query_fn(Jhat[i]),
                    100,
                    ax=ax,
                    label="surrogate_J, fhat={:.3e}".format(fhat[i].item()),
                    linestyle="-.",
                    normalizer=normalizer,
                )
                ax.legend()
            fig.canvas.draw()
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            plt.close()
            self.tflogger.log_images("{} Jacobians".format(dataset_name), [buf], step)
        self.surrogate.fsm.cuda = cuda

    def stats(self, step, u, f, J, Hvp, fhat, Jhat, Hvphat, phase="train"):
        """Take ground truth and predictions. Log stats and return loss."""

        f_loss = torch.nn.functional.mse_loss(
            self.surrogate.scaler.scale(f, u), self.surrogate.scaler.scale(fhat, u)
        )
        J_loss = torch.nn.functional.mse_loss(J, Jhat)

        # pdb.set_trace()
        H_loss = torch.nn.functional.mse_loss(Hvp, Hvphat)

        total_loss = f_loss + self.args.J_weight * J_loss + self.args.H_weight * H_loss

        f_pce = error_percent(f, fhat)
        J_pce = error_percent(J, Jhat)
        H_pce = error_percent(Hvp, Hvphat)

        J_sim = similarity(J, Jhat)
        H_sim = similarity(Hvp, Hvphat)

        if self.tflogger is not None:
            self.tflogger.log_scalar("train_set_size", len(self.train_data), step)
            self.tflogger.log_scalar("val_set_size", len(self.val_data), step)

            if hasattr(self.surrogate, "net") and hasattr(
                self.surrogate.net, "parameters"
            ):
                self.tflogger.log_scalar(
                    "param_norm_sum",
                    sum(
                        [p.norm().sum().item() for p in self.surrogate.net.parameters()]
                    ),
                    step,
                )
            self.tflogger.log_scalar("total_loss_" + phase, total_loss.item(), step)
            self.tflogger.log_scalar("f_loss_" + phase, f_loss.item(), step)
            self.tflogger.log_scalar("f_pce_" + phase, f_pce.item(), step)

            self.tflogger.log_scalar("f_mean_" + phase, f.mean().item(), step)
            self.tflogger.log_scalar("f_std_" + phase, f.std().item(), step)
            self.tflogger.log_scalar("fhat_mean_" + phase, fhat.mean().item(), step)
            self.tflogger.log_scalar("fhat_std_" + phase, fhat.std().item(), step)

            self.tflogger.log_scalar("J_loss_" + phase, J_loss.item(), step)
            self.tflogger.log_scalar("J_pce_" + phase, J_pce.item(), step)
            self.tflogger.log_scalar("J_sim_" + phase, J_sim.item(), step)

            self.tflogger.log_scalar("H_loss_" + phase, H_loss.item(), step)
            self.tflogger.log_scalar("H_pce_" + phase, H_pce.item(), step)
            self.tflogger.log_scalar("H_sim_" + phase, H_sim.item(), step)

            self.tflogger.log_scalar("J_mean_" + phase, J.mean().item(), step)
            self.tflogger.log_scalar(
                "J_std_mean_" + phase, J.std(dim=1).mean().item(), step
            )
            self.tflogger.log_scalar("Jhat_mean_" + phase, Jhat.mean().item(), step)
            self.tflogger.log_scalar(
                "Jhat_std_mean_" + phase, Jhat.std(dim=1).mean().item(), step
            )

        return (f_loss, f_pce, J_loss, J_sim, H_loss, H_sim, total_loss)


def log(message, *args):
    message = str(message)
    for arg in args:
        message = message + str(arg)
    print(message)
    with open("log.txt", "w+") as logfile:
        logfile.write(message)


def error_percent(y, y_):
    """Mean of abs(err) / abs(true_val)"""
    y_ = y_.view(y_.size(0), -1).cpu()
    y = y.view(y_.size()).cpu()
    return torch.mean(torch.norm(y - y_, dim=1) / (torch.norm(y, dim=1) + 1e-14))


def similarity(y, y_):
    """Cosine similarity between vectors"""
    y = y.view(y.size(0), -1).cpu()
    y_ = y_.view(y_.size(0), -1).cpu()
    assert y.size(0) == y_.size(0)
    assert y.size(1) == y_.size(1)
    return torch.mean(
        torch.sum(y * y_, dim=1)
        / (torch.norm(y, dim=1) * torch.norm(y_, dim=1) + 1e-14)
    )
