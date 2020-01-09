import argparse


# str2bool
def s2b(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()

parser.add_argument("--run_local", help="Run locally", type=s2b, default=True)

parser.add_argument("--dagger", help="Do dagger", type=s2b, default=False)

parser.add_argument("--deploy", help="Deploy", type=s2b, default=True)

parser.add_argument(
    "--batch_normalize_loss",
    help="per-batch normalize f, J before loss",
    type=s2b,
    default=False,
)

parser.add_argument("--train_size", type=int, help="n train data", default=55000)
parser.add_argument("--val_size", type=int, help="n val data", default=5000)
parser.add_argument(
    "--n_safe", type=int, help="n train data maintained from original dist", default=0
)

parser.add_argument(
    "--max_collectors", help="max Collector workers", type=int, default=0
)
parser.add_argument(
    "--max_evaluators", help="max Evaluator workers", type=int, default=24
)

parser.add_argument(
    "--deploy_error_alpha",
    help="moving average alpha for deploy error",
    type=float,
    default=0.9,
)

parser.add_argument("--verbose", help="Verbose for debug", type=s2b, default=False)
parser.add_argument("--seed", help="Random seed", type=int, default=0)


parser.add_argument(
    "--sample_c", help="sample c1, c2. else take mean", type=s2b, default=False
)
parser.add_argument(
    "--c1_low", help="minimum low-freq param for pore shape", type=float, default=-0.1
)
parser.add_argument(
    "--c1_high", help="maximum low-freq param for pore shape", type=float, default=0.1
)
parser.add_argument(
    "--c2_low", help="minimum high-freq param for pore shape", type=float, default=-0.1
)
parser.add_argument(
    "--c2_high", help="maximum high-freq param for pore shape", type=float, default=0.1
)
parser.add_argument(
    "--min_feature_size",
    help="minimum distance between pore boundaries = minimum "
    "width of a ligament / material section in structure. We "
    "also use this as minimum width of pore.",
    type=float,
    default=0.15,
)
parser.add_argument(
    "--boundary_freq_scale",
    type=float,
    help="maximum frequency scale for boundary random fourier fn",
    default=20.0,
)
parser.add_argument(
    "--boundary_amp_scale",
    type=float,
    help="maximum amplitude scale for boundary random fourier fn,",
    default=0.25,
)
parser.add_argument(
    "--boundary_sin_scale",
    type=float,
    help="maximum amplitude scale for boundary sinusoid,",
    default=0.3,
)
parser.add_argument(
    "--boundary_gauss_scale",
    type=float,
    help="maximum amplitude scale for boundary iid gaussian strain,",
    default=0.1,
)
parser.add_argument(
    "--boundary_ax_scale",
    type=float,
    help="maximum amplitude scale for boundary axial strain,",
    default=0.3,
)
parser.add_argument(
    "--boundary_shear_scale",
    type=float,
    help="maximum amplitude scale for boundary shear strain,",
    default=0.05,
)
parser.add_argument(
    "--force_freq_scale",
    type=float,
    help="maximum frequency scale for force random fourier fn",
    default=0.0,
)
parser.add_argument(
    "--force_amp_scale",
    type=float,
    help="maximum amplitude scale for force random fourier fn,",
    default=0.0,
)
parser.add_argument(
    "--anneal_steps",
    type=int,
    help="number of anneal steps for data gathering",
    default=12,
)

parser.add_argument(
    "--remove_rigid",
    help="remove rigid body transforms before energy calculation",
    default=True,
    type=s2b,
)
parser.add_argument(
    "--semipolarize", help="preproc inputs to semipolar coords", default=True, type=s2b,
)
parser.add_argument("--use_bias", help="use biases in nets", default=True, type=s2b)
parser.add_argument(
    "--fix_normalizer", help="adapt whitening of net inputs", default=True, type=s2b
)
parser.add_argument(
    "--normalizer_alpha", help="alpha for normalizer EMA", default=0.999, type=float
)
parser.add_argument(
    "--solve_optimizer", help="adam or lbfgs", type=str, default="lbfgs"
)
parser.add_argument(
    "--solve_steps", help="steps for adam or sgd", default=1000, type=int
)
parser.add_argument(
    "--solve_lbfgs_steps", help="steps for lbfgs", default=100, type=int
)
parser.add_argument(
    "--solve_lbfgs_stepsize", help="stepsize for lbfgs", default=1e-1, type=float
)
parser.add_argument(
    "--solve_adam_stepsize", help="stepsize for adam", default=1e-2, type=float
)
parser.add_argument(
    "--solve_sgd_stepsize", help="stepsize for adam", default=1e-2, type=float
)
parser.add_argument(
    "--ffn_layer_sizes",
    help="Layer sizes for feed forward net",
    default="[256,128,64,32]",
    type=str,
)
parser.add_argument("--drop_prob", default=0.0, type=float, help="dropout probability")
parser.add_argument("--bV_dim", default=5, type=int, help="side length of surrogate")
parser.add_argument(
    "--quadratic_scale",
    help="Scale energies by average of squared inputs",
    default=True,
    type=s2b,
)
parser.add_argument(
    "--fenics_scale",
    help="Scale energies by energy computed from small Fenics pde",
    default=False,
    type=s2b,
)
parser.add_argument("--log_scale", help="Log scale energies", type=s2b, default=True)

parser.add_argument(
    "--max_train_steps", help="Maximum training steps", type=int, default=int(1e7)
)
parser.add_argument(
    "--optimizer", help="adam or sgd or amsgrad", type=str, default="adam"
)
parser.add_argument(
    "--results_dir",
    help="Dir for tensorboard and other output",
    type=str,
    default="results",
)
parser.add_argument(
    "--data_name", help="Name of data run", type=str, default="bV5_big_new"
)
parser.add_argument(
    "--experiment_name", help="Name of experiment run", type=str, default="default"
)
parser.add_argument(
    "--reload_data", help="Reload data if found", default=True, type=s2b
)

parser.add_argument("--batch_size", help="Batch size", type=int, default=512)

parser.add_argument(
    "--nonlinearity",
    help="selu, elu, relu, swish, sigmoid, tanh",
    type=str,
    default="selu",
)
parser.add_argument(
    "--init", help="kaiming, xavier, orthogonal", type=str, default="xavier"
)

parser.add_argument(
    "--relaxation_parameter", default=0.1, type=float, help="relaxation parameter"
)
parser.add_argument(
    "--max_newton_iter", default=1000, type=int, help="maximum Newton iters"
)

parser.add_argument(
    "--metamaterial_mesh_size",
    default=16,
    type=int,
    help="N points along one dim in each cell. "
    " Overvelde&Bertoldi use about sqrt(1000)",
)
parser.add_argument(
    "--pore_radial_resolution",
    default=120,
    type=int,
    help=" Number of points around each pore",
)
parser.add_argument(
    "--n_cells", default=2, type=int, help="number cells on one side of ref volume"
)
parser.add_argument("--c1", default=0.0, type=float, help="c1")
parser.add_argument("--c2", default=0.0, type=float, help="c2")
parser.add_argument("--L0", default=None, type=float, help="metamaterial L0")
parser.add_argument("--porosity", default=0.5, type=float, help="metamaterial porosity")
parser.add_argument(
    "--young_modulus", help="young's modulus of base material", type=float, default=1.0
)
parser.add_argument(
    "--poisson_ratio", help="poisson's ratio of base material", type=float, default=0.49
)

parser.add_argument("--lr", help="Learning rate", type=float, default=3e-4)
parser.add_argument("--wd", help="Weight decay", type=float, default=0.0)
parser.add_argument(
    "--J_weight", help="Weight on Jacobian loss", type=float, default=32.0
)
parser.add_argument(
    "--H_weight", help="Weight on Jacobian loss", type=float, default=0.1
)
parser.add_argument(
    "--angle_magnitude", help="Decoupled angle and log magnitude loss for vecs",
    type=s2b, default=False
)
parser.add_argument(
    "--mag_weight", help="magnitude weight for decoupled angle and log magnitude loss for vecs",
    type=float, default=1.0
)
parser.add_argument(
    "--hess", help="Use Hvp training", type=s2b, default=True
)
parser.add_argument(
    "--clip_grad_norm", help="Norm for gradient clipping", type=float, default=0.01
)

parser.add_argument(
    "--weight_space_trajectory",
    help="Interpolate along deploy trajectory with even-size steps in "
    "weight space. Otherwise interpolate between the iterates, with equal "
    "interpolation distance between each iterate.",
    type=s2b,
    default=False,
)
