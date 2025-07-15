import torch
import sys
import gpytorch
import math
import argparse
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.utils.gpytorch_modules import (
    get_covar_module_with_dim_scaled_prior,
    get_gaussian_likelihood_with_gamma_prior,
    get_gaussian_likelihood_with_lognormal_prior,
    get_matern_kernel_with_gamma_prior)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO, ComputationAwareELBO

sys.path.append('../src')

from set_seed import set_seed
from models.svgp import SVGPModel
from models.ca_gp import CaGP

from trainers.ca_gp_trainer import LassoDNALogEICaGPSlidingWindowTrainer, OsmbLogEICaGPSlidingWindowTrainer
from trainers.exact_gp_trainer import LassoDNALogEIExactGPTrainer, OsmbLogEIExactGPTrainer
from trainers.svgp_trainer import LassoDNALogEISVGPTrainer, OsmbLogEISVGPTrainer
from models.exact_gp import ExactGPModel

mock_args = dict(
    epochs=200,
    eulbo_epochs=30,
    device='cuda',
    batch_size=1,
    dropout_prob=0.3,
    save_dir='./',  # Set to your desired path
    data_dir='./',  # Set to your desired path
    optimizer='adam',
    num_repeats=3,
    seed=42,
    learning_rate=1e-2,
    grad_clip=-1,
    ca_gp_actions_learning_rate=1e-2,
    svgp_inducing_point_learning_rate=1e-2,
    max_oracle_calls=101,
    trainer_type='lunar_ei_ca_gp',
    kernel_type='matern_5_2',
    kernel_likelihood_prior='none',
    use_ard_kernel=False,
    ca_gp_init_mode='random',
    norm_data=True,
    turn_off_wandb=True,
    use_analytic_acq_func=True,
    early_stopping_threshold=20,
    num_initial_points=100,
    update_train_size=100,
    num_inducing_points=100,
    proj_dim_ratio=0.5,
    static_proj_dim=101,
    debug=True,
    enable_raasp=True,
    path_to_selfies_vae_files='../src/tasks/utils/selfies_vae/')

dataset_trainer_mapping = {
    'osmb': {
        'svgp': OsmbLogEISVGPTrainer,
        'exact': OsmbLogEIExactGPTrainer,
        'cagp': OsmbLogEICaGPSlidingWindowTrainer
    },
    'lasso': {
        'svgp': LassoDNALogEISVGPTrainer,
        'exact': LassoDNALogEIExactGPTrainer,
        'cagp': LassoDNALogEICaGPSlidingWindowTrainer
    }
}


def train_exact(dataset):
    exact_gp_trainer = dataset_trainer_mapping[dataset]['exact'](
        optimizer_type=torch.optim.Adam,
        criterion=None,
        tracker=None,
        **mock_args)

    # simulate first round of training
    # train_x_exact, train_y_exact = exact_gp_trainer.initialize_data()
    train_x_exact, train_y_exact = torch.load(
        '../src/train_x_exact_101.pt'), torch.load(
            '../src/train_y_exact_101.pt')
    print(f'initial y max: {train_y_exact.max().item()}')
    exact_gp_trainer.train_y_mean = train_y_exact.mean()
    exact_gp_trainer.train_y_std = train_y_exact.std()
    if exact_gp_trainer.train_y_std == 0:
        exact_gp_trainer.train_y_std = 1

    for i in range(exact_gp_trainer.max_oracle_calls -
                   exact_gp_trainer.num_initial_points):
        if exact_gp_trainer.norm_data:
            # get normalized train y
            model_train_y_exact = (
                train_y_exact -
                exact_gp_trainer.train_y_mean) / exact_gp_trainer.train_y_std
        else:
            model_train_y_exact = train_y_exact

        # Init exact gp model
        if exact_gp_trainer.use_ard_kernel:
            ard_num_dims = train_x_exact.shape[-1]
        else:
            ard_num_dims = None

        if exact_gp_trainer.kernel_likelihood_prior == 'gamma':
            covar_module = get_matern_kernel_with_gamma_prior(
                ard_num_dims=ard_num_dims)
            likelihood = get_gaussian_likelihood_with_gamma_prior()
        elif exact_gp_trainer.kernel_likelihood_prior == 'lognormal':
            covar_module = get_covar_module_with_dim_scaled_prior(
                ard_num_dims=ard_num_dims, use_rbf_kernel=False)
            likelihood = get_gaussian_likelihood_with_lognormal_prior()
        else:
            if exact_gp_trainer.kernel_type == 'rbf':
                base_kernel = gpytorch.kernels.RBFKernel()
            elif exact_gp_trainer.kernel_type == 'matern_3_2':
                base_kernel = gpytorch.kernels.MaternKernel(1.5)
            else:
                base_kernel = gpytorch.kernels.MaternKernel(2.5)

            covar_module = gpytorch.kernels.ScaleKernel(base_kernel)
            likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
                exact_gp_trainer.device)

        # exact_gp_trainer.model = SingleTaskGP(
        #     train_x_exact,
        #     model_train_y_exact,
        #     covar_module=covar_module,
        #     likelihood=likelihood,
        #     outcome_transform=None
        # ).to(exact_gp_trainer.device)

        exact_gp_trainer.model = ExactGPModel(
            train_x_exact,
            model_train_y_exact.squeeze(),
            covar_module=covar_module,
            likelihood=likelihood,
        ).to(exact_gp_trainer.device)
        exact_gp_mll = ExactMarginalLogLikelihood(
            exact_gp_trainer.model.likelihood, exact_gp_trainer.model)

        # fit model to data
        # mll = fit_gpytorch_mll(exact_gp_mll)
        # exact_gp_trainer.optimizer = torch.optim.LBFGS(
        #     exact_gp_trainer.model.parameters(), lr=1e-1)
        exact_gp_trainer.optimizer = torch.optim.Adam(
            exact_gp_trainer.model.parameters(), lr=1e-1)

        train_loader = exact_gp_trainer.generate_dataloaders(
            train_x=train_x_exact, train_y=model_train_y_exact.squeeze())

        final_loss, epochs_trained = exact_gp_trainer.train_model(
            train_loader, exact_gp_mll)
        exact_gp_trainer.model.eval()

        # get train rmse
        train_rmse = exact_gp_trainer.eval(train_x_exact,
                                           model_train_y_exact.squeeze())
        train_nll = exact_gp_trainer.compute_nll(train_x_exact,
                                                 model_train_y_exact.squeeze(),
                                                 exact_gp_mll)
        print(f'train_rmse: {train_rmse:.3f}, train_nll: {train_nll:.3f}')
        if exact_gp_trainer.kernel_likelihood_prior == 'lognormal':
            outputscale = torch.tensor([1])
            raw_lengthscale = exact_gp_trainer.model.covar_module.raw_lengthscale
            constraint = exact_gp_trainer.model.covar_module.raw_lengthscale_constraint
            lengthscale = constraint.transform(raw_lengthscale)
        else:
            raw_outputscale = exact_gp_trainer.model.covar_module.raw_outputscale
            constraint = exact_gp_trainer.model.covar_module.raw_outputscale_constraint
            outputscale = constraint.transform(raw_outputscale)

            raw_lengthscale = exact_gp_trainer.model.covar_module.base_kernel.raw_lengthscale
            constraint = exact_gp_trainer.model.covar_module.base_kernel.raw_lengthscale_constraint
            lengthscale = constraint.transform(raw_lengthscale)

        print(
            f'outputscale: {outputscale.item():.3f}, lengthscale: {torch.mean(lengthscale).item() if mock_args["use_ard_kernel"] else lengthscale.item():.3f}, noise: {exact_gp_trainer.model.likelihood.noise.item():.3f}'
        )

        # check acquired data point
        x_next_exact = exact_gp_trainer.data_acquisition_iteration(
            exact_gp_trainer.model, model_train_y_exact,
            train_x_exact).to(exact_gp_trainer.device)

        # Evaluate candidates
        y_next_exact = exact_gp_trainer.task(x_next_exact)
        print(f'y next {y_next_exact.item()}')

        train_x_exact = torch.cat((train_x_exact, x_next_exact), dim=-2)
        train_y_exact = torch.cat((train_y_exact, y_next_exact), dim=-2)


def train_cagp(dataset):
    # for repeatability
    set_seed(mock_args['seed'])

    cagp_trainer = dataset_trainer_mapping[dataset]['cagp'](
        optimizer_type=torch.optim.Adam, tracker=None, **mock_args)
    # train_x_cagp, train_y_cagp = cagp_trainer.initialize_data()
    train_x_cagp, train_y_cagp = torch.load(
        '../src/train_x_exact_101.pt'), torch.load(
            '../src/train_y_exact_101.pt')

    # log initial y_max
    print(f'initial y max: {train_y_cagp.max().item()}')

    cagp_trainer.train_y_mean = train_y_cagp.mean()
    cagp_trainer.train_y_std = train_y_cagp.std()
    if cagp_trainer.train_y_std == 0:
        cagp_trainer.train_y_std = 1

    if cagp_trainer.static_proj_dim != -1:
        proj_dim = cagp_trainer.static_proj_dim
    else:
        proj_dim = int(cagp_trainer.proj_dim_ratio * train_x_cagp.size(0))

    if cagp_trainer.norm_data:
        # get normalized train y
        model_train_y_cagp = (train_y_cagp - cagp_trainer.train_y_mean
                              ) / cagp_trainer.train_y_std
    else:
        model_train_y_cagp = train_y_cagp

    cagp_trainer.model = CaGP(
        train_inputs=train_x_cagp,
        train_targets=model_train_y_cagp.squeeze(),
        projection_dim=proj_dim,
        likelihood=GaussianLikelihood().to(cagp_trainer.device),
        kernel_type=cagp_trainer.kernel_type,
        init_mode=cagp_trainer.ca_gp_init_mode,
        kernel_likelihood_prior=cagp_trainer.kernel_likelihood_prior,
        use_ard_kernel=cagp_trainer.use_ard_kernel).to(cagp_trainer.device)

    # init_lengthscale = math.sqrt(train_x_cagp.size(1)) / 10
    # cagp_trainer.model.covar_module.base_kernel.lengthscale = init_lengthscale

    if cagp_trainer.norm_data:
        # get normalized train y
        model_train_y_cagp = (train_y_cagp - cagp_trainer.train_y_mean
                              ) / cagp_trainer.train_y_std
    else:
        model_train_y_cagp = train_y_cagp

    update_x_cagp = train_x_cagp
    update_y_cagp = model_train_y_cagp.squeeze()

    action_params = [
        p for name, p in cagp_trainer.model.named_parameters()
        if 'action' in name
    ]
    others = [
        p for name, p in cagp_trainer.model.named_parameters()
        if 'action' not in name
    ]

    # cagp_trainer.optimizer = torch.optim.LBFGS(cagp_trainer.model.parameters(),
    #                                            lr=5e-1)

    cagp_trainer.optimizer = torch.optim.Adam([{
        'params': others
    }, {
        'params': action_params,
        'lr': 1
    }],
                                              lr=1)

    mll = ComputationAwareELBO(cagp_trainer.model.likelihood,
                               cagp_trainer.model)
    exact_mll = ExactMarginalLogLikelihood(cagp_trainer.model.likelihood,
                                           cagp_trainer.model)

    train_loader = cagp_trainer.generate_dataloaders(train_x=update_x_cagp,
                                                     train_y=update_y_cagp)

    final_loss, epochs_trained = cagp_trainer.train_model(train_loader, mll)

    # calc gradients of actions
    total_norm = 0.0
    for p in action_params:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    print(f'action norm: {total_norm:.3f}')
    cagp_trainer.model.eval()

    train_rmse = cagp_trainer.eval(train_x_cagp, model_train_y_cagp)
    train_nll = cagp_trainer.compute_nll(train_x_cagp,
                                         model_train_y_cagp.squeeze(),
                                         exact_mll)
    print(f'train_rmse: {train_rmse:.3f}, train_nll: {train_nll:.3f}')

    if cagp_trainer.kernel_likelihood_prior == 'lognormal':
        outputscale = torch.tensor([1])
        raw_lengthscale = cagp_trainer.model.covar_module.raw_lengthscale
        constraint = cagp_trainer.model.covar_module.raw_lengthscale_constraint
        lengthscale = constraint.transform(raw_lengthscale)
    else:
        raw_outputscale = cagp_trainer.model.covar_module.raw_outputscale
        constraint = cagp_trainer.model.covar_module.raw_outputscale_constraint
        outputscale = constraint.transform(raw_outputscale)

        raw_lengthscale = cagp_trainer.model.covar_module.base_kernel.raw_lengthscale
        constraint = cagp_trainer.model.covar_module.base_kernel.raw_lengthscale_constraint
        lengthscale = constraint.transform(raw_lengthscale)
    print(
        f'outputscale: {outputscale.item():.3f}, lengthscale: {torch.mean(lengthscale).item() if mock_args["use_ard_kernel"] else lengthscale.item():.3f}, noise: {cagp_trainer.model.likelihood.noise.item():.3f}'
    )
    # check acquired data point
    x_next_cagp = cagp_trainer.data_acquisition_iteration(
        cagp_trainer.model, model_train_y_cagp,
        train_x_cagp).to(cagp_trainer.device)

    # Evaluate candidates
    y_next_cagp = cagp_trainer.task(x_next_cagp)
    print(f'y next {y_next_cagp.item()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Exact GP or CaGP training loop')
    parser.add_argument('--model_type',
                        choices=['exact', 'cagp'],
                        required=True,
                        help='Choose which GP model to train')
    parser.add_argument('--dataset',
                        choices=['osmb', 'lasso'],
                        default='osmb',
                        help='Dataset to use')
    args = parser.parse_args()

    # for repeatability
    set_seed(mock_args['seed'])

    # need this precision for gp fitting
    torch.set_default_dtype(torch.float64)
    # set default device for cagp
    torch.set_default_device(torch.device(mock_args['device']))

    if args.model_type == 'exact':
        train_exact(args.dataset)
    elif args.model_type == 'cagp':
        train_cagp(args.dataset)
