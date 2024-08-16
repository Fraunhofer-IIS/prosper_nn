import torch
from prosper_nn.utils import sensitivity_analysis
from prosper_nn.models.hcnn import HCNN


def test_sensitivity_analysis():
    in_features = 10
    out_features = 5
    batchsize = 3
    n_batches = 4
    model = torch.nn.Linear(in_features=in_features, out_features=out_features)
    data = torch.randn(n_batches, batchsize, in_features)
    sensi = sensitivity_analysis.sensitivity_analysis(
        model, data=data, output_neuron=(slice(0, batchsize), 0), batchsize=batchsize
    )
    assert isinstance(sensi, torch.Tensor)
    assert sensi.shape == torch.Size((batchsize * n_batches, in_features))


def test_calculate_sensitivity_analysis():
    in_features = 10
    out_features = 5
    batchsize = 3
    n_batches = 4
    model = torch.nn.Linear(in_features=in_features, out_features=out_features)
    data = torch.randn(n_batches, batchsize, in_features)
    sensi = sensitivity_analysis.calculate_sensitivity_analysis(
        model, data, output_neuron=(slice(0, batchsize), 0), batchsize=batchsize
    )
    sensi = sensi.reshape((sensi.shape[0], -1))
    assert isinstance(sensi, torch.Tensor)
    assert sensi.shape == torch.Size((batchsize * n_batches, in_features))


def test_plot_sensitivity_curve():
    in_features = 5
    samples = 10
    sensi = torch.randn(samples, in_features)
    sensitivity_analysis.plot_sensitivity_curve(sensi, output_neuron=1)


def test_analyse_temporal_sensitivity():
    n_features_Y = 3
    n_state_neurons = 5
    batchsize = 2
    past_horizon = 4
    forecast_horizon = 3
    task_nodes = [0, 1]

    model = HCNN(
        n_features_Y=n_features_Y,
        n_state_neurons=n_state_neurons,
        past_horizon=past_horizon,
        forecast_horizon=forecast_horizon,
    )
    data = torch.randn(past_horizon, batchsize, n_features_Y)

    sensi = sensitivity_analysis.analyse_temporal_sensitivity(
        model,
        data=data,
        task_nodes=task_nodes,
        n_future_steps=forecast_horizon,
        past_horizon=past_horizon,
        n_features=input_size,
    )
    assert isinstance(sensi, torch.Tensor)
    assert sensi.shape == torch.Size((len(task_nodes), forecast_horizon, n_features_Y))


def test_plot_analyse_temporal_sensitivity():
    n_features_Y = 3
    n_target_vars = 2
    forecast_horizon = 3
    target_var = [f"target_var_{i}" for i in range(n_target_vars)]
    features = [f"feat_{i}" for i in range(n_features_Y)]

    sensis = torch.randn(len(target_var), forecast_horizon, n_features_Y)
    sensitivity_analysis.plot_analyse_temporal_sensitivity(
        sensis,
        target_var,
        features,
        n_future_steps=forecast_horizon,
    )
