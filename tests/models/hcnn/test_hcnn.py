import torch
from prosper_nn.models.hcnn import HCNN
import pytest


class TestHcnn:
    @pytest.mark.parametrize(
    "n_state_neurons, n_features_Y, past_horizon, forecast_horizon, batchsize, sparsity, teacher_forcing, backward_full_Y, ptf_in_backward",
    [
        (10, 4, 5, 2, 1, 0, 1, True, True),
        (1, 1, 1, 1, 1, 0, 1, True, True),
        (10, 4, 5, 2, 1, 0.5, 1, True, True),
        (10, 4, 5, 2, 1, 0, 0.5, False, True),
        (10, 4, 5, 2, 1, 0, 0.5, False, False),
        (10, 4, 5, 2, 1, 0, 0.5, True, False),
        (10, 4, 5, 2, 1, 0, 1, True, True),
    ],
    	)
    def test_forward(self, n_state_neurons, n_features_Y, past_horizon, forecast_horizon, batchsize, sparsity, teacher_forcing, backward_full_Y, ptf_in_backward):
        hcnn = HCNN(
            n_state_neurons=n_state_neurons,
            n_features_Y=n_features_Y,
            past_horizon=past_horizon,
            forecast_horizon=forecast_horizon,
            sparsity=sparsity,
            teacher_forcing=teacher_forcing,
            backward_full_Y=backward_full_Y,
            ptf_in_backward=ptf_in_backward,
        )
        observation = torch.zeros(past_horizon, batchsize, n_features_Y)
        output_ = hcnn(observation)

        assert output_.shape == torch.Size((past_horizon + forecast_horizon, batchsize, n_features_Y))
        assert isinstance(output_, torch.Tensor)
        assert not (output_.isnan()).any()

    @pytest.mark.parametrize(
    "n_state_neurons, past_horizon, forecast_horizon, batchsize, sparsity, teacher_forcing, backward_full_Y, ptf_in_backward",
    [
        (5, 50, 5, 1, 0, 1, True, True),

    ],
    	)
    def test_train(self, n_state_neurons, past_horizon, forecast_horizon, batchsize, sparsity, teacher_forcing, backward_full_Y, ptf_in_backward):
        n_features_Y = 1
        n_epochs = 10000

        hcnn = HCNN(
            n_state_neurons=n_state_neurons,
            n_features_Y=n_features_Y,
            past_horizon=past_horizon,
            forecast_horizon=forecast_horizon,
            sparsity=sparsity,
            teacher_forcing=teacher_forcing,
            backward_full_Y=backward_full_Y,
            ptf_in_backward=ptf_in_backward,
        )
        observation = torch.zeros(past_horizon, batchsize, n_features_Y)
        observation = torch.sin(torch.linspace(0.5, 10 * torch.pi, past_horizon + forecast_horizon))
        observation = observation.unsqueeze(1).unsqueeze(1)

        optimizer = torch.optim.Adam(hcnn.parameters(), lr=0.001)
        target = torch.zeros_like(observation[:past_horizon])
        loss_fct = torch.nn.MSELoss()

        start_weight = hcnn.HCNNCell.A.weight.clone()

        for epoch in range(n_epochs):
            output_ = hcnn(observation[:past_horizon])
            loss = loss_fct(output_[:past_horizon], target)
            loss.backward()
            assert hcnn.HCNNCell.A.weight.grad is not None
            optimizer.step()
            if epoch == 1:
                start_loss = loss.detach()
                assert (hcnn.HCNNCell.A.weight != start_weight).all()
            hcnn.zero_grad()

        forecast = hcnn(observation[:past_horizon])[past_horizon:]
        assert loss < start_loss
        assert torch.isclose(observation[past_horizon:], forecast, atol=1).all()

    @pytest.mark.parametrize("teacher_forcing, decrease_teacher_forcing, result", [(1, 0, 1), (1, 0.2, 0.8), (0, 0.1, 0)],)
    def test_adjust_teacher_forcing(self, teacher_forcing, decrease_teacher_forcing, result):
        hcnn = HCNN(
            n_state_neurons=10,
            n_features_Y=2,
            past_horizon=10,
            forecast_horizon=5,
            teacher_forcing=teacher_forcing,
            decrease_teacher_forcing=decrease_teacher_forcing)
        hcnn.adjust_teacher_forcing()
        assert hcnn.HCNNCell.teacher_forcing == result
        assert hcnn.teacher_forcing == result
