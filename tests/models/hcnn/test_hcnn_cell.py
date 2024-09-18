import torch
from prosper_nn.models.hcnn.hcnn_cell import HCNNCell, PartialTeacherForcing


class TestPartialTeacherForcing:
    ptf = PartialTeacherForcing(p=0.5)

    def test_evaluation(self):
        self.ptf.eval()
        input = torch.randn((20, 1, 100))

        output = self.ptf(input)
        # fill dropped nodes
        output = torch.where(output == 0, input, output)
        assert (output == input).all()

    def test_train(self):
        self.ptf.train()
        input = torch.randn((20, 1, 100))

        output = self.ptf(input)
        # fill dropped nodes
        output = torch.where(output == 0, input, output)
        assert (output == input).all()


class TestHcnnCell:
    n_state_neurons = 10
    n_features_Y = 5
    batchsize = 7

    hcnn_cell = HCNNCell(
        n_state_neurons=n_state_neurons,
        n_features_Y=n_features_Y,
    )
    hcnn_cell.A.weight = torch.nn.Parameter(torch.ones_like(hcnn_cell.A.weight))
    state = 0.5 * torch.ones((batchsize, n_state_neurons))
    expectation = state[..., :n_features_Y]
    observation = torch.ones(batchsize, n_features_Y)

    def test_get_teacher_forcing_full_Y(self):
        self.hcnn_cell.ptf_dropout.p = 0
        output_, teacher_forcing_ = self.hcnn_cell.get_teacher_forcing_full_Y(
            self.observation, self.expectation
        )
        self.checks_get_teacher_forcing(output_, teacher_forcing_)

        ### with partial teacher forcing
        self.hcnn_cell.ptf_dropout.p = 0.5
        output_, teacher_forcing_ = self.hcnn_cell.get_teacher_forcing_full_Y(
            self.observation, self.expectation
        )

        # fill dropped nodes
        teacher_forcing_[..., : self.n_features_Y] = torch.where(
            teacher_forcing_[..., : self.n_features_Y] == 0,
            -0.5,
            teacher_forcing_[..., : self.n_features_Y],
        )

        self.checks_get_teacher_forcing(output_, teacher_forcing_)

    def test_get_teacher_forcing_partial_Y(self):
        self.hcnn_cell.ptf_dropout.p = 0
        output_, teacher_forcing_ = self.hcnn_cell.get_teacher_forcing_partial_Y(
            self.observation, self.expectation
        )
        self.checks_get_teacher_forcing(output_, teacher_forcing_)

        ### with partial teacher forcing
        self.hcnn_cell.ptf_dropout.p = 0.5
        output_, teacher_forcing_ = self.hcnn_cell.get_teacher_forcing_partial_Y(
            self.observation, self.expectation
        )
        # fill dropped nodes
        teacher_forcing_[..., : self.n_features_Y] = torch.where(
            teacher_forcing_[..., : self.n_features_Y] == 0,
            -0.5,
            teacher_forcing_[..., : self.n_features_Y],
        )
        output_ = torch.where(output_ == 0, -0.5, output_)
        self.checks_get_teacher_forcing(output_, teacher_forcing_)

    def checks_get_teacher_forcing(self, output_, teacher_forcing_):
        assert (output_ == -0.5 * torch.ones(self.batchsize, self.n_features_Y)).all()
        assert (teacher_forcing_[..., : self.n_features_Y] == -self.expectation).all()
        assert (teacher_forcing_[..., self.n_features_Y :] == 0).all()
        assert (
            (self.expectation - teacher_forcing_[..., : self.n_features_Y])
            == self.observation
        ).all()

    def test_forward(self):
        state_, output_ = self.hcnn_cell.forward(self.state)
        self.checks_forward(state_, output_)

        state_, output_ = self.hcnn_cell.forward(self.state, self.observation)
        self.checks_forward(state_, output_)

    def test_forward_past_horizon(self):
        state_, output_ = self.hcnn_cell.forward_past_horizon(
            self.state, self.observation, self.expectation
        )
        self.checks_forward(state_, output_)

    def test_forward_forecast_horizon(self):
        state_, output_ = self.hcnn_cell.forward_forecast_horizon(
            self.state, self.expectation
        )
        self.checks_forward(state_, output_)

    def checks_forward(self, state_, output_):
        assert state_.shape == torch.Size((self.batchsize, self.n_state_neurons))
        assert output_.shape == torch.Size((self.batchsize, self.n_features_Y))
        assert not (state_.isnan()).any()
        assert not (output_.isnan()).any()
