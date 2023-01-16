import torch.utils.data
from typing import Tuple  

# define dataset class
class GasEmissionDataset(torch.utils.data.Dataset):
    """Gas Emission Dataset."""

    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
     
        self.X = X
        self.Y = Y
        

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        x = self.X[idx]
        y = self.Y[idx]
        return x,y