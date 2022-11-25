import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()


class State:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0


class RNN(nn.Module):
    def __init__(self, in_dim, latent_dim, output_dim, device):
        super().__init__()

        self.latent_dim = latent_dim

        self.lin = nn.Linear(in_dim + latent_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, output_dim)

        self.device = device

    def one_step(self, x, h):
        combined_data = torch.cat((x, h), dim=1)
        return torch.tanh(self.lin(combined_data))

    def forward(self, x):
        # Init first hidden state
        h_t = torch.zeros((x.shape[0], self.latent_dim)).to(self.device)

        for i in range(x.shape[1]):
            x_t = x[:, i, :]
            h_t = self.one_step(x_t, h_t)

        return torch.tanh(self.decoder(h_t))


class SampleMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        # Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        """
        Longueur en fonction de la longueur considérée des séquences.
        """
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self, i):
        """
        Transformation de l'index 1d vers une indexation 3d.
        Renvoie une séquence de longueur length et l'id de la station.
        """
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day, timeslot:(timeslot+self.length), station], station

    
class ForecastMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        # Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[0]
        # Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        """
        Longueur en fonction de la longueur considérée des séquences.
        """
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self, i):
        """
        Transformation de l'indexation 1d vers indexation 2d
        renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        """
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day, timeslot:(timeslot+self.length-1)], self.data[day,(timeslot+1):(timeslot+self.length)]

