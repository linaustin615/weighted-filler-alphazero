import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, n_colors=6, size=10):
        super().__init__()
        self.n_colors = n_colors

        #encode each action with shared local weights
        self.local = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        #compare action consequences with shared board features
        self.reason = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(33, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        #initialize residual at zero for a predictable greedy anchor
        nn.init.zeros_(self.policy_head[-1].weight)
        nn.init.zeros_(self.policy_head[-1].bias)
        self.gain_scale = nn.Parameter(torch.tensor(50.0))

        self.value_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(), #bound output to [-1, 1] for game outcome prediction
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        colors = x[:, :self.n_colors] #color planes
        context = x[:, self.n_colors:self.n_colors + 4] #territories, walls, and cell values
        captures = x[:, self.n_colors + 4: self.n_colors * 2 + 4] #capture planes

        expanded_context = context[:, None].expand(-1, self.n_colors, -1, -1, -1)

        #per-action input tensor; six planes
        action_inputs = torch.cat(
            (
                colors[:, :, None], #add singleton plane dimension
                captures[:, :, None],
                expanded_context
            ),
            dim=2 #join along plane dimension
        ).reshape(batch_size * self.n_colors, 6, height, width) #merge batch and color axes

        local_features = self.local(action_inputs).reshape(
            batch_size,
            self.n_colors,
            16,
            height,
            width
        )

        shared_features = local_features.mean(dim=1, keepdim=True).expand(-1, self.n_colors, -1, -1, -1) #board summary

        combined = torch.cat((local_features, shared_features), dim=2).reshape(
            batch_size * self.n_colors,
            32,
            height,
            width
        )

        tokens = self.reason(combined).mean(dim=(2, 3))
        tokens = tokens.reshape(
            batch_size,
            self.n_colors,
            32
        )

        values = x[:, self.n_colors + 3]
        weighted_gain = (
            captures * values[:, None]
        ).sum(dim=(2, 3))

        total_value = values.sum(
            dim=(1, 2)
        ).clamp_min(1.0)

        weighted_gain = weighted_gain / total_value[:, None]

        policy_input = torch.cat( #(batch_size, n_colors, 33)
            (tokens, weighted_gain[:, :, None]),
            dim=2
        )

        residual_logits = self.policy_head(policy_input).squeeze(-1) #(batch_size, colors)
        logits = residual_logits + self.gain_scale * weighted_gain

        value = self.value_head(tokens.mean(dim=1)) #(batch_size, 1)

        return logits, value
