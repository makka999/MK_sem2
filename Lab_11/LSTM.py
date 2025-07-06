import torch
import torch.nn as nn
import numpy as np

# ======= Dane wejściowe =======
text = ("Computer scientists and philosophers have since suggested that AI may become "
        "an existential risk to humanity if its rational capacities are not steered towards beneficial goals")

# Mapa znaków
chars = sorted(list(set(text)))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(chars)

# Dane jako indeksy
data_idx = [char2idx[ch] for ch in text]
input_seq = torch.tensor(data_idx[:-1]).unsqueeze(0)  # shape: (1, seq_len)
target_seq = torch.tensor(data_idx[1:]).unsqueeze(0)  # shape: (1, seq_len)

# ======= Model LSTM =======
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super(CharLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# ======= Trening =======
model = CharLSTM(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    output, _ = model(input_seq)
    loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or loss.item() < 0.01:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        if loss.item() < 0.01:
            break

# ======= Generowanie tekstu =======
def generate_text(model, start_char, length=200):
    model.eval()
    input_char = torch.tensor([[char2idx[start_char]]])
    hidden = None
    result = [start_char]

    for _ in range(length):
        output, hidden = model(input_char, hidden)
        prob = output[0, -1].softmax(dim=0)
        next_char_idx = torch.multinomial(prob, 1).item()
        result.append(idx2char[next_char_idx])
        input_char = torch.tensor([[next_char_idx]])

    return ''.join(result)

# ======= Przykład generowanego tekstu =======
print("\n=== Generowany tekst ===")
print(generate_text(model, start_char="C", length=200))
