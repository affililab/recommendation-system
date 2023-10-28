import torch
import torch.nn as nn
import torch.optim as optim

class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_size=50):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size, padding_idx=num_users-1)  # Add padding index for new users
        self.product_embedding = nn.Embedding(num_products, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)

    def forward(self, user_ids, product_ids):
        user_embedded = self.user_embedding(user_ids)
        product_embedded = self.product_embedding(product_ids)
        concatenated = torch.cat([user_embedded, product_embedded], dim=1)
        output = self.fc(concatenated)
        return torch.sigmoid(output)


def train_collaborative_filtering_model(user_product_interactions, num_users):
    # Extract user and product IDs
    user_ids = torch.LongTensor(user_product_interactions['user_ids'])
    product_ids = torch.LongTensor(user_product_interactions['product_ids'])
    interactions = torch.FloatTensor(user_product_interactions['interactions'])  # Binary: 0 or 1

    # Initialize collaborative filtering model
    model = CollaborativeFilteringModel(num_users=num_users + 1,
                                        num_products=max(product_ids) + 1)  # Add 1 for new users

    # Binary cross-entropy loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        predictions = model(user_ids, product_ids)
        loss = criterion(predictions.view(-1), interactions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


user_product_interactions = {
    'user_ids': [0, 1, 2, 3, 4],
    'product_ids': [0, 1, 2, 3, 4],
    'interactions': [1.0, 0.0, 1.0, 1.0, 0.0]  # Binary interactions: 1 for interaction, 0 for no interaction
}

if __name__ == '__main__':
    train_collaborative_filtering_model()