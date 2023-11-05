import os

import torch
import torch.nn as nn
import torch.optim as optim
from bson import ObjectId

import database


class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_size=50):
        super(CollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size, padding_idx=num_users-1)
        self.product_embedding = nn.Embedding(num_products, embedding_size)
        self.fc = nn.Linear(embedding_size * 2, 1)

    def forward(self, user_ids, product_ids):
        user_embedded = self.user_embedding(user_ids)
        product_embedded = self.product_embedding(product_ids)
        concatenated = torch.cat([user_embedded, product_embedded], dim=1)
        output = self.fc(concatenated)
        return output

def save_model(model, num_users, num_products, path='collaborative_model.pth'):
    # Save the model with additional information about num_users and num_products
    torch.save({'model_state_dict': model.state_dict(), 'num_users': num_users, 'num_products': num_products}, path)

def load_model(path='collaborative_model.pth'):
    # Load the model and return it along with num_users and num_products
    checkpoint = torch.load(path)
    model = CollaborativeFilteringModel(num_users=checkpoint['num_users'], num_products=checkpoint['num_products'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['num_users'], checkpoint['num_products']

def train_collaborative_filtering_model(user_product_interactions, user_id_mapping, product_id_mapping):
    user_ids_tensor = torch.LongTensor(
        [user_id_mapping[str(interaction['user_id'])] for interaction in user_product_interactions])
    product_ids_tensor = torch.LongTensor(
        [product_id_mapping[str(interaction['product_id'])] for interaction in user_product_interactions])
    interactions_tensor = torch.FloatTensor([1 if interaction['interaction'] > 0 else 0 for interaction in user_product_interactions])

    model = CollaborativeFilteringModel(num_users=len(user_id_mapping),
                                        num_products=len(product_id_mapping))

    # Binary cross-entropy loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # Apply weights based on interaction types
    weights_tensor = torch.FloatTensor(
        [interaction['interaction'] for interaction in user_product_interactions])

    # Binary cross-entropy loss function with weights
    criterion = nn.BCEWithLogitsLoss(weight=weights_tensor)


    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        predictions = model(user_ids_tensor, product_ids_tensor)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()

        # Calculate loss with weighted BCE
        loss = criterion(predictions.view(-1), interactions_tensor)

        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    return model


def recommend_for_user(user_id, model, user_id_mapping, product_id_mapping):
    # Generate recommendations for a specific user
    user_id_str = str(user_id)
    user_index = user_id_mapping[user_id_str]

    # Get the number of products from the model
    num_products = model.product_embedding.weight.size(0)

    user_ids = torch.LongTensor([user_index] * num_products)
    product_ids = torch.arange(0, num_products)

    # Switch the model to evaluation mode
    model.eval()

    # Disable gradient computation for predictions
    with torch.no_grad():
        predictions = model(user_ids, product_ids)

    # Convert predictions to a list of tuples (product_id, predicted_interaction)
    recommendations = list(zip(product_ids.numpy(), predictions.view(-1).detach().numpy()))

    # Sort recommendations by predicted interaction (descending)
    recommendations.sort(key=lambda x: x[1], reverse=True)

    reverse_product_id_mapping = {v: k for k, v in product_id_mapping.items()}

    recommendations = [(reverse_product_id_mapping[product_index], reverse_product_id_mapping) for
                       product_index, predicted_interaction in recommendations]

    return recommendations

def handle_new_users(model, new_user_ids, user_id_mapping):
    # Initialize embeddings for new users
    new_user_mapping = {user_id: len(user_id_mapping) + i for i, user_id in enumerate(new_user_ids)}
    model.user_embedding.weight.data = torch.cat([
        model.user_embedding.weight.data,
        torch.randn(len(new_user_ids), model.user_embedding.embedding_dim)
    ])
    return new_user_mapping

def get_recommendations_for_user(user_id):
    # get interactions
    user_product_interactions, unique_user_ids, unique_product_ids, num_users, num_products = database.get_user_product_interactions()

    product_id_mapping = {product_id: i for i, product_id in enumerate(unique_product_ids)}
    user_id_mapping = {user_id: i for i, user_id in enumerate(unique_user_ids)}

    # Load the saved model
    model_path = 'collaborative_model.pth'
    if os.path.exists(model_path):
        model, num_users, num_products = load_model(path=model_path)
    else:
        # Train collaborative filtering model if the model doesn't exist
        model = train_collaborative_filtering_model(user_product_interactions, user_id_mapping, product_id_mapping)
        num_users, num_products = len(user_id_mapping), len(product_id_mapping)
        # Save the trained model
        save_model(model, num_users=num_users, num_products=num_products, path=model_path)

    # Check if the user_id is in the user_id_mapping
    if user_id not in user_id_mapping:
        # Handle new user: Update the model and mappings
        new_user_ids = [user_id]
        new_user_mapping = handle_new_users(model, new_user_ids, user_id_mapping)
        user_id_mapping.update(new_user_mapping)

        # Update product_ids_tensor with the new user
        user_ids_tensor = torch.LongTensor(
            [user_id_mapping[str(interaction['user_id'])] for interaction in user_product_interactions])
        product_ids_tensor = torch.LongTensor(
            [product_id_mapping[str(interaction['product_id'])] for interaction in user_product_interactions])
        interactions_tensor = torch.FloatTensor(
            [1 if interaction['interaction'] > 0 else 0 for interaction in user_product_interactions])

        # Train the model with the new user
        model = train_collaborative_filtering_model(user_product_interactions, user_id_mapping, product_id_mapping)
        num_users, num_products = len(user_id_mapping), len(product_id_mapping)
        # Save the updated model
        save_model(model, num_users=num_users, num_products=num_products, path=model_path)

    # Get recommendations for the specified user
    recommendations = recommend_for_user(user_id, model, user_id_mapping, product_id_mapping)

    recommended_ids = [ObjectId(product_id) for product_id, predicted_interaction in recommendations[:10]]
    recommended_items_db = database.getPartnerPrograms(recommended_ids)
    result = []
    for item in recommended_items_db:
        result.append({'mongodb_id': str(item['_id']), 'title': item['title']})
    return result

if __name__ == '__main__':
    user_id_to_recommend = "6481a1da1ab908a52518fff2"
    recommendations = get_recommendations_for_user(user_id_to_recommend)