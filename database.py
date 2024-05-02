import os
import pymongo
from pymongo.mongo_client import MongoClient
import sys
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv()

def connect_to_mongodb():
    try:
        client = MongoClient(os.getenv('DB_MONGODB_CONNECTION'), connect=False)
        return client
    except pymongo.errors.ConfigurationError:
        print("An Invalid URI host error was received. Is your Atlas host name correct in your connection string?")
        sys.exit(1)

def getPartnerPrograms(ids=[]):
    db_client = connect_to_mongodb()
    partnerprograms = []
    if db_client:
        db = db_client[os.getenv('DB_MONGODB_DATABASE')]
        collection_partnerprograms = db['partnerprograms']

        # resolve relationships
        pipeline = [
            {
                "$lookup": {
                    "from": "trackingtypes",
                    "localField": "trackingTypes",
                    "foreignField": "_id",
                    "as": "resolved_trackingTypes"
                }
            },
            {
                "$lookup": {
                    "from": "revenuetypes",
                    "localField": "revenueType",
                    "foreignField": "_id",
                    "as": "resolved_revenueType"
                }
            },
            {
                "$lookup": {
                    "from": "salarymodels",
                    "localField": "salaryModel",
                    "foreignField": "_id",
                    "as": "resolved_salaryModel"
                }
            },
            {
                "$lookup": {
                    "from": "categories",
                    "localField": "categories",
                    "foreignField": "_id",
                    "as": "resolved_categories"
                }
            },
            {
                "$lookup": {
                    "from": "sources",
                    "localField": "sources.source",
                    "foreignField": "_id",
                    "as": "resolved_sources"
                }
            },
            {
                "$lookup": {
                    "from": "targetgroups",
                    "localField": "targetGroups",
                    "foreignField": "_id",
                    "as": "resolved_targetGroups"
                }
            },
            {
                "$lookup": {
                    "from": "advertisementassets",
                    "localField": "advertisementAssets",
                    "foreignField": "_id",
                    "as": "resolved_advertisementAssets"
                }
            }
        ]

        if len(ids):
            pipeline.append({
                "$match": {
                    "_id": {"$in": ids}
                }})

        cursor = collection_partnerprograms.aggregate(pipeline)
        partnerprograms = list(cursor)
        if len(ids):
            partnerprograms.sort(key=lambda x: ids.index(x["_id"]))
    return partnerprograms

def getUserPreferencesById(id):
    db_client = connect_to_mongodb()
    preferredCategories = []
    if db_client:
        db = db_client[os.getenv('DB_MONGODB_DATABASE')]
        users_collection = db["users"]
        categories_collection = db["categorygroups"]
        if not isinstance(id, ObjectId):
            id = ObjectId(id)
        user = users_collection.find_one({"_id": id})
        if user and 'preferred' in user:
            preferred = user['preferred']
            # Manually populate 'preferred.categories' using a separate query
            categories_cursor = categories_collection.find({"_id": {"$in": preferred['categories']}})
            preferredCategories = list(categories_cursor)
    return [category['title'] for category in preferredCategories]

def getUsersWithInteractions():
    db_client = connect_to_mongodb()
    users_with_interactions = []

    if db_client:
        db = db_client[os.getenv('DB_MONGODB_DATABASE')]
        users_collection = db["users"]

        cursor = users_collection.find({"partnerProgramInteractions": {"$exists": True}})

        for user in cursor:
            users_with_interactions.append(user)
    return users_with_interactions

def calculate_user_ratings(interactions_aggregated, max_interaction_weight=15):
    # Normalisieren Sie die aggregierten Interaktionsgewichte auf eine Skala von 0 bis 5
    for user_id, products in interactions_aggregated.items():
        for product_id, weight in products.items():
            normalized_weight = max(weight, 0)  # Verhindert negative Bewertungen
            rating = (normalized_weight / max_interaction_weight) * 5
            interactions_aggregated[str(user_id)][str(product_id)] = min(rating, 5)  # Stellt sicher, dass die Bewertung nicht über 5 steigt

    return interactions_aggregated


def get_user_product_interactions():
    client = pymongo.MongoClient(os.getenv('DB_MONGODB_CONNECTION'), connect=False)
    db = client[os.getenv('DB_MONGODB_DATABASE')]
    collection = db['users']
    products = db['partnerprograms']
    all_products = products.count()

    interactions = []
    interactions_aggregated = {}
    user_product_ratings = {}
    unique_user_ids = set()
    unique_product_ids = set()

    interaction_weights = {
        'viewed': 1,
        'noticed': 2,
        'unnoticed': -1,
        'campaign_added': 3,
        'removed_from_campaign': -2
    }

    max_interaction_weight = 15

    for user in collection.find():
        user_id = user['_id']
        unique_user_ids.add(str(user_id))
        interactions_aggregated[str(user_id)] = {}
        for interaction in user.get('partnerProgramInteractions', []):
            product_id = interaction['partnerProgram']
            unique_product_ids.add(str(product_id))

            weight = interaction_weights.get(interaction['type'], 1)

            interactions.append({
                'user_id': user_id,
                'product_id': product_id,
                'interaction': weight
            })

            if str(product_id) not in interactions_aggregated[str(user_id)]:
                interactions_aggregated[str(user_id)][str(product_id)] = 0
            interactions_aggregated[str(user_id)][str(product_id)] += weight

    for user_id, products in interactions_aggregated.items():
        if str(user_id) not in user_product_ratings:
            user_product_ratings[str(user_id)] = {}
        for product_id, weight in products.items():
            if str(product_id) not in user_product_ratings[user_id]:
                user_product_ratings[str(user_id)][str(product_id)] = 2.5
            normalized_weight = max(weight, 0)  # Verhindert negative Bewertungen
            rating = (normalized_weight / max_interaction_weight) * 5
            user_product_ratings[str(user_id)][str(product_id)] = min(rating, 5)  # Stellt sicher, dass die Bewertung nicht über 5 steigt

    num_products = len(unique_product_ids)
    num_users = len(unique_user_ids)

    print(f'user_product_ratings: {user_product_ratings}')

    return user_product_ratings, interactions, unique_user_ids, unique_product_ids, num_users, num_products, all_products
