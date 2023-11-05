import math
import os
import sys
import faiss
import numpy as np
import pandas as pd
import pymongo
from pymongo.mongo_client import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, DataType, CollectionSchema
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from database import getPartnerPrograms, getUserPreferencesById

load_dotenv()

# Load SBERT-Model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')

def transformPartnerProgramsDataset(partnerprograms):
    df = pd.DataFrame(partnerprograms)
    selection_df = [
        '_id',
        'title',
        'description',
        'commissionInPercent',
        'commissionFixed',
        'rating',
        'reviews',
        'resolved_sources',
        'summary',
        'trackingLifetime',
        'trackingTypeSession',
        'semHints',
        'products',
        'directActivation'
    ]

    # Use DataFrame.get to handle the case where the column doesn't exist
    subset_df = pd.DataFrame({col: df.get(col, None) for col in selection_df})

    subset_df.loc[:, 'categories'] = df['resolved_categories'].apply(
        lambda x: [category.get('title') for category in x] if x else [])
    subset_df.loc[:, 'trackingTypes'] = df['resolved_trackingTypes'].apply(
        lambda x: [trackingTypes.get('title') for trackingTypes in x] if x else [])
    subset_df.loc[:, 'revenueType'] = df['resolved_revenueType'].apply(
        lambda x: [revenueType.get('title') for revenueType in x] if x else [])
    subset_df.loc[:, 'salaryModel'] = df['resolved_salaryModel'].apply(
        lambda x: [item.get('title') for item in x] if x else [])
    subset_df.loc[:, 'sources'] = df['resolved_sources'].apply(lambda x: [item.get('title') for item in x] if x else [])
    subset_df.loc[:, 'targetGroups'] = df['resolved_targetGroups'].apply(
        lambda x: [item.get('title') for item in x] if x else [])
    subset_df.loc[:, 'advertisementAssets'] = df['resolved_advertisementAssets'].apply(
        lambda x: [item.get('title') for item in x] if x else [])

    subset_df.loc[:, 'id'] = df['_id'].astype(str)
    return subset_df


def create_index(collection, field_name='embedding'):
    index_param = {
        "index_type": 'IVF_FLAT',
        "params": {"nlist": 1024},
        "metric_type": 'COSINE'}
    collection.create_index(field_name, index_param)
    print("\nCreated index:\n{}".format(collection.index().params))


def milvus_connect(collection_name='products'):
    connections.connect(
        alias="default",
        host=os.getenv('MILVUS_DATABASE_HOST'),
        port=os.getenv('MILVUS_DATABASE_PORT')
    )
    collection = Collection(name=collection_name)
    collection.load()
    return collection


def retrieve_embeddings_from_milvus(collection_name='products'):
    collection = milvus_connect(collection_name)

    # Perform a range query to retrieve all embeddings
    # results = collection.search(
    #     data=[create_embeddings_user(users)[0]],
    #     anns_field='embedding',
    #     param={'nprobe': 16},
    #     limit=10000
    # )

    # Load the collection
    # collection.load()

    # Check the number of entities in the collection
    num_entities = collection.num_entities
    print(f"Number of entities in the collection: {num_entities}")

    collection.load()

    result = collection.query(expr="id >= 0", output_fields=['mongodb_id', 'embedding'])

    return result



def vector_similarity_search_preferences_products(preferences_embedding):
    collection = milvus_connect()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
    results = collection.search(
        data=[preferences_embedding],
        anns_field="embedding",
        param=search_params,
        limit=10,
        expr=None,
        output_fields=['mongodb_id'],
        consistency_level="Strong"
    )

    results_elements = []
    for entry in results[0]:
        mongodb_id = entry.entity.get('mongodb_id')
        results_elements.append({'mongodb_id': mongodb_id, 'similarity_score': entry.distance})
    return results_elements


def handleEmbeddings(collection, embeddings, weights):
    ids = [item['mongodb_id'] for item in embeddings]
    embeddings_data = [item['embedding'] for item in embeddings]
    embeddings_weighted = np.array(embeddings_data, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)

    print(f"Length of embeddings list: {len(embeddings)}")
    print(f"Shape of embeddings_weighted before weighting: {embeddings_weighted.shape}")

    assert embeddings_weighted.shape[0] == len(
        embeddings), f"Unexpected batch size for remaining embeddings: {embeddings_weighted.shape[0]}"

    assert weights_np.shape[0] == len(embeddings), f"Unexpected length for weights: {weights_np.shape[0]}"

    weights_sum = sum(weights)
    normalized_weights = np.array([w / weights_sum for w in weights_np])

    embeddings_weighted *= normalized_weights[:, np.newaxis]


    print(f"Shape of embeddings_weighted after weighting: {embeddings_weighted.shape}")

    schema = collection.describe()
    print(schema)

    expected_dtype = DataType.FLOAT_VECTOR
    print(f"Expected DataType: {expected_dtype}")

    expected_dtype = DataType.FLOAT_VECTOR
    if embeddings_weighted.dtype != np.float32:
        embeddings_weighted = embeddings_weighted.astype(np.float32)
    embeddings_weighted = embeddings_weighted.tolist()

    print("embedding", embeddings_weighted[0])
    print("ids", ids)

    try:
        data = [
            ids,
            embeddings_weighted,
        ]

        collection.insert(data)
        collection.flush()
    except Exception as e:
        print(f"Error during insert operation: {e}")


def create_and_save_product_embeddings_to_milvus(data, collection_name='products', batch_size=1000):
    connections.connect(
        alias="default",
        host=os.getenv('MILVUS_DATABASE_HOST'),
        port=os.getenv('MILVUS_DATABASE_PORT')
    )

    if collection_name in utility.list_collections():
        collection = Collection(name=collection_name)
    else:
        dimension = model.encode("test").shape[0]
        field1 = FieldSchema(name='mongodb_id', dtype=DataType.VARCHAR, description="id of item in mongodb",
                             max_length=100)
        field2 = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description="float vector", dim=dimension,
                             is_primary=False)
        field3 = FieldSchema(name='id', dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
        schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
        collection = Collection(name=collection_name, data=None, schema=schema)
        create_index(collection)

    embeddings = []
    weights = []
    current_timestamp = int(datetime.utcnow().timestamp())

    for i, entry in data:
        weight = 1.0
        combined_data = ("Name:" + entry['title']
                         + ", Description: " + entry['description']
                         + ", categories: " + ",".join(entry['categories']))

        combined_data += ", Instant Accept: "
        if entry['directActivation']:
            weight += 0.2
            combined_data += "yes"
        else:
            combined_data += "no"

        commissionSettingsAvailable = False
        weight += (0.01 * len(entry["advertisementAssets"]))
        if (entry.get("trackingLifetime")
                and isinstance(entry["trackingLifetime"], (int, float))
                and not math.isnan(entry["trackingLifetime"])):
            weight += (0.01 * entry["trackingLifetime"])
        if (entry.get("commissionInPercent")
                and isinstance(entry["commissionInPercent"], (int, float))
                and not math.isnan(entry["commissionInPercent"])):
            weight += (2 * entry["commissionInPercent"])
            commissionSettingsAvailable = True
        else:
            if (isinstance(entry["commissionFixed"], (int, float))
                    and not math.isnan(entry["commissionFixed"])):
                weight += (0.5 * entry.get("commissionFixed", 0.0))
                commissionSettingsAvailable = True

        #lastUpdated newer ones higher weighted
        if 'lastUpdated' in entry:
            weight += 0.1 / (1 + (current_timestamp - entry['lastUpdated']))

        if (len(entry.get("products", []) )):
            #products implement to combined_data : is Array of strings
            combined_data += ", Products: " + ",".join(entry.get('products', []))

        #semAllowed : should be higher weighted if true
        if entry.get('semAllowed', False):
            weight += 0.2

        # TODO: semHints : is Array of strings should be included in combined data
        combined_data += ", SEM Hints: " + ",".join(entry.get('semHints', []))

        # TODO: trackingTypeSession : if true it should be lower weighted than when trackingLifetime is set
        if entry.get('trackingTypeSession', False):
            weight *= 0.9

        # TODO: trackingTypes : the more the better, array elements
        tracking_types = entry.get('trackingTypes', [])
        if isinstance(tracking_types, list):
            weight += 0.01 * len(tracking_types)

        # TODO: rating : higher is better
        rating = entry.get('rating', 0.0)
        if isinstance(rating, (int, float)) and not math.isnan(rating):
            print("rating", rating)
            weight += 200 * rating
            # TODO: reviews : more is better
            reviews = entry.get('reviews', 0.0)
            if isinstance(reviews, (int, float)) and not math.isnan(reviews):
                weight += 0.01 * reviews
        else:
            # reduce non ranked products
            weight *= 0.1
            print("weights reduced")

        # Adjust weight for empty description
        if 'description' not in entry or not entry['description']:
            if (entry['description'] == ""):
                weight = 0
        if not commissionSettingsAvailable:
            weight = 0

        embedding = model.encode(combined_data)
        embeddings.append({"mongodb_id": str(entry._id), "embedding": embedding})
        weights.append(weight)

        print(i, str(entry._id), entry.title, entry.rating, weight)

        if i % batch_size == 0 and i > 0:
            handleEmbeddings(collection, embeddings, weights)
            embeddings = []
            weights = []

    if embeddings:
        handleEmbeddings(collection, embeddings, weights)

    connections.disconnect("default")

    return embeddings


def create_product_embeddings(data, index_filename='embedding_index.index'):
    embeddings = []
    weights = []
    weight = 1.0
    for i, entry in data:
        weight_description = 1.0

        combined_data = ("Name:" + entry['title']
                         + ", Description: " + entry['description']
                         + ", categories: " + ",".join(entry['categories']))

        combined_data += ", Instant Accept: "
        if entry['directActivation']:
            weight += 0.2
            combined_data += "yes"
        else:
            combined_data += "no"

        weight += (0.01 * len(entry["advertisementAssets"]))
        if entry["trackingLifetime"]:
            weight += (0.01 * entry["trackingLifetime"])
        if entry["commissionInPercent"]:
            weight += (0.1 * entry["commissionInPercent"])
        else:
            weight += (0.2 * entry["commissionFixed"])

        # description empty
        if entry['description']:
            weight_description *= 0.5

        embedding = model.encode(combined_data)
        embeddings.append(embedding)
        weights.append(weight)

    embeddings_np = np.array(embeddings, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)

    embeddings_weighted = embeddings_np * weights_np[:, np.newaxis]

    index = faiss.IndexFlatL2(embeddings_weighted.shape[1])

    index.add(embeddings_weighted)

    faiss.write_index(index, index_filename)
    return embeddings


def create_embeddings_user(data):
    embeddings = []
    for entry in data:
        combined_data = "Interessen: " + ",".join(entry['preferences']) + ", Vorwissen: " + ",".join(
            entry['pre_knowledge'])
        embedding = model.encode(combined_data)
        embeddings.append(embedding)
    return embeddings


def create_embeddings_user_preferences(preferences):
    categories = ",".join(preferences)

    # weightings
    category_weight = 5.0
    # preference_weight = 1.0
    # tools_weight = 1.0
    # channels_weight = 1.0

    weighted_embedding = (
            category_weight * model.encode(categories)
        # preference_weight * model.encode(preferences_section) +
        # tools_weight * model.encode(tools_section) +
        # channels_weight * model.encode(channels_section)
    )
    return weighted_embedding


def create_embeddings_categories(categories):
    categories = ",".join(categories)

    # weightings
    category_weight = 2.0

    weighted_embedding = (
            category_weight * model.encode(categories)
    )
    return weighted_embedding


def create_embeddings_preferences(preferences):
    categories = ",".join(preferences['categories'])
    category_groups = ",".join(preferences['categories_groups'])
    preferences_section = ",".join(preferences['preferences'])
    tools_section = ",".join(preferences['tools_categories'])
    channels_section = ",".join(preferences['marketing_channels'])

    # weightings
    category_weight = 10.0
    category_groups_weight = 20.0
    preference_weight = 1.0
    tools_weight = 1.0
    channels_weight = 1.0

    # normalize weights
    total_weight = category_weight + category_groups_weight + preference_weight + tools_weight + channels_weight
    normalized_category_weight = category_weight / total_weight
    normalized_category_groups_weight = category_groups_weight / total_weight
    normalized_preference_weight = preference_weight / total_weight
    normalized_tools_weight = tools_weight / total_weight
    normalized_channels_weight = channels_weight / total_weight

    weighted_embedding = (
            normalized_category_weight * model.encode(categories) +
            normalized_category_groups_weight * model.encode(category_groups) +
            normalized_preference_weight * model.encode(preferences_section) +
            normalized_tools_weight * model.encode(tools_section) +
            normalized_channels_weight * model.encode(channels_section)
    )

    return weighted_embedding


def index():
    user_embeddings = create_embeddings_user([])
    products = retrieve_embeddings_from_milvus()
    embeddings_data = [item['embedding'] for item in products]

    if embeddings_data:
        print("results")

        similarities_matrix = util.pytorch_cos_sim(np.array(user_embeddings), embeddings_data)

        recommended_products = []

        user_similarity_scores = similarities_matrix[0].numpy()
        top_n_indices = np.argsort(user_similarity_scores)[::-1][:10]
        # Display the recommendations
        for product_idx in top_n_indices:
            product = products[product_idx]
            recommended_products.append(
                {'mongodb_id': product['mongodb_id'], 'similarity_score': user_similarity_scores[product_idx]})
        recommended_ids = [ObjectId(item['mongodb_id']) for item in recommended_products]
        recommended_items = getPartnerPrograms(recommended_ids)
        df = transformPartnerProgramsDataset(recommended_items)
        for product in recommended_products:
            desired_id = str(product['mongodb_id'])
            index = df.index[df['id'] == desired_id].tolist()[0]
            desired_element = df.iloc[index]
            print(
                f"  - {desired_element['title']} '{','.join(desired_element['categories'])}' with similarity score: {product['similarity_score']:.4f}")
    else:
        print("empty result")

def recommendation_wizzard(preferences):

    preferences_embedding = create_embeddings_preferences(preferences)

    recommended_items = vector_similarity_search_preferences_products(preferences_embedding)
    recommended_ids = [ObjectId(item['mongodb_id']) for item in recommended_items]
    recommended_items_db = getPartnerPrograms(recommended_ids)
    df = transformPartnerProgramsDataset(recommended_items_db)
    result = []
    for item in recommended_items:
        index = df.index[df['id'] == item['mongodb_id']].tolist()[0]
        desired_element = df.iloc[index]
        print(
            f"  - {item['mongodb_id']} {desired_element['title']} '{','.join(desired_element['categories'])}' with similarity score: {item['similarity_score']:.4f}")
        result.append({'mongodb_id': item['mongodb_id'], 'title': desired_element['title'],
                       'categories': desired_element['categories'], 'similarity_score': item['similarity_score']})
    return result


def recommendation_user(id):
    preferences = getUserPreferencesById(id)
    print(preferences)

    preferences_embedding = create_embeddings_user_preferences(preferences)

    recommended_items = vector_similarity_search_preferences_products(preferences_embedding)
    recommended_ids = [ObjectId(item['mongodb_id']) for item in recommended_items]
    recommended_items_db = getPartnerPrograms(recommended_ids)
    df = transformPartnerProgramsDataset(recommended_items_db)
    result = []
    for item in recommended_items:
        index = df.index[df['id'] == item['mongodb_id']].tolist()[0]
        desired_element = df.iloc[index]
        print(
            f"  - {item['mongodb_id']} {desired_element['title']} '{','.join(desired_element['categories'])}' with similarity score: {item['similarity_score']:.4f}")
        result.append({'mongodb_id': item['mongodb_id'], 'title': desired_element['title'],
                       'categories': desired_element['categories'], 'similarity_score': item['similarity_score']})
    return result


def recommendation_categories(categories):
    preferences_embedding = create_embeddings_categories(categories)

    recommended_items = vector_similarity_search_preferences_products(preferences_embedding)
    recommended_ids = [ObjectId(item['mongodb_id']) for item in recommended_items]
    recommended_items_db = getPartnerPrograms(recommended_ids)
    df = transformPartnerProgramsDataset(recommended_items_db)
    result = []
    for item in recommended_items:
        index = df.index[df['id'] == item['mongodb_id']].tolist()[0]
        desired_element = df.iloc[index]
        print(
            f"  - {item['mongodb_id']} {desired_element['title']} '{','.join(desired_element['categories'])}' with similarity score: {item['similarity_score']:.4f}")
        result.append({'mongodb_id': item['mongodb_id'], 'title': desired_element['title'],
                       'categories': desired_element['categories'], 'similarity_score': item['similarity_score']})
    return result


def store_embeddings():
    partnerprograms = getPartnerPrograms()
    df = transformPartnerProgramsDataset(partnerprograms)

    product_embeddings = create_product_embeddings(df.head(100).iterrows())


def store_embeddings_milvus():
    partnerprograms = getPartnerPrograms()
    df = transformPartnerProgramsDataset(partnerprograms)
    print(df['semHints'].dtypes)

    product_embeddings = create_and_save_product_embeddings_to_milvus(df.iterrows())


if __name__ == '__main__':
    store_embeddings_milvus()
