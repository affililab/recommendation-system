import random

import pymongo
import pandas as pd
from pymilvus.orm import utility
from pymongo.mongo_client import MongoClient
import os
from dotenv import load_dotenv
import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss
from pymilvus import connections, Collection, FieldSchema, DataType, Milvus, CollectionSchema

load_dotenv()

# Load SBERT-Model
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')

users = [
    {
        "firstname": "Julian",
        "lastname": "Bertsch",
        "preferences": ["Sexual Happiness"],
        "pre_assets": ["Youtube-Channel", "Blog"],
        "pre_knowledge": ["Facebook Ads", "Google Ads", "SEO", "SEA"]
    }
]


def connect_to_mongodb():
    try:
        client = MongoClient(os.getenv('DB_MONGODB_CONNECTION'), connect=False)
        return client
    except pymongo.errors.ConfigurationError:
        print("An Invalid URI host error was received. Is your Atlas host name correct in your connection string?")
        sys.exit(1)

def getPartnerPrograms():
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
        cursor = collection_partnerprograms.aggregate(pipeline)
        partnerprograms = list(cursor)
    return partnerprograms

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

    subset_df = df[selection_df].copy()

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
    return subset_df

def retrieve_embeddings_from_milvus(collection_name='products', batch_size=1000):
    connections.connect(
        alias="default",
        host='localhost',
        port='19530'
    )

    collection = Collection(name=collection_name)

    all_ids = []
    offset = 0
    ids = collection.list_id_array(limit=offset + batch_size)
    while ids:
        all_ids.extend(ids)
        offset += batch_size
        ids = collection.list_id_array(limit=offset + batch_size)

    all_embeddings = []
    for batch_ids in all_ids:
        embeddings = collection.get_entity_by_id(batch_ids)
        all_embeddings.extend(embeddings)

    connections.disconnect("default")

    return np.array(all_embeddings, dtype=np.float32)

def handleEmbeddings(collection, embeddings, weights):
    embeddings_weighted = np.array(embeddings, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)

    # Debug prints
    print(f"Length of embeddings list: {len(embeddings)}")
    print(f"Shape of embeddings_weighted before weighting: {embeddings_weighted.shape}")

    # Ensure that the dimensions match for weighting
    assert embeddings_weighted.shape[0] == len(
        embeddings), f"Unexpected batch size for remaining embeddings: {embeddings_weighted.shape[0]}"

    # Ensure that the weights have the correct shape
    assert weights_np.shape[0] == len(embeddings), f"Unexpected length for weights: {weights_np.shape[0]}"

    # Apply weights to each embedding vector
    embeddings_weighted *= weights_np[:, np.newaxis]

    # Debug prints
    print(f"Shape of embeddings_weighted after weighting: {embeddings_weighted.shape}")

    # Explicitly convert the entire batch to float32
    # embeddings_weighted = embeddings_weighted.astype(np.float32)

    schema = collection.describe()
    print(schema)

    expected_dtype = DataType.FLOAT_VECTOR
    print(f"Expected DataType: {expected_dtype}")

    expected_dtype = DataType.FLOAT_VECTOR
    if embeddings_weighted.dtype != np.float32:
        embeddings_weighted = embeddings_weighted.astype(np.float32)
    embeddings_weighted = embeddings_weighted.tolist()

    try:
        data = [
            embeddings_weighted,
        ]

        collection.insert(data)
        collection.flush()
    except Exception as e:
        print(f"Error during insert operation: {e}")


def create_and_save_product_embeddings_to_milvus(data, collection_name='products', batch_size=1000):
    connections.connect(
        alias="default",
        host='localhost',
        port='19530'
    )

    if collection_name in utility.list_collections():
        collection = Collection(name=collection_name)
    else:
        dimension = model.encode("test").shape[0]
        field1 = FieldSchema(name='id', dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
        field2 = FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description="float vector", dim=dimension,
                             is_primary=False)
        schema = CollectionSchema(fields=[field1, field2], description="collection description")
        collection = Collection(name=collection_name, data=None, schema=schema, properties={"collection.ttl.seconds": 15})

    embeddings = []
    weights = []
    weight = 1.0

    for i, entry in data:
        weight_interests = 2.0
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
        # if entry["trackingLifetime"]:
        #     weight += (0.01 * entry["trackingLifetime"])
        # if entry["commissionInPercent"]:
        #     weight += (0.1 * entry["commissionInPercent"])
        # else:
        #     weight += (0.2 * entry["commissionFixed"])

        weight += weight_interests + weight_description

        # print(entry['title'] + " " + str(weight))

        print(i)

        record_id = i
        embedding = model.encode(combined_data)
        embeddings.append(embedding)
        weights.append(weight)

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
        weight_interests = 2.0
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
        # if entry["trackingLifetime"]:
        #     weight += (0.01 * entry["trackingLifetime"])
        # if entry["commissionInPercent"]:
        #     weight += (0.1 * entry["commissionInPercent"])
        # else:
        #     weight += (0.2 * entry["commissionFixed"])

        weight += weight_interests + weight_description

        # print(entry['title'] + " " + str(weight))

        embedding = model.encode(combined_data)
        embeddings.append(embedding)
        weights.append(weight)
        print(i)

    # Initialize the Faiss index
    embeddings_np = np.array(embeddings, dtype=np.float32)
    weights_np = np.array(weights, dtype=np.float32)

    # Concatenate embeddings with weights
    embeddings_weighted = embeddings_np * weights_np[:, np.newaxis]

    index = faiss.IndexFlatL2(embeddings_weighted.shape[1])

    # Add embeddings to the index
    index.add(embeddings_weighted)

        # Save the index to a file
    faiss.write_index(index, index_filename)
    return embeddings

def create_embeddings_user(data):
    embeddings = []
    for entry in data:
        combined_data = "Interessen: " + ",".join(entry['preferences']) + ", Vorwissen: " + ",".join(entry['pre_knowledge'])
        embedding = model.encode(combined_data)
        embeddings.append(embedding)
    return embeddings

def index(index_filename='embedding_index.index'):
    partnerprograms = getPartnerPrograms()
    df = transformPartnerProgramsDataset(partnerprograms)
    user_embeddings = create_embeddings_user(users)

    # Load the Faiss index
    faiss_index = faiss.read_index(index_filename)

    full_embedding = np.zeros((faiss_index.ntotal, faiss_index.d), dtype=np.float32)


    print(faiss_index.ntotal)

    for i in range(faiss_index.ntotal):
        full_embedding[i] = faiss_index.reconstruct(i)

    similarities_matrix = util.pytorch_cos_sim(np.array(user_embeddings), np.array(full_embedding))


    #Iterate through users and products to print similarity scores
    for user_idx, user in enumerate(users):
        user_similarity_scores = similarities_matrix[user_idx].numpy()
        top_n_indices = np.argsort(user_similarity_scores)[::-1][:5]
        # Display the recommendations
        print(f"Top 5 Recommendations for User '{user['firstname']} {user['lastname']}':")
        for product_idx in top_n_indices:
            product = df.iloc[product_idx]
            print(
                f"  - {product['title']} '{','.join(product['categories'])}' with similarity score: {user_similarity_scores[product_idx]:.4f}")

def store_embeddings():
    partnerprograms = getPartnerPrograms()
    df = transformPartnerProgramsDataset(partnerprograms)

    product_embeddings = create_product_embeddings(df.iterrows())
def store_embeddings_milvus():
    partnerprograms = getPartnerPrograms()
    df = transformPartnerProgramsDataset(partnerprograms)

    product_embeddings = create_and_save_product_embeddings_to_milvus(df.head(100).iterrows())


store_embeddings_milvus()