import pymongo
import pandas as pd
from pymongo.mongo_client import MongoClient
import os
from dotenv import load_dotenv
import sys
from sentence_transformers import SentenceTransformer, util
import numpy as np

load_dotenv()


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
            },
            # Füge weitere $lookup-Stufen für andere referenzierte Felder hinzu
        ]
        cursor = collection_partnerprograms.aggregate(pipeline)
        partnerprograms = list(cursor)
    return partnerprograms

def transformPartnerProgramsDataset(partnerprograms):
    df = pd.DataFrame(partnerprograms)
    selection_df = [
        'title',
        'description',
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

def index():
    partnerprograms = getPartnerPrograms()
    df = transformPartnerProgramsDataset(partnerprograms)

    # TODO: use dataset to get recommendations

index()