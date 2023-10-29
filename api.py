from flask import Flask, request
from flask_restful import Resource, Api

import collaborative
import contentbased

app = Flask(__name__)
api = Api(app)

class RecommendationSystem(Resource):
    def post(self):
        data = request.get_json()
        print(data)
        recommends = contentbased.recommendation_wizzard(data)
        # recommends = main.index_vector_search_on_milvus()
        return recommends

class RecommendationSystemUserPreferences(Resource):
    def post(self):
        data = request.get_json()
        print(data['userId'])
        recommends = contentbased.recommendation_user(data['userId'])
        return recommends

class RecommendationSystemCategories(Resource):
    def post(self):
        data = request.get_json()
        recommends = contentbased.recommendation_categories(data['categories'])
        return recommends

class CollaborativeFilteredRecommendation(Resource):
    def post(self):
        data = request.get_json()
        recommends = collaborative.get_recommendations_for_user(data['userId'])
        return recommends



api.add_resource(RecommendationSystem, '/recommendation-system')
api.add_resource(RecommendationSystemUserPreferences, '/recommendation-user')
api.add_resource(RecommendationSystemCategories, '/recommendation-categories')
api.add_resource(CollaborativeFilteredRecommendation, '/collaborative-user')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
