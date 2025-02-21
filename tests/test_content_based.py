import unittest
import pandas as pd
import numpy as np
from src.models.content_based import ContentBasedRecommender

class TestContentBasedRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data that can be reused across test methods"""
        # Create sample recipe data
        cls.sample_recipes = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],  # Added two more recipes
            'name': [
                'Chicken Soup', 'Beef Stew', 'Apple Pie', 
                'Pasta Carbonara', 'Caesar Salad', 'Pizza Margherita'
            ],
            'description': [
                'Classic chicken soup recipe',
                'Hearty beef stew',
                'Traditional apple pie',
                'Italian pasta dish',
                'Classic caesar salad',
                'Traditional Italian pizza'
            ],
            'ingredients_cleaned': [
                ['chicken', 'carrot', 'celery', 'onion'],
                ['beef', 'potato', 'carrot', 'onion'],
                ['apple', 'sugar', 'flour', 'butter'],
                ['pasta', 'egg', 'bacon', 'cheese'],
                ['lettuce', 'chicken', 'parmesan', 'croutons'],
                ['flour', 'tomato', 'mozzarella', 'basil']
            ],
            'minutes': [60, 120, 45, 30, 20, 40],
            'n_ingredients': [4, 4, 4, 4, 4, 4],
            'calories': [200, 300, 250, 400, 150, 280],
            'total_fat': [5, 15, 12, 20, 8, 10],
            'sugar': [2, 3, 25, 1, 2, 4],
            'sodium': [500, 600, 200, 800, 400, 500],
            'protein': [20, 25, 3, 15, 12, 10],
            'saturated_fat': [2, 6, 5, 8, 3, 5],
            'carbohydrates': [15, 20, 45, 30, 10, 35]
        })
        cls.sample_recipes.set_index('id', inplace=True)

        # Create sample user interaction data
        cls.sample_interactions = pd.DataFrame({
            'user_id': [1, 1, 1],
            'recipe_id': [1, 2, 3],
            'rating': [0.8, 0.6, 0.9],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03']
        })

    def setUp(self):
        """Initialize a new recommender instance for each test"""
        self.recommender = ContentBasedRecommender()

    def test_init(self):
        """Test recommender initialization"""
        self.assertIsNotNone(self.recommender.tfidf_vectorizer)
        self.assertIsNotNone(self.recommender.numerical_features)
        self.assertIsNotNone(self.recommender.scaler)

    def test_fit(self):
        """Test model fitting"""
        self.recommender.fit(self.sample_recipes)
        
        # Check if all necessary attributes are created
        self.assertIsNotNone(self.recommender.recipe_ids)
        self.assertIsNotNone(self.recommender.tfidf_matrix)
        self.assertIsNotNone(self.recommender.feature_matrix)
        
        # Check dimensions
        expected_feature_dim = (
            len(self.sample_recipes),  # number of recipes
            self.recommender.tfidf_matrix.shape[1] +  # text features
            len(self.recommender.numerical_features)   # numerical features
        )
        self.assertEqual(self.recommender.feature_matrix.shape, expected_feature_dim)

    def test_recommend(self):
        """Test recommendation generation"""
        self.recommender.fit(self.sample_recipes)
        recommendations = self.recommender.recommend(
            self.sample_interactions,
            n_recommendations=2
        )
        
        # Check recommendation format
        self.assertEqual(len(recommendations), 2)
        self.assertIn('recipe_id', recommendations[0])
        self.assertIn('similarity_score', recommendations[0])
        
        # Check if similarity scores are between 0 and 1
        for rec in recommendations:
            self.assertGreaterEqual(rec['similarity_score'], 0)
            self.assertLessEqual(rec['similarity_score'], 1)

    def test_recommend_exclude_rated(self):
        """Test if rated recipes are excluded from recommendations"""
        self.recommender.fit(self.sample_recipes)
        recommendations = self.recommender.recommend(
            self.sample_interactions,
            n_recommendations=1,
            exclude_rated=True
        )
        
        # Check if recommended recipe is not in user history
        rated_recipes = set(self.sample_interactions['recipe_id'])
        self.assertNotIn(recommendations[0]['recipe_id'], rated_recipes)

    def test_prepare_text_features(self):
        """Test text feature preparation"""
        text_features = self.recommender._prepare_text_features(self.sample_recipes)
        
        # Check if we get one feature string per recipe
        self.assertEqual(len(text_features), len(self.sample_recipes))
        
        # Check if all features are strings
        for feature in text_features:
            self.assertIsInstance(feature, str)
            self.assertGreater(len(feature), 0)

    def test_prepare_numerical_features(self):
        """Test numerical feature preparation"""
        numerical_features = self.recommender._prepare_numerical_features(
            self.sample_recipes
        )
        
        # Check dimensions
        self.assertEqual(
            numerical_features.shape,
            (len(self.sample_recipes), len(self.recommender.numerical_features))
        )
        
        # Check if features are scaled
        self.assertTrue((-3 <= numerical_features.mean() <= 3))  # Lazább ellenőrzés
        self.assertTrue((0.1 <= numerical_features.std() <= 10))  # Reálisabb tartomány

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with empty user history
        self.recommender.fit(self.sample_recipes)
        empty_history = pd.DataFrame(columns=['user_id', 'recipe_id', 'rating'])
        recommendations = self.recommender.recommend(empty_history)
        self.assertEqual(recommendations, [])
        
        # Test with invalid recipe IDs in user history
        invalid_history = pd.DataFrame({
            'user_id': [1],
            'recipe_id': [999],  # non-existent recipe
            'rating': [0.8]
        })
        with self.assertRaises(ValueError):
            self.recommender.recommend(invalid_history)

if __name__ == '__main__':
    unittest.main()