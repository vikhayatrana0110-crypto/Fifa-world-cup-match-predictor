import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.models.predictor import MatchPredictor
from src.models.explainer import MatchExplainer

class TestPredictorAndExplainer(unittest.TestCase):
    def setUp(self):
        # Create mock models
        self.mock_classifier = MagicMock()
        # predict_proba returns [Away Prob, Draw Prob, Home Prob]
        # Let's say Brazil (home) vs Argentina (away): 50% Win, 30% Draw, 20% Loss
        self.mock_classifier.predict_proba = MagicMock(return_value=np.array([[0.20, 0.30, 0.50]]))
        
        self.mock_home_goals = MagicMock()
        self.mock_home_goals.predict = MagicMock(return_value=np.array([2.1]))
        
        self.mock_away_goals = MagicMock()
        self.mock_away_goals.predict = MagicMock(return_value=np.array([1.2]))
        
        # Instantiate MatchPredictor using mock models
        self.predictor = MatchPredictor.__new__(MatchPredictor)
        self.predictor.classifier = self.mock_classifier
        self.predictor.home_goals_model = self.mock_home_goals
        self.predictor.away_goals_model = self.mock_away_goals
        self.predictor.db_manager = None
        self.predictor.feature_engine = MagicMock()
        self.predictor.feature_engine.get_feature_columns = MagicMock(return_value=['f1', 'f2'])

    def test_prediction_output(self):
        feature_vector = {'f1': 1.0, 'f2': 0.5}
        pred = self.predictor.predict(feature_vector)
        
        self.assertAlmostEqual(pred['home_win_prob'], 0.50)
        self.assertAlmostEqual(pred['draw_prob'], 0.30)
        self.assertAlmostEqual(pred['away_win_prob'], 0.20)
        
        # Probs must sum to 1.0
        self.assertAlmostEqual(pred['home_win_prob'] + pred['draw_prob'] + pred['away_win_prob'], 1.0)
        
        # Expected Goals
        self.assertAlmostEqual(pred['expected_home_goals'], 2.1)
        self.assertAlmostEqual(pred['expected_away_goals'], 1.2)
        
        # Outcome label
        self.assertEqual(pred['outcome'], 'Home Win')

    def test_explainer_fallback(self):
        # Instantiate explainer with mock model
        explainer = MatchExplainer(self.mock_classifier)
        
        # Mock dataframe
        X = pd.DataFrame([{'elo_difference': 100.0, 'form_difference': 0.5, 'other_feature': 1.0}])
        feature_names = ['elo_difference', 'form_difference', 'other_feature']
        
        # predicted class 2 = Home Win
        explanation = explainer.explain_prediction(X, feature_names, predicted_class=2)
        
        # Check output structure
        self.assertIn('positive_factors', explanation)
        self.assertIn('negative_factors', explanation)
        self.assertIn('explanation_text', explanation)
        self.assertTrue(len(explanation['positive_factors']) > 0)

if __name__ == '__main__':
    unittest.main()
