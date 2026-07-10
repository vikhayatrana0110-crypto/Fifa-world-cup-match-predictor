import unittest
import pandas as pd
from src.features.team_features import TeamFeatureEngine
from src.features.match_features import MatchFeatureEngine, compute_h2h

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.matches = pd.DataFrame([
            {'match_id': 1, 'date': '2026-06-01', 'home_team': 'France', 'away_team': 'England', 'home_score': 2, 'away_score': 1, 'tournament': 'Friendly', 'neutral': False},
            {'match_id': 2, 'date': '2026-06-05', 'home_team': 'England', 'away_team': 'Germany', 'home_score': 3, 'away_score': 2, 'tournament': 'Friendly', 'neutral': False},
            {'match_id': 3, 'date': '2026-06-10', 'home_team': 'France', 'away_team': 'Germany', 'home_score': 1, 'away_score': 1, 'tournament': 'Friendly', 'neutral': False}
        ])
        self.elo_history = pd.DataFrame([
            {'date': '2026-06-01', 'team': 'France', 'elo_rating': 1515.0},
            {'date': '2026-06-01', 'team': 'England', 'elo_rating': 1485.0},
            {'date': '2026-06-05', 'team': 'England', 'elo_rating': 1500.0},
            {'date': '2026-06-05', 'team': 'Germany', 'elo_rating': 1485.0},
            {'date': '2026-06-10', 'team': 'France', 'elo_rating': 1515.0},
            {'date': '2026-06-10', 'team': 'Germany', 'elo_rating': 1485.0}
        ])

    def test_team_features(self):
        engine = TeamFeatureEngine(window=5)
        team_features = engine.compute_team_features(self.matches, self.elo_history)
        
        # Test columns
        expected_cols = ['team', 'date', 'elo_rating', 'recent_win_rate', 'goals_scored_avg', 'goals_conceded_avg', 'attack_rating', 'defense_rating', 'form_score', 'matches_played']
        for col in expected_cols:
            self.assertIn(col, team_features.columns)

    def test_h2h_calculation(self):
        # 1 match between France and England: France won 2-1
        # H2H advantage for France vs England should be 1.0 (win)
        val = compute_h2h(self.matches, 'France', 'England')
        self.assertEqual(val, 1.0)
        
        # H2H advantage for England vs France should be -1.0 (loss)
        val2 = compute_h2h(self.matches, 'England', 'France')
        self.assertEqual(val2, -1.0)

    def test_match_features(self):
        t_engine = TeamFeatureEngine(window=5)
        team_features = t_engine.compute_team_features(self.matches, self.elo_history)
        
        m_engine = MatchFeatureEngine()
        match_features = m_engine.compute_match_features(self.matches, team_features, self.elo_history)
        
        # Check expected columns
        feature_cols = m_engine.get_feature_columns()
        for col in feature_cols:
            self.assertIn(col, match_features.columns)
            
        # Target column result should exist
        self.assertIn('result', match_features.columns)
        self.assertIn('home_goals', match_features.columns)
        self.assertIn('away_goals', match_features.columns)

if __name__ == '__main__':
    unittest.main()
