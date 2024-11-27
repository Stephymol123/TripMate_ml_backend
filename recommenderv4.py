import os
import logging
import traceback
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import pymysql
from pymysql import Error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TravelRecommender:
    def __init__(self):
        self.content_scaler = StandardScaler()
        self.label_encoders = {}
        self.similarity_matrix = None
        self.packages_df = None
        self.user_item_matrix = None
        self.model_dir = 'model'
        
        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Model file paths
        self.similarity_matrix_path = os.path.join(self.model_dir, 'similarity_matrix.joblib')
        self.scaler_path = os.path.join(self.model_dir, 'content_scaler.joblib')
        self.encoders_path = os.path.join(self.model_dir, 'label_encoders.joblib')
        
        # Database configuration
        self.db_config = {
            'host': 'localhost',
            'database': 'trip',
            'user': 'root',
            'password': '',
            'charset': 'utf8mb4',
            'cursorclass': pymysql.cursors.DictCursor
        }

    def connect_to_database(self):
        """Connect to MySQL database"""
        try:
            logger.debug("Attempting to connect to database...")
            connection = pymysql.connect(
                host='localhost',
                user='root',
                password='',
                database='trip',
                cursorclass=pymysql.cursors.DictCursor  # Return results as dictionaries
            )
            logger.info("Successfully connected to database")
            return connection
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def load_packages_data(self) -> pd.DataFrame:
        """Load package data from database"""
        try:
            connection = self.connect_to_database()
            if connection:
                logger.debug("Executing packages query...")
                query = """
                    SELECT id, pack_name, city, state, country, days, night,
                           price_per_person, description, image_url
                    FROM package
                """
                logger.debug(f"Query: {query}")
                
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    logger.debug(f"Query returned {len(result)} rows")
                    logger.debug(f"First row: {result[0] if result else None}")
                    
                    if not result:
                        logger.error("No packages found in database")
                        return pd.DataFrame()
                    
                    # Convert result to DataFrame
                    packages_df = pd.DataFrame(result)
                    logger.info(f"Loaded {len(packages_df)} package records")
                    logger.debug(f"Columns in packages_df: {list(packages_df.columns)}")
                    logger.debug(f"Sample data:\n{packages_df.head()}")
                    
                    # Ensure numeric columns are properly typed
                    numeric_columns = ['price_per_person', 'days', 'night']
                    for col in numeric_columns:
                        if col in packages_df.columns:
                            packages_df[col] = pd.to_numeric(packages_df[col], errors='coerce')
                    
                    return packages_df
            else:
                logger.error("Failed to connect to database")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading package data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    def load_user_bookings(self) -> pd.DataFrame:
        """Load user booking data from database"""
        try:
            connection = self.connect_to_database()
            if connection:
                logger.debug("Executing bookings query...")
                query = """
                    SELECT b.user_id, b.package_id, b.booking_date, b.user_count, b.amount,
                           p.pack_name, p.city, p.state, p.country, p.days, p.night,
                           p.price_per_person, p.description, p.image_url
                    FROM pack_bookings b
                    JOIN package p ON b.package_id = p.id
                """
                logger.debug(f"Query: {query}")
                
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    result = cursor.fetchall()
                    logger.debug(f"Query returned {len(result)} rows")
                    logger.debug(f"First row: {result[0] if result else None}")
                    
                    if not result:
                        logger.error("No bookings found in database")
                        return pd.DataFrame()
                    
                    # Convert result to DataFrame
                    bookings_df = pd.DataFrame(result)
                    logger.info(f"Loaded {len(bookings_df)} booking records")
                    logger.debug(f"Columns in bookings_df: {list(bookings_df.columns)}")
                    logger.debug(f"Sample data:\n{bookings_df.head()}")
                    
                    # Ensure numeric columns are properly typed
                    numeric_columns = ['user_id', 'package_id', 'user_count', 'amount']
                    for col in numeric_columns:
                        if col in bookings_df.columns:
                            bookings_df[col] = pd.to_numeric(bookings_df[col], errors='coerce')
                    
                    return bookings_df
            else:
                logger.error("Failed to connect to database")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading booking data: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    def get_user_recommendations(self, user_id: int, num_recommendations: int = 5) -> List[Dict]:
        """Get package recommendations for a user based on their booking history"""
        try:
            logger.info(f"Generating {num_recommendations} recommendations for user_id: {user_id}")
            
            # Load all packages first
            all_packages = self.load_packages_data()
            if all_packages.empty:
                logger.error("No package data available")
                return []

            # Load bookings data
            bookings_df = self.load_user_bookings()
            if bookings_df.empty:
                logger.error("No booking data available")
                return self.get_popular_recommendations(num_recommendations)

            # Get user's booking history
            user_bookings = bookings_df[bookings_df['user_id'] == user_id]
            if user_bookings.empty:
                logger.info(f"No booking history for user {user_id}")
                # Fall back to popular packages
                return self.get_popular_recommendations(num_recommendations)

            logger.debug(f"Found {len(user_bookings)} bookings for user {user_id}")
            logger.debug(f"User bookings: {user_bookings.to_dict('records')}")

            # Get booked packages details
            booked_packages = all_packages[all_packages['id'].isin(user_bookings['package_id'])]
            if booked_packages.empty:
                logger.warning("No matching package details found for user's bookings")
                return self.get_popular_recommendations(num_recommendations)

            # Get preferences from booked packages
            user_preferences = {
                'cities': booked_packages['city'].tolist(),
                'states': booked_packages['state'].tolist(),
                'price_range': {
                    'min': float(booked_packages['price_per_person'].min() * 0.6),  # More flexible price range
                    'max': float(booked_packages['price_per_person'].max() * 1.4)
                },
                'duration': {
                    'min': max(1, int(booked_packages['days'].min() - 2)),  # More flexible duration
                    'max': int(booked_packages['days'].max() + 2)
                }
            }
            
            logger.debug(f"User preferences: {user_preferences}")

            # Calculate base scores for all packages
            all_packages['location_score'] = all_packages.apply(
                lambda x: (
                    2 if x['city'] in user_preferences['cities'] else
                    1 if x['state'] in user_preferences['states'] else
                    0.5  # Base score for all packages
                ),
                axis=1
            )

            # Calculate price similarity score (inverse of difference)
            user_avg_price = booked_packages['price_per_person'].astype(float).mean()
            all_packages['price_score'] = 1 / (1 + abs(
                all_packages['price_per_person'].astype(float) - user_avg_price
            ) / user_avg_price)

            # Calculate duration similarity score
            user_avg_duration = booked_packages['days'].astype(int).mean()
            all_packages['duration_score'] = 1 / (1 + abs(
                all_packages['days'].astype(int) - user_avg_duration
            ))

            # Combine scores with weights
            all_packages['total_score'] = (
                0.4 * all_packages['location_score'] +
                0.3 * all_packages['price_score'] +
                0.3 * all_packages['duration_score']
            )

            # Sort by total score
            recommended_packages = all_packages.sort_values(
                'total_score', ascending=False
            )

            # Exclude already booked packages
            booked_package_ids = user_bookings['package_id'].tolist()
            recommended_packages = recommended_packages[~recommended_packages['id'].isin(booked_package_ids)]

            # Get top N recommendations
            recommended_packages = recommended_packages.head(num_recommendations)

            logger.info(f"Found {len(recommended_packages)} recommendations")
            logger.debug(f"Recommended packages: {recommended_packages.to_dict('records')}")

            # Format recommendations
            recommendations = []
            for _, package in recommended_packages.iterrows():
                recommendations.append({
                    'id': int(package['id']),
                    'pack_name': str(package['pack_name']),
                    'city': str(package['city']),
                    'state': str(package['state']),
                    'country': str(package['country']),
                    'days': int(package['days']),
                    'night': int(package['night']),
                    'price_per_person': float(package['price_per_person']),
                    'image_url': str(package['image_url']) if pd.notna(package['image_url']) else None,
                    'match_score': float(package['total_score'])  # Use the combined score
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error generating user recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def get_popular_recommendations(self, num_recommendations: int = 5) -> List[Dict]:
        """Get popular package recommendations when user has no history"""
        try:
            logger.info("Getting popular package recommendations")
            
            # Load bookings data
            bookings_df = self.load_user_bookings()
            if bookings_df.empty:
                logger.error("No booking data available")
                return []

            # Get package booking counts
            package_popularity = bookings_df.groupby('package_id').agg({
                'user_id': 'count',  # booking count
                'user_count': 'sum',  # total travelers
                'amount': 'mean'  # average booking amount
            }).reset_index()

            # Calculate popularity score
            package_popularity['popularity_score'] = (
                0.5 * package_popularity['user_id'] +  # booking count weight
                0.3 * package_popularity['user_count'] +  # travelers weight
                0.2 * (package_popularity['amount'] / package_popularity['amount'].max())  # normalized amount weight
            )

            # Get top packages
            popular_package_ids = package_popularity.nlargest(
                num_recommendations, 'popularity_score'
            )['package_id'].tolist()

            # Get package details
            all_packages = self.load_packages_data()
            popular_packages = all_packages[all_packages['id'].isin(popular_package_ids)]

            # Format recommendations
            recommendations = []
            for _, package in popular_packages.iterrows():
                recommendations.append({
                    'id': int(package['id']),
                    'pack_name': str(package['pack_name']),
                    'city': str(package['city']),
                    'state': str(package['state']),
                    'country': str(package['country']),
                    'days': int(package['days']),
                    'night': int(package['night']),
                    'price_per_person': float(package['price_per_person']),
                    'image_url': str(package['image_url']) if pd.notna(package['image_url']) else None,
                    'match_score': 0.5  # Default score for popular recommendations
                })

            return recommendations

        except Exception as e:
            logger.error(f"Error getting popular recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def save_model(self) -> bool:
        """Save model artifacts to disk"""
        try:
            logger.info("Saving model artifacts...")
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(os.path.dirname(__file__), 'model')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save similarity matrix
            similarity_path = os.path.join(model_dir, 'similarity_matrix.pkl')
            logger.debug(f"Saving similarity matrix to {similarity_path}")
            joblib.dump(self.similarity_matrix, similarity_path)
            
            # Save packages data
            packages_path = os.path.join(model_dir, 'packages_data.pkl')
            logger.debug(f"Saving packages data to {packages_path}")
            self.packages_df.to_pickle(packages_path)
            
            logger.info("Successfully saved model artifacts")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def train_model(self) -> bool:
        """Train the recommendation model"""
        try:
            logger.info("Starting model training...")
            
            # Load package data
            packages_df = self.load_packages_data()
            if packages_df.empty:
                logger.error("No package data available for training")
                return False

            logger.debug(f"Loaded {len(packages_df)} packages for training")
            logger.debug(f"Package columns: {packages_df.columns.tolist()}")
            logger.debug(f"Sample package data:\n{packages_df.head()}")

            # Process text features
            text_features = []
            text_columns = ['pack_name', 'description', 'city', 'state', 'country']
            
            # Check for missing text columns
            missing_cols = [col for col in text_columns if col not in packages_df.columns]
            if missing_cols:
                logger.error(f"Missing text columns: {missing_cols}")
                return False

            # Combine and process text features
            logger.debug("Processing text features...")
            packages_df[text_columns] = packages_df[text_columns].fillna('')
            combined_text = packages_df[text_columns].apply(
                lambda x: ' '.join(str(val).strip() for val in x),
                axis=1
            )

            # TF-IDF transformation
            try:
                tfidf = TfidfVectorizer(
                    stop_words='english',
                    min_df=1,
                    max_features=100,
                    strip_accents='unicode',
                    lowercase=True
                )
                text_features = tfidf.fit_transform(combined_text)
                logger.debug(f"Text features shape: {text_features.shape}")
            except Exception as e:
                logger.error(f"Error in TF-IDF processing: {str(e)}")
                logger.error(traceback.format_exc())
                return False

            # Process numerical features
            numerical_columns = ['days', 'night', 'price_per_person']
            
            # Check for missing numerical columns
            missing_cols = [col for col in numerical_columns if col not in packages_df.columns]
            if missing_cols:
                logger.error(f"Missing numerical columns: {missing_cols}")
                return False

            logger.debug("Processing numerical features...")
            try:
                # Convert to numeric and handle missing values
                numerical_data = packages_df[numerical_columns].copy()
                for col in numerical_columns:
                    numerical_data[col] = pd.to_numeric(numerical_data[col], errors='coerce')
                    if numerical_data[col].isna().any():
                        logger.warning(f"Found {numerical_data[col].isna().sum()} NA values in {col}")
                    numerical_data[col] = numerical_data[col].fillna(numerical_data[col].mean())

                # Scale numerical features
                scaler = StandardScaler()
                numerical_features = scaler.fit_transform(numerical_data)
                logger.debug(f"Numerical features shape: {numerical_features.shape}")
            except Exception as e:
                logger.error(f"Error in numerical feature processing: {str(e)}")
                logger.error(traceback.format_exc())
                return False

            # Combine all features
            logger.debug("Combining features...")
            try:
                # Convert sparse matrix to dense
                text_features_dense = text_features.toarray()
                
                # Combine text and numerical features
                combined_features = np.hstack([text_features_dense, numerical_features])
                logger.debug(f"Combined features shape: {combined_features.shape}")

                # Calculate similarity matrix
                logger.debug("Calculating similarity matrix...")
                similarity_matrix = cosine_similarity(combined_features)
                logger.debug(f"Similarity matrix shape: {similarity_matrix.shape}")

                # Save model artifacts
                logger.info("Saving model artifacts...")
                model_dir = os.path.join(os.path.dirname(__file__), 'model')
                os.makedirs(model_dir, exist_ok=True)

                # Save similarity matrix
                similarity_matrix_path = os.path.join(model_dir, 'similarity_matrix.pkl')
                joblib.dump(similarity_matrix, similarity_matrix_path)
                logger.debug(f"Saved similarity matrix to {similarity_matrix_path}")

                # Save packages data
                packages_data_path = os.path.join(model_dir, 'packages_data.pkl')
                packages_df.to_pickle(packages_data_path)
                logger.debug(f"Saved packages data to {packages_data_path}")

                logger.info("Model training completed successfully")
                return True

            except Exception as e:
                logger.error(f"Error in feature combination or model saving: {str(e)}")
                logger.error(traceback.format_exc())
                return False

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def load_model(self) -> bool:
        """Load trained model artifacts"""
        try:
            if os.path.exists(self.similarity_matrix_path):
                self.similarity_matrix = joblib.load(self.similarity_matrix_path)
                self.content_scaler = joblib.load(self.scaler_path)
                self.label_encoders = joblib.load(self.encoders_path)
                self.packages_df = self.load_packages_data()
                logger.info("Model artifacts loaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            return False

    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[int]:
        """Get collaborative filtering recommendations for a user"""
        if self.user_item_matrix.empty:
            logger.warning("No user-item matrix available for collaborative filtering")
            return []
        
        try:
            if user_id in self.user_item_matrix.index:
                # Get user's booking history
                user_bookings = self.user_item_matrix.loc[user_id]
                # Calculate similarity with other users based on booking patterns
                similar_users = cosine_similarity([user_bookings], self.user_item_matrix)[0]
                similar_users_indices = np.argsort(similar_users)[::-1][1:]  # Exclude the user themselves
                
                recommendations = []
                for sim_user_idx in similar_users_indices:
                    if len(recommendations) >= n_recommendations:
                        break
                    
                    sim_user_id = self.user_item_matrix.index[sim_user_idx]
                    sim_user_bookings = self.user_item_matrix.loc[sim_user_id]
                    # Get packages booked by similar user that current user hasn't booked
                    potential_recs = sim_user_bookings[sim_user_bookings > 0].index.tolist()
                    
                    for package_id in potential_recs:
                        if user_bookings[package_id] == 0 and package_id not in recommendations:
                            recommendations.append(package_id)
                            if len(recommendations) >= n_recommendations:
                                break
                
                return recommendations[:n_recommendations]
            else:
                logger.warning(f"User {user_id} not found in user-item matrix")
                return []
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {str(e)}")
            return []

    def get_recommendations(self, package_id: int, num_recommendations: int = 5) -> List[Dict]:
        """Get package recommendations based on package ID"""
        try:
            logger.info(f"Generating {num_recommendations} recommendations for package_id: {package_id}")
            
            # Load model if not loaded
            if self.similarity_matrix is None or self.packages_df is None:
                logger.info("Loading model artifacts...")
                if not self.load_model():
                    logger.error("Failed to load model")
                    return []

            # Find the package index
            package_idx = self.packages_df.index[self.packages_df['id'] == package_id].tolist()
            if not package_idx:
                logger.error(f"Package ID {package_id} not found in dataset")
                return []
            package_idx = package_idx[0]
            
            logger.debug(f"Found package at index {package_idx}")
            
            # Get similarity scores for this package
            package_similarities = self.similarity_matrix[package_idx]
            
            # Get indices of most similar packages (excluding self)
            similar_indices = np.argsort(package_similarities)[::-1]
            similar_indices = similar_indices[similar_indices != package_idx][:num_recommendations]
            
            # Get similarity scores for recommended packages
            similar_scores = package_similarities[similar_indices]
            
            logger.debug(f"Found {len(similar_indices)} similar packages")
            logger.debug(f"Similarity scores: {similar_scores}")
            
            # Get the similar packages
            similar_packages = self.packages_df.iloc[similar_indices]
            
            # Format recommendations
            recommendations = self.format_recommendations(similar_packages, similar_scores)
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def format_recommendations(self, similar_packages: pd.DataFrame, similar_scores: np.ndarray) -> List[Dict]:
        """Format recommendations into a list of dictionaries"""
        try:
            recommendations = []
            for idx, package in similar_packages.iterrows():
                recommendation = {
                    'id': int(package['id']),
                    'pack_name': str(package['pack_name']),
                    'city': str(package['city']),
                    'state': str(package['state']),
                    'country': str(package['country']),
                    'days': int(package['days']),
                    'night': int(package['night']),
                    'price_per_person': float(package['price_per_person']),
                    'similarity_score': float(similar_scores[len(recommendations)]),
                    'image_url': str(package['image_url']) if pd.notna(package['image_url']) else None
                }
                recommendations.append(recommendation)
            return recommendations
        except Exception as e:
            logger.error(f"Error formatting recommendations: {str(e)}")
            logger.error(traceback.format_exc())
            return []

    def load_model(self) -> bool:
        """Load trained model artifacts"""
        try:
            logger.info("Loading model artifacts...")
            
            # Check if model files exist
            model_dir = os.path.join(os.path.dirname(__file__), self.model_dir)
            similarity_path = os.path.join(model_dir, 'similarity_matrix.pkl')
            packages_path = os.path.join(model_dir, 'packages_data.pkl')
            
            if not os.path.exists(similarity_path) or not os.path.exists(packages_path):
                logger.error("Model files not found")
                return False
            
            # Load model artifacts
            logger.debug(f"Loading similarity matrix from {similarity_path}")
            self.similarity_matrix = joblib.load(similarity_path)
            
            logger.debug(f"Loading packages data from {packages_path}")
            self.packages_df = pd.read_pickle(packages_path)
            
            logger.info("Successfully loaded model artifacts")
            logger.debug(f"Similarity matrix shape: {self.similarity_matrix.shape}")
            logger.debug(f"Packages data shape: {self.packages_df.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            return False

# Create recommender instance
recommender = TravelRecommender()

# Initialize Flask app
app = Flask(__name__)

@app.route('/api/train', methods=['GET'])
def train():
    """Train the recommendation model"""
    try:
        logger.info("Training model...")
        if recommender.train_model():
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Error during model training'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during model training: {str(e)}'
        }), 500

@app.route('/api/recommendations')
def get_recommendations():
    """Get travel package recommendations"""
    try:
        # Get parameters
        user_id = request.args.get('user_id', type=int)
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': 'user_id is required'
            }), 400

        num_recommendations = request.args.get('num_recommendations', default=5, type=int)

        # Get recommendations
        recommendations = recommender.get_user_recommendations(
            user_id=user_id,
            num_recommendations=num_recommendations
        )

        if not recommendations:
            return jsonify({
                'status': 'error',
                'message': 'No recommendations available'
            }), 404

        return jsonify({
            'status': 'success',
            'recommendations': recommendations
        })

    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    try:
        # Test database connection on startup
        logger.info("Testing database connection on startup...")
        connection = recommender.connect_to_database()
        if not connection:
            logger.error("Failed to connect to database on startup. Please check database configuration and ensure MySQL server is running.")
            raise RuntimeError("Database connection failed")
        else:
            connection.close()
            logger.info("Database connection test successful")
        
        # Load initial data
        logger.info("Loading initial package data...")
        packages_df = recommender.load_packages_data()
        if packages_df.empty:
            logger.warning("No package data loaded. The recommendation system will have limited functionality.")
        else:
            logger.info(f"Successfully loaded {len(packages_df)} packages")
            logger.debug(f"Package columns: {packages_df.columns.tolist()}")
            logger.debug(f"First package: {packages_df.iloc[0].to_dict()}")
        
        # Try loading bookings data
        logger.info("Loading initial bookings data...")
        bookings_df = recommender.load_user_bookings()
        if bookings_df.empty:
            logger.warning("No booking data found. Will proceed with content-based filtering only.")
        else:
            logger.info(f"Successfully loaded {len(bookings_df)} bookings")
            logger.debug(f"Booking columns: {bookings_df.columns.tolist()}")
            logger.debug(f"First booking: {bookings_df.iloc[0].to_dict()}")
        
        # Start Flask server
        logger.info("Starting Flask server...")
        app.run(port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error during server startup: {str(e)}")
        logger.error(traceback.format_exc())
        raise