import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np



class modelOne:

    model=None
    X_train=None
    Y_train=None

    def __init__(self,name):
        self.name = name

    def setNormalizers(self, normalizers):
        """Set the normalization layers to be used in the model"""
        modelOne.normalizers = normalizers


    def setRecords(self, x):
        """
        Set training records. Can accept either:
        - Old format: pre-normalized array
        - New format: dict with separate feature arrays (from prepare_season_raw)
        """
        if isinstance(x, dict):
            # New format: concatenate features
            modelOne.X_train = np.column_stack([
                x['weeks'],
                x['conferences'],
                x['win_pcts'],
                x['ap_ranks'],
                x['coaches_ranks'],
                x['cfp_ranks']
            ])
        else:
            # Old format: already normalized array
            modelOne.X_train = np.array(x)

    def setRanks(self,y):
        modelOne.Y_train=np.array(y)

    def setSeeds(self,y):
        modelOne.Y_train=np.array(y)

    def setBracket(self,y):
        modelOne.Y_train=np.array(y)

    def _build_preprocessing_layer(self):
        """
        Build a preprocessing layer that normalizes inputs.
        This layer becomes part of the model and is saved with it.
        """
        if modelOne.normalizers is None:
            # No normalizers provided, assume input is already normalized
            return None
        
        # Create input layer
        inputs = layers.Input(shape=(6,), name='raw_input')
        
        # Split the input into individual features
        week = layers.Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(inputs)
        conference = layers.Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(inputs)
        win_pct = layers.Lambda(lambda x: tf.expand_dims(x[:, 2], axis=-1))(inputs)
        ap_rank = layers.Lambda(lambda x: tf.expand_dims(x[:, 3], axis=-1))(inputs)
        coaches_rank = layers.Lambda(lambda x: tf.expand_dims(x[:, 4], axis=-1))(inputs)
        cfp_rank = layers.Lambda(lambda x: tf.expand_dims(x[:, 5], axis=-1))(inputs)
        
        # Apply normalization to each feature
        week_norm = modelOne.normalizers['week'](week)
        conf_norm = modelOne.normalizers['conference'](conference)
        winpct_norm = modelOne.normalizers['win_pct'](win_pct)
        ap_norm = modelOne.normalizers['ap_rank'](ap_rank)
        coaches_norm = modelOne.normalizers['coaches_rank'](coaches_rank)
        cfp_norm = modelOne.normalizers['cfp_rank'](cfp_rank)
        
        # Concatenate normalized features
        normalized = layers.Concatenate()([
            week_norm, conf_norm, winpct_norm,
            ap_norm, coaches_norm, cfp_norm
        ])
        
        return inputs, normalized

    def buildRankModel(self):
        """Build model to predict final CFP rank (1-25)"""
        print("------building Rank Model------")
        
        if modelOne.normalizers is not None:
            # Build model with preprocessing
            inputs, normalized = self._build_preprocessing_layer()
            
            # Hidden layers
            x = layers.Dense(128, activation='relu')(normalized)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dense(32, activation='relu')(x)
            
            # Output layer - predicts rank (1-25)
            # Using linear activation because we'll clip to valid range
            outputs = layers.Dense(1, activation='linear', name='rank_output')(x)
            
            modelOne.model = models.Model(inputs=inputs, outputs=outputs, name='rank_model')
        else:
            # Old approach: assume input is pre-normalized
            modelOne.model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(6,)),
                layers.Dense(64, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ], name='rank_model_legacy')
        
        # Compile the model
        modelOne.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        # Display model summary
        modelOne.model.summary()

    def buildSeedModel(self):
        """Build model to predict playoff seed (1-12)"""
        print("------building Seed Model------")
        
        if modelOne.normalizers is not None:
            # Build model with preprocessing
            inputs, normalized = self._build_preprocessing_layer()
            
            # Hidden layers
            x = layers.Dense(32, activation='relu')(normalized)
            x = layers.Dense(16, activation='relu')(x)
            x = layers.Dense(8, activation='relu')(x)
            
            # Output layer - predicts seed (1-12)
            outputs = layers.Dense(1, activation='linear', name='seed_output')(x)
            
            modelOne.model = models.Model(inputs=inputs, outputs=outputs, name='seed_model')
        else:
            # Old approach
            modelOne.model = models.Sequential([
                layers.Dense(32, activation='relu', input_shape=(6,)),
                layers.Dense(16, activation='relu'),
                layers.Dense(8, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ], name='seed_model_legacy')
        
        modelOne.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )

        # Display model summary
        modelOne.model.summary()

    def buildBracketModel(self):
        """Build model to predict playoff bracket result (6 categories)"""
        print("------building Bracket Model------")
        
        if modelOne.normalizers is not None:
            # Build model with preprocessing
            inputs, normalized = self._build_preprocessing_layer()
            
            # Hidden layers
            x = layers.Dense(16, activation='relu')(normalized)
            x = layers.Dense(16, activation='relu')(x)
            
            # Output layer for 6 classes
            outputs = layers.Dense(6, activation='softmax', name='bracket_output')(x)
            
            modelOne.model = models.Model(inputs=inputs, outputs=outputs, name='bracket_model')
        else:
            # Old approach
            modelOne.model = models.Sequential([
                layers.Dense(16, activation='relu', input_shape=(6,)),
                layers.Dense(16, activation='relu'),
                layers.Dense(6, activation='softmax')
            ], name='bracket_model_legacy')
        
        modelOne.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Display model summary
        modelOne.model.summary()

    def train(self):
        """Train rank or seed model"""
        print("-----TRAINING-----")
        modelOne.model.fit(
            modelOne.X_train,
            modelOne.Y_train,
            epochs=15,
            batch_size=32,  # Changed from 860 to allow for better gradient updates
            validation_split=0.2  # Added validation split
        )


    def trainBracket(self):
        """Train bracket classification model"""
        print("-----TRAINING-----")
        y_train = tf.keras.utils.to_categorical(modelOne.Y_train, 6)
        
        modelOne.model.fit(
            modelOne.X_train,
            y_train,
            epochs=15,
            batch_size=32,
            validation_split=0.2
        )


    def predictRank(self, input_data):
        """
        Predict final CFP rank.
        
        Args:
            input_data: Either pre-normalized array or raw values (if model has preprocessing)
        """
        prediction = modelOne.model.predict(np.array(input_data))
        
        if modelOne.normalizers is not None:
            # Model outputs raw rank (1-25)
            predicted_rank = np.clip(prediction[0][0], 1, 25)
        else:
            # Legacy: denormalize from sigmoid output (0-1)
            predicted_rank = 26 - (25 * prediction[0][0])
            predicted_rank = np.clip(predicted_rank, 1, 25)
        
        print(f"Predicted Final CFP Rank: {predicted_rank:.0f}")
        return predicted_rank

    def predictSeed(self, input_data):
        """
        Predict playoff seed.
        
        Args:
            input_data: Either pre-normalized array or raw values (if model has preprocessing)
        """
        prediction = modelOne.model.predict(np.array(input_data))
        
        if modelOne.normalizers is not None:
            # Model outputs raw seed (1-12)
            predicted_seed = np.clip(prediction[0][0], 1, 12)
        else:
            # Legacy: denormalize from sigmoid output
            predicted_seed = 13 - (12 * prediction[0][0])
            predicted_seed = np.clip(predicted_seed, 1, 12)
        
        print(f"Predicted Final CFP Seed: {predicted_seed:.0f}")
        return predicted_seed

    def predictResult(self, input_data):
        """
        Predict playoff bracket result.
        
        Args:
            input_data: Either pre-normalized array or raw values (if model has preprocessing)
        """
        playoffResults = [
            'Not in Playoff',
            'National Champion',
            'Runner Up',
            'Semifinals',
            'Round 2',
            'Round 1'
        ]
        
        prediction = modelOne.model.predict(np.array(input_data))
        predicted_result = np.argmax(prediction, axis=1)
        
        result_name = playoffResults[predicted_result[0]]
        confidence = prediction[0][predicted_result[0]] * 100
        
        print(f"Predicted playoff result: {result_name} (confidence: {confidence:.1f}%)")
        return result_name, confidence


    def save_model(self, filepath):
        """Save the model (including preprocessing layers)"""
        if modelOne.model is not None:
            modelOne.model.save(filepath)
            print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a saved model (including preprocessing layers)"""
        modelOne.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")

