import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np



class modelOne:

    model=None
    X_train=None
    Y_train=None

    def __init__(self,name):
        self.name = name

    def setRecords(self,x):
        modelOne.X_train=np.array(x)

    def setRanks(self,y):
        modelOne.Y_train=np.array(y)

    def setSeeds(self,y):
        modelOne.Y_train=np.array(y)

    def setBracket(self,y):
        modelOne.Y_train=np.array(y)

    def buildRankModel(self):
        print("------building Rank Model------")
        # Build the neural network model
        modelOne.model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(6,)), # Hidden layer with 64 neurons
            layers.Dense(64, activation='relu'),                   # Hidden layer with 32 neurons
            layers.Dense(32, activation='relu'),                   # Hidden layer with 32 neurons
            layers.Dense(1, activation='sigmoid')                # Output layer for regression (1 numerical output)
        ])

        # Compile the model
        modelOne.model.compile(optimizer='adam',
                      loss='mean_squared_error',  # Suitable for regression
                      metrics=['mean_squared_error'])

        # Display model summary
        modelOne.model.summary()

    def buildSeedModel(self):
        print("------building Seed Model------")
        # Build the neural network model
        modelOne.model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(6,)),  # Hidden layer with 64 neurons
            layers.Dense(16, activation='relu'),                   # Hidden layer with 32 neurons
            layers.Dense(8, activation='relu'),                   # Hidden layer with 32 neurons
            layers.Dense(1, activation='sigmoid')                  # Output layer for regression (1 numerical output)
        ])

        # Compile the model
        modelOne.model.compile(optimizer='adam',
                      loss='mean_squared_error',  # Suitable for regression
                      metrics=['mean_squared_error'])

        # Display model summary
        modelOne.model.summary()

    def buildBracketModel(self):
        print("------building Bracket Model------")
        # Build the neural network model
        modelOne.model = models.Sequential([
            layers.Dense(16, activation='relu', input_shape=(6,)),  # Hidden layer with 64 neurons
            layers.Dense(16, activation='relu'),                   # Hidden layer with 32 neurons
            layers.Dense(6, activation='softmax')                   # Output layer for 6 classes (1-5 and not in playoff)

        ])

        # Compile the model
        modelOne.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',   #Suitable for categorizing
                      metrics=['accuracy'])

        # Display model summary
        modelOne.model.summary()

    def train(self):
        print("-----TRAINING-----")
        #
        # Train the model
        modelOne.model.fit(modelOne.X_train, modelOne.Y_train, epochs=15, batch_size=860)

        # Evaluate the model on test data
        #test_loss, test_mae = modelOne.model.evaluate(X_test, y_test)
        #print(f"\nTest Mean Absolute Error: {test_mae:.4f}")


    def trainBracket(self):
        print("-----TRAINING-----")
        y_train=tf.keras.utils.to_categorical(modelOne.Y_train, 6)
        #
        # Train the model
        modelOne.model.fit(modelOne.X_train, y_train, epochs=15, batch_size=860)

        # Evaluate the model on test data
        #test_loss, test_mae = modelOne.model.evaluate(X_test, y_test)
        #print(f"\nTest Mean Absolute Error: {test_mae:.4f}")


    def predictRank(self,input):
        # Make a prediction 
        prediction = 26-(25*modelOne.model.predict(np.array(input)))
        print(f"Predicted Final CFP Rank for input: {prediction[0][0]:.0f}")

    def predictSeed(self,input):
        # Make a prediction 
        prediction = 13-(12*modelOne.model.predict(np.array(input)))
        print(f"Predicted Final CFP Seed for input: {prediction[0][0]:.0f}")

    def predictResult(self,input):
        # Make a prediction 
        playoffResults=['Not in Playoff','National Champion','Runner Up','Semifinals','Round 2','Round 1']
        prediction = modelOne.model.predict(np.array(input))
        predicted_result = np.argmax(prediction, axis=1)
        print(f"Predicted playoff result: {playoffResults[predicted_result[0]]}")


