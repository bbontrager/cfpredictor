import cfp_stats
import model1 

import numpy as np

def main():  
    # Step 1: Load reference data
    print("Step 1: Loading reference data...")
    cfp_stats.get_reference_data()
    
    # Step 2: Load training data (2024 season)
    print("\nStep 2: Loading 2024 training data...")
    season_24 = cfp_stats.read_sheet_data(cfp_stats.TRAIN_RECORDS)

    # Step 3: Prepare raw data (NO manual normalization)
    print("\nStep 3: Preparing raw data...")
    raw_training_data = cfp_stats.prepare_season_raw(season_24)
    print(f"  - {len(raw_training_data['weeks'])} records loaded")
    print(f"  - Sample AP ranks: {raw_training_data['ap_ranks'][:5]}")
    
    # Step 4: Create normalization layers based on training data
    print("\nStep 4: Creating normalization layers...")
    normalizers = cfp_stats.create_normalization_layers(raw_training_data)
    print("  - Normalization layers created and adapted to training data")
    
    # Step 5: Create and configure model
    print("\nStep 5: Building rank prediction model...")
    rank_model = model1.modelOne("Rank Predictor")
    
    # Pass normalizers to model so it can include preprocessing
    rank_model.setNormalizers(normalizers)
    
    # Set training data (raw values, not pre-normalized)
    rank_model.setRecords(raw_training_data)
    
    # TODO: Load actual rank results from your data
    # This is a placeholder - you'll need to implement getting actual ranks
    # from your spreadsheet in cfp_stats.py
    dummy_ranks = np.random.randint(1, 26, size=len(raw_training_data['weeks']))
    rank_model.setRanks(dummy_ranks)
    
    # Build the model (includes preprocessing layers)
    rank_model.buildRankModel()
    
    # Step 6: Train the model
    print("\nStep 6: Training model...")
    rank_model.train()
    
    # Step 7: Make predictions
    print("\nStep 7: Making predictions...")
    
    # Example prediction input (raw values, NOT normalized)
    test_input = [[
        16,    # Week 16
        1,     # Conference ID
        0.92,  # Win percentage (11-1)
        3,     # AP Rank
        4,     # Coaches Rank
        2      # CFP Rank
    ]]
    
    print("\nTest input (raw values):")
    print(f"  Week: 16, Conference: 1, Win%: 0.92")
    print(f"  AP Rank: 3, Coaches: 4, CFP: 2")

    test_input = [[
        12,    # Week 16
        1,     # Conference ID
        1.0,  # Win percentage (9-0)
        1,     # AP Rank
        1,     # Coaches Rank
        1      # CFP Rank
    ]]
    
    
    print("\nTest input (raw values):")
    print(f"  Week: 12, Conference: 1 (Big Ten), Win%: 1.0")
    print(f"  AP Rank: 1, Coaches: 1, CFP: 1")

    
    predicted_rank = rank_model.predictRank(test_input)
    
    # Step 8: Save the model (includes preprocessing layers!)
    print("\nStep 8: Saving model...")
    rank_model.save_model('rank_model_with_preprocessing.keras')
    print("  - Model saved with preprocessing layers included")
    print("  - No need to save normalization separately!")
    
    print("\n=== Key Benefits of This Approach ===")
    print("✓ No manual normalization/denormalization code")
    print("✓ Normalization is part of the model")
    print("✓ Model can accept raw input values")
    print("✓ Easier to maintain and less error-prone")
    print("✓ Model is self-contained when saved")


def example_using_saved_model():
    """
    Example of loading and using a saved model.
    The model includes preprocessing, so you can pass raw values!
    """
    print("\n=== Using Saved Model ===\n")
    
    # Load model
    loaded_model = model1.modelOne("Loaded Model")
    loaded_model.load_model('rank_model_with_preprocessing.keras')
    
    # Make prediction with raw values
    test_input = [[
        12,    # Week 12
        2,     # Conference ID
        0.83,  # Win percentage (10-2)
        8,     # AP Rank
        9,     # Coaches Rank
        7      # CFP Rank
    ]]
    
    print("Making prediction with raw input values...")
    loaded_model.predictRank(test_input)


def backward_compatible_example():
    """
    Example showing backward compatibility with old approach.
    """
    print("\n=== Backward Compatible Usage ===\n")
    print("The old approach still works if you don't pass normalizers:")
    
    # Load and normalize data the old way
    season_24 = cfp_stats.read_sheet_data(cfp_stats.RECORDS_2024)
    normalized_data = cfp_stats.normalize_season(season_24)
    
    # Create model without normalizers
    old_model = model1.modelOne("Legacy Model")
    old_model.setRecords(normalized_data)
    
    # Build and use as before
    old_model.buildRankModel()
    print("  - Old approach still works for backward compatibility")


if __name__ == '__main__':
    main()
    
    # Uncomment these to try other examples:
    # example_using_saved_model()
    # backward_compatible_example()



