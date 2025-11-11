import cfp_stats as cfp
import model1 


print ("loading data from workspace")
#
# load team records and poll rankings  (model 1)
cfp.init_cfp_model()
records=cfp.get_m1_train_data()
print(records)

#
# load the final CFP rankings (model 1)
ranks=cfp.get_m1_results_rank()

#
# load the final CFP playoff seeds  (model 1)
seeds=cfp.get_m1_results_seed()

#
# load the final CFP playoff results   (model 1)
bracket=cfp.get_m1_results_bracket()

#print(bracket)

print ("creating prediction model")
m1=model1.modelOne("Ranks")

print ("loading training data")
m1.setRecords(records)
m1.setRanks(ranks)
m1.setSeeds(seeds)
m1.setBracket(bracket)

print("training data loaded")
#print (m1.X_train)
#print (m1.Y_train)


print("build and train model")
m1.buildRankModel()
m1.train()

print("MODEL READY")
#print(m1.model)

# ARRAY FIELDS
#   Week
#   Conference
#   Win/Loss %
#   AP rank
#   Coaches rank
#   CFP rank

# test with OSU 2024 week 16
TEST_DATA_2024=[[1,0.1,0.8333333333,0.8,0.76,0.8]]


# test with TEXAS 2024 week 15
TEST_DATA_2024_TX=[[0.9375,0.2,0.9166666667,0.96,0.96,0.96]]

# test with OSU 2025 week 8
TEST_DATA_2025=[[0.5,0.1,1.0,1.0,1.0,0]]


# test with TTUN 2025 week 7
TEST_DATA_2025_TTUN=[[0.4375,0.1,0.8,0.44,0.44,0]]

# test by looking up the team and week we want to test
TEST_DATA_LOOKUP=cfp.current_lookup('Ohio State',12)
TEST_DATA_LOOKUP=cfp.normalize_season(TEST_DATA_LOOKUP)


print("making a CFP RANK prediction -------------------------------------- ")

print("2024 OSU week 16")
m1.predictRank(TEST_DATA_2024)

print("2024 TEXAS week 15")
m1.predictRank(TEST_DATA_2024_TX)

print("2025 OSU week 8")
m1.predictRank(TEST_DATA_2025)

print("2025 TTUN week 7")
m1.predictRank(TEST_DATA_2025_TTUN)

print("2025 Lookup (OSU week 12)")
m1.predictRank(TEST_DATA_LOOKUP)


# now run it again for SEEDING
m1.buildSeedModel()
m1.train()


print("making a CFP SEED prediction -------------------------------------- ")


print("2024 OSU week 15")
m1.predictSeed(TEST_DATA_2024)

print("2024 TEXAS week 15")
m1.predictSeed(TEST_DATA_2024_TX)

print("2025 OSU week 7")
m1.predictSeed(TEST_DATA_2025)

print("2025 TTUN week 7")
m1.predictSeed(TEST_DATA_2025_TTUN)


print("2025 Lookup (OSU week 12)")
m1.predictSeed(TEST_DATA_LOOKUP)



# now run it again for PLAYOFF RESULTS
m1.buildBracketModel()
m1.trainBracket()

print("making a CFP RESULTS prediction -------------------------------------- ")

print("2024 OSU week 15")
m1.predictResult(TEST_DATA_2024)

print("2024 TEXAS week 15")
m1.predictResult(TEST_DATA_2024_TX)

print("2025 OSU week 7")
m1.predictResult(TEST_DATA_2025)

print("2025 TTUN week 7")
m1.predictResult(TEST_DATA_2025_TTUN)


print("2025 Lookup (OSU week 12)")
m1.predictResult(TEST_DATA_LOOKUP)




