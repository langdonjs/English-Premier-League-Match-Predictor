import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
# Replace 'your_file.csv' with the actual path to your CSV file
file_path = './data/2021-2022.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
# Initiates X and Y and score
X = np.zeros((380, 2))
Y = []
home = df["HomeTeam"]
away = df["AwayTeam"]
scores = df["FTR"]
# Makes scores numerical from H A D to 0 1 2
newScore = []
for letter in scores:
    if letter == 'H':
        newScore.append(0)
    elif letter == 'D':
        newScore.append(1)
    else:
        newScore.append(2)
# dictionary to hold team names and points
teams = {"Man City": 0, "Liverpool": 0, "Chelsea": 0, "Tottenham": 0, "Arsenal": 0,
       "Man United": 0, "West Ham": 0, "Leicester": 0, "Brighton": 0, "Wolves": 0,
       "Newcastle": 0, "Crystal Palace": 0, "Brentford": 0, "Aston Villa": 0, "Southampton": 0,
       "Everton": 0, "Leeds": 0, "Burnley": 0, "Watford": 0, "Norwich": 0}

#algorithm: Inputs values for X(teams) and Y(score) by position
# it adds the current positions of the team every 10 games

count = 0
for index,score in enumerate(newScore):
    home_team = home[index]
    away_team = away[index]
    if score == 0:
        teams[home_team] += 3
    elif score == 2:
        teams[away_team] += 3
    else:
        teams[home_team] += 1
        teams[away_team] += 1
    X[index] = list(teams).index(home_team),list(teams).index(away_team)
    Y.append(score)
    count += 1
    if count == 10:
        # sorts dictionary by numerical value from greatest to least
        teams = dict(sorted(teams.items(), key=lambda x: x[1], reverse=True))
        print(teams)
        count = 0
print(teams)

##takes out first 20% fixtures
percent_to_keep = 0.80
num_rows_to_keep = int(X.shape[0] * percent_to_keep)
X = X[num_rows_to_keep:]

num_elements_to_keep = int(len(Y) * percent_to_keep)
Y = Y[num_elements_to_keep:]

##split dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
MLP = MLPClassifier(max_iter=400).fit(X_train, y_train)
predictionsMLP = MLP.predict(X_test)
correct = 0
for index,prediction in enumerate(predictionsMLP):
    if prediction == y_test[index]:
        correct += 1
accuracy = correct/len(predictionsMLP)
print(accuracy)

# traverse i in range X_test if predictionsMLP,RFC,DTC get right increase weight by 1
RFC = RandomForestClassifier(max_depth=2, random_state=0)
RFC.fit(X_train, y_train)
predictionsRFC = RFC.predict(X_test)
secondCorrect = 0
for index,prediction in enumerate(predictionsRFC):
    if prediction == y_test[index]:
        secondCorrect += 1
secondAccuracy = secondCorrect/len(predictionsRFC)
print(secondAccuracy)

# predictionsRFC = RFC.predict(X_train)
# secondCorrect = 0
# for index,prediction in enumerate(predictionsRFC):
#     if prediction == y_train[index]:
#         secondCorrect += 1
# secondAccuracy = secondCorrect/len(predictionsRFC)
# print(secondAccuracy) (do this for all models)
# measure the games another test to check for all home game wins how many home games were predicted right(for both draw and away also)
#

# code this 3rd model
## change variables to first second third
DTC = DecisionTreeClassifier(random_state=0)
DTC.fit(X_train, y_train)
predictionsDTC = DTC.predict(X_test)
thirdCorrect = 0
for index,prediction in enumerate(predictionsDTC):
    if prediction == y_test[index]:
        thirdCorrect += 1
thirdAccuracy = thirdCorrect/len(predictionsDTC)
print(thirdAccuracy)

w1,w2,w3 = 0,0,0
for i in range(len(y_test)):
    if(predictionsMLP[i] == y_test[i]):
        w1 += 1
    if (predictionsRFC[i] == y_test[i]):
        w2 += 1
    if (predictionsDTC[i] == y_test[i]):
        w3 += 1
print(w1)
print(w2)
print(w3)

def calculate_accuracy(predictions, y_test):
    correct_wins, correct_draws, correct_away = 0, 0, 0
    total_predicted_wins, total_actual_wins = 0, 0
    total_predicted_draws, total_actual_draws = 0, 0
    total_predicted_away, total_actual_away = 0, 0

    for index, prediction in enumerate(predictions):
        actual_outcome = y_test[index]

        if prediction == 0:
            total_predicted_wins += 1
            if actual_outcome == 0:
                correct_wins += 1

        elif prediction == 1:
            total_predicted_draws += 1
            if actual_outcome == 1:
                correct_draws += 1

        elif prediction == 2:
            total_predicted_away += 1
            if actual_outcome == 2:
                correct_away += 1

    accuracy_wins = correct_wins / total_predicted_wins if total_predicted_wins > 0 else 0
    accuracy_draws = correct_draws / total_predicted_draws if total_predicted_draws > 0 else 0
    accuracy_away = correct_away / total_predicted_away if total_predicted_away > 0 else 0

    return accuracy_wins, accuracy_draws, accuracy_away

# Calculate and print accuracy for MLP
accuracy_wins_mlp, accuracy_draws_mlp, accuracy_away_mlp = calculate_accuracy(predictionsMLP, y_test)
print("MLP Model:")
print(f"Accuracy for Predicting Wins: {accuracy_wins_mlp:.2%}")
print(f"Accuracy for Predicting Draws: {accuracy_draws_mlp:.2%}")
print(f"Accuracy for Predicting Away Outcomes: {accuracy_away_mlp:.2%}")
print()

# Calculate and print accuracy for RFC
accuracy_wins_rfc, accuracy_draws_rfc, accuracy_away_rfc = calculate_accuracy(predictionsRFC, y_test)
print("RFC Model:")
print(f"Accuracy for Predicting Wins: {accuracy_wins_rfc:.2%}")
print(f"Accuracy for Predicting Draws: {accuracy_draws_rfc:.2%}")
print(f"Accuracy for Predicting Away Outcomes: {accuracy_away_rfc:.2%}")
print()

# Calculate and print accuracy for DTC
accuracy_wins_dtc, accuracy_draws_dtc, accuracy_away_dtc = calculate_accuracy(predictionsDTC, y_test)
print("DTC Model:")
print(f"Accuracy for Predicting Wins: {accuracy_wins_dtc:.2%}")
print(f"Accuracy for Predicting Draws: {accuracy_draws_dtc:.2%}")
print(f"Accuracy for Predicting Away Outcomes: {accuracy_away_dtc:.2%}")

probabilitiesMLP = MLP.predict_proba(X_test)
print("MLP Probabilities:")
print(probabilitiesMLP)

probabilitiesRFC = RFC.predict_proba(X_test)
print("RFC Probabilities:")
print(probabilitiesRFC)

probabilitiesDTC = DTC.predict_proba(X_test)
print("DTC Probabilities:")
print(probabilitiesDTC)


# Function to calculate bootstrap confidence intervals
def bootstrap_confidence_intervals(data, n_iterations=1000, confidence_level=0.95):
    bootstrapped_means = []
    for _ in range(n_iterations):
        sample = resample(data)
        bootstrapped_means.append(np.mean(sample, axis=0))
    lower_bound = np.percentile(bootstrapped_means, ((1 - confidence_level) / 2) * 100, axis=0)
    upper_bound = np.percentile(bootstrapped_means, (confidence_level + (1 - confidence_level) / 2) * 100, axis=0)
    return lower_bound, upper_bound

# Example probabilities from your models
probabilitiesMLP = np.array([
    [0.46664946, 0.24956841, 0.28378213],
    [0.13501781, 0.65453348, 0.21044871],
    [0.30758097, 0.43945894, 0.25296009],
    [0.46501707, 0.40810934, 0.12687359],
    [0.20756221, 0.47861934, 0.31381846],
    [0.22959142, 0.08314073, 0.68726785],
    [0.97452019, 0.02312746, 0.00235235],
    [0.06913474, 0.51077016, 0.4200951 ],
    [0.43190205, 0.17023299, 0.39786496],
    [0.11100931, 0.11935053, 0.76964016],
    [0.38635956, 0.11480993, 0.49883051],
    [0.4411094, 0.19475506, 0.36413554],
    [0.43777561, 0.48050163, 0.08172276],
    [0.44841856, 0.39703307, 0.15454838],
    [0.08195014, 0.50307939, 0.41497047],
    [0.12061351, 0.54717413, 0.33221236],
    [0.46413274, 0.21375786, 0.3221094 ],
    [0.84072279, 0.13204541, 0.0272318 ],
    [0.10280572, 0.51760183, 0.37959245],
    [0.96832256, 0.02862217, 0.00305527],
    [0.06294874, 0.52390893, 0.41314233],
    [0.3148942, 0.29297758, 0.39212821],
    [0.37063113, 0.10617429, 0.52319458],
    [0.48069198, 0.32769415, 0.19161387],
    [0.4352315, 0.4020732, 0.1626953 ],
    [0.4180173, 0.13946629, 0.4425164 ]
])

probabilitiesRFC = np.array([
    [0.45222107, 0.27554802, 0.27223091],
    [0.24267322, 0.45742013, 0.29990666],
    [0.44417179, 0.28509367, 0.27073454],
    [0.64478403, 0.22402616, 0.1311898 ],
    [0.39899599, 0.27567029, 0.32533372],
    [0.27077881, 0.25585381, 0.47336738],
    [0.67588836, 0.1746864, 0.14942525],
    [0.41367854, 0.34895565, 0.23736581],
    [0.30300073, 0.22383499, 0.47316427],
    [0.24733539, 0.27601859, 0.47664602],
    [0.38720871, 0.21600843, 0.39678286],
    [0.49185782, 0.23186796, 0.27627422],
    [0.50891764, 0.32363456, 0.16744779],
    [0.40165935, 0.36654258, 0.23179806],
    [0.42190938, 0.3346093, 0.24348132],
    [0.3621987, 0.34425543, 0.29354587],
    [0.44988448, 0.26334863, 0.28676689],
    [0.67522024, 0.20324646, 0.1215333 ],
    [0.39018319, 0.36853538, 0.24128143],
    [0.67535082, 0.21632993, 0.10831925],
    [0.39681594, 0.36992715, 0.23325691],
    [0.3355534, 0.33133209, 0.3331145 ],
    [0.37238779, 0.21342679, 0.41418542],
    [0.48648944, 0.29453668, 0.21897388],
    [0.47786171, 0.32946328, 0.19267501],
    [0.46526391, 0.2259397, 0.3087964 ]
])

probabilitiesDTC = np.array([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
    [0., 0., 1.],
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 0.],
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [0., 1., 0.],
    [1., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.]
])

# Calculate confidence intervals for each model
lower_mlp, upper_mlp = bootstrap_confidence_intervals(probabilitiesMLP)
lower_rfc, upper_rfc = bootstrap_confidence_intervals(probabilitiesRFC)
lower_dtc, upper_dtc = bootstrap_confidence_intervals(probabilitiesDTC)

# Print the confidence intervals
print("MLP Confidence Intervals:")
print("Lower bounds:", lower_mlp)
print("Upper bounds:", upper_mlp)
print()

print("RFC Confidence Intervals:")
print("Lower bounds:", lower_rfc)
print("Upper bounds:", upper_rfc)
print()

print("DTC Confidence Intervals:")
print("Lower bounds:", lower_dtc)
print("Upper bounds:", upper_dtc)