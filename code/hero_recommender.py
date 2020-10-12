import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras.models import load_model

from flask import Flask, request, Response, render_template

# initialize the flask app
app = Flask('hero_recommender')

# importing the cleaned match data
match_df = pd.read_csv("../data/complete_match_df.csv")

rad_cvec = CountVectorizer()
dire_cvec = CountVectorizer()
rad_df = pd.DataFrame(rad_cvec.fit_transform(match_df['radiant']).toarray() , columns = rad_cvec.get_feature_names())
dire_df = pd.DataFrame(dire_cvec.fit_transform(match_df['dire']).toarray() , columns = dire_cvec.get_feature_names())
match_hero_df = pd.concat([rad_df,dire_df], axis=1)

X = match_hero_df
y = match_df['winner'].map({'radiant' : 1 , 'dire' : 0})

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=2020)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)



@app.route('/')
def form():
    return render_template('form.html')

@app.route('/submit')
def submit():

    #importing the trained NN to generate recommendations
    NN_model = load_model("../data/NN_model/NN_model")

    test_game_df = pd.DataFrame()
    data = request.args
    for key,value in data.items():
        test_game_df.loc[0,key] = value

    test_game_df.replace("",np.NaN,inplace=True)

    rad_heroes = ''
    dire_heroes = ''

    if test_game_df.isna().any().any():
        test_game_df.dropna(axis=1,inplace=True)

    for x in test_game_df.columns:
        if "radiant" in x:
            rad_heroes += (' r_'+ test_game_df[x])
        else:
            dire_heroes += (' d_'+ test_game_df[x])

    test_game_df['radiant'] = rad_heroes[0]
    test_game_df['dire'] = dire_heroes[0]

    unused_radiant_heroes_list = [x for x in rad_cvec.vocabulary_.keys() if (x[2:] not in rad_heroes.any().split(' r_')) and (x[2:] not in dire_heroes.any().split(' d_'))]
    unused_dire_heroes_list = [x for x in dire_cvec.vocabulary_.keys() if (x[2:] not in rad_heroes.any().split(' r_')) and (x[2:] not in dire_heroes.any().split(' d_'))]

    probabilities_df = pd.DataFrame()

    if (len(rad_heroes[0].split()) == 4) and (len(dire_heroes[0].split()) == 5):
        #radiant team hero missing
        temp_dire_df = pd.DataFrame(dire_cvec.transform(test_game_df['dire']).toarray() , columns = dire_cvec.get_feature_names())
        for hero in unused_radiant_heroes_list:
            team = test_game_df['radiant'].values[0]
            team += (' ' + hero)

            test_game_df['radiant_team'] = team
            temp_rad_df = pd.DataFrame(rad_cvec.transform(test_game_df['radiant_team']).toarray() , columns = rad_cvec.get_feature_names())
            temp_match_hero_df = pd.concat([temp_rad_df,temp_dire_df], axis=1)

            X_team_sc = sc.transform(temp_match_hero_df)

            probabilities_df.loc['prob',hero] = NN_model.predict_proba(X_team_sc)[0][0]

        pred = probabilities_df.T.sort_values(by='prob',ascending=False).head(5)

    elif (len(rad_heroes[0].split()) == 5) and (len(dire_heroes[0].split()) == 4):
        #dire team hero missing
        temp_rad_df = pd.DataFrame(rad_cvec.transform(test_game_df['radiant']).toarray() , columns = rad_cvec.get_feature_names())

        for hero in unused_dire_heroes_list:
            team = test_game_df['dire'].values[0]
            team += (' ' + hero)

            test_game_df['dire_team'] = team
            temp_dire_df = pd.DataFrame(dire_cvec.transform(test_game_df['dire_team']).toarray() , columns = dire_cvec.get_feature_names())
            temp_match_hero_df = pd.concat([temp_rad_df,temp_dire_df], axis=1)

            X_team_sc = sc.transform(temp_match_hero_df)

            probabilities_df.loc['prob',hero] = (1-NN_model.predict_proba(X_team_sc)[0][0])

        pred = probabilities_df.T.sort_values(by='prob',ascending=False).head(5)

    else:
        pred = "Please only have 1 missing hero"

    if len(pred) == 5:
        rec_1_hero = pred.index[0][2:]
        rec_1_prob = round(pred.iloc[0,0],4)
        rec_2_hero = pred.index[1][2:]
        rec_2_prob = round(pred.iloc[1,0],4)
        rec_3_hero = pred.index[2][2:]
        rec_3_prob = round(pred.iloc[2,0],4)
        rec_4_hero = pred.index[3][2:]
        rec_4_prob = round(pred.iloc[3,0],4)
        rec_5_hero = pred.index[4][2:]
        rec_5_prob = round(pred.iloc[4,0],4)

        return render_template('results.html', rec_1_hero=rec_1_hero, rec_1_prob=rec_1_prob, rec_2_hero=rec_2_hero, rec_2_prob=rec_2_prob, rec_3_hero=rec_3_hero, rec_3_prob=rec_3_prob, rec_4_hero=rec_4_hero, rec_4_prob=rec_4_prob, rec_5_hero=rec_5_hero, rec_5_prob=rec_5_prob)

    else:
        return pred

if __name__ == '__main__':
    app.run(debug=True)
