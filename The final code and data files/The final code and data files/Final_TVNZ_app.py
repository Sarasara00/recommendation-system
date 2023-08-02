import streamlit as st
import pandas as pd
import itertools
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import requests
from io import BytesIO
from random import random
import time
import webbrowser

st.set_page_config(layout="wide")
pd.set_option('display.max_colwidth', None)

#upload the data 
#Movies data set 
df_movies= pd.read_csv('movies_data.csv', sep=',')
#fake users data set
df_fakeuser = pd.read_csv('fake_users_data.csv', dtype={'user_id': 'str'})

#getting the user_id and his view_history splitted in new data frame
df_fakeuser['view_history'] = df_fakeuser['view_history'].str.split(', ')
df = df_fakeuser.explode('view_history')
df=df[['user_id','view_history']]

#we clean these qoutation here in order to not loss data if we clean them when the title where toghether as a list
df['view_history']=df['view_history'].str.strip('"')
df['view_history']=df['view_history'].str.strip("'")

#defining functions part

#user_based recommendation function
def get_jaccard_recommendations(id):
    user_df = df_fakeuser.groupby('user_id')['view_history'].apply(list)
    new_content= []
    similar_users= []
    
    mother_language_id=df_fakeuser['mother_language'][df_fakeuser['user_id']==id].iloc[0]

    for user, value in user_df.items():
        mother_language_user= df_fakeuser['mother_language'][df_fakeuser['user_id']==user].iloc[0]
        
        if mother_language_id == mother_language_user:
            a = df[df['user_id']==id]
            b = df[df['user_id']==user]
            a=set(a['view_history'])
            b=set(b['view_history'])

            # calculate jaccard
            distance =1- float(len(a.intersection(b))) / len(a.union(b))

            # tweak this parameter. Closer to 0.0 is more the same. 0.0 is the user.
            if distance < 0.9 and distance != 0.0:
                new = b.difference(a)
                new_content.append(new)
                similar_users.append(user)

    # flatten the list with the sets
    new_content = list(itertools.chain(*new_content))

    # select the movies
    df_recommendations = df[df['user_id'].isin(similar_users) & df['view_history'].isin(new_content)]
    # in case there are duplicated movies name 
    if df_recommendations['view_history'].duplicated().sum() != 0:
        recommendations=df_recommendations['view_history'].drop_duplicates().tolist()
    else:
        recommendations=df_recommendations['view_history'].head(10).tolist()
        
    # in case there are no recommendations (no similarity between the user and the users who have the same mother language)
    zero_recom_content=[]
    if len(recommendations)==0:
        
        for user, value in user_df.items():
            mother_language_user= df_fakeuser['mother_language'][df_fakeuser['user_id']==user].iloc[0]
        
            if mother_language_id == mother_language_user:
                b = df[df['user_id']==user]
                b=set(b['view_history'])
            
                zero_recom_content.append(b)
                
        zero_recom_content = list(itertools.chain(*zero_recom_content))
        df_recommendations = df[df['view_history'].isin(zero_recom_content)]
        
        # in case there are duplicated movies name 
        if df_recommendations['view_history'].duplicated().sum() != 0:
            
            recommendations=df_recommendations['view_history'].drop_duplicates().tolist()
        else:
            recommendations=df_recommendations['view_history'].head(10).tolist()

    return recommendations

#prepartion for the item_based and nearest neighbours recommendations
#prepare the data
user_movie_matrix = df.pivot_table(index='user_id', columns='view_history', aggfunc=lambda x: 1, fill_value=0)
# Drop movies that no one has watched
user_movie_matrix = user_movie_matrix.loc[:, (user_movie_matrix != 0).any(axis=0)]
#tanspose the usermovie matrix (the index is the movies and users are the columns)
item_movie_matrix = user_movie_matrix.T
# Calculate item similarity matrix
item_similarity_matrix = pd.DataFrame(cosine_similarity(item_movie_matrix))
#change the name of the index and columns
item_similarity_matrix.index = item_movie_matrix.index
item_similarity_matrix.columns = item_movie_matrix.index

#item_based recommendation function
def get_recommendations_IBF(user_id):
    #watched_items(output has the length of all item_movie_matrix index )
    watched_items = item_movie_matrix.loc[:, user_id]
    
    item_weights = item_similarity_matrix.dot(watched_items)/(item_similarity_matrix.sum(axis=1))
    #get Weighted Sum of the watched_items
    weights=item_weights[item_weights !=0].sort_values(ascending=False)
    weights=pd.DataFrame(weights)    
    
    weights = weights.reset_index()
    weights = weights.rename(columns = {'view_history':'movies',0:'weight'})
    
    #watched_items(just the watched items without the unwatched)
    user_watched_items=watched_items[item_movie_matrix.loc[:,user_id] != 0]
    
    recommendations = pd.DataFrame(columns=['movies', 'weight'])

    for i in range(len(weights)):
        if weights['movies'][i] not in user_watched_items.index:
            new_row = {'movies': weights['movies'][i], 'weight': weights['weight'][i]}
            recommendations = pd.concat([recommendations, pd.DataFrame(new_row, index=[0])], ignore_index=True)
            
    
    recommendations=recommendations['movies'].head(10).tolist()        
    
    return recommendations

#nearest neighbours recommendations
knn = NearestNeighbors(metric='cosine', algorithm='brute')

knn.fit(item_movie_matrix.values)

distances, indices = knn.kneighbors(item_movie_matrix.values, n_neighbors=5)

#get the neighbours of all movies 
df_neighbours = pd.DataFrame(columns=['movies', 'neighbours'])

for i in range(0, len(indices)):
    nn = indices[i] 
    movie = item_movie_matrix.index[nn[0]]
    neighbours=[]
    for n in nn[1:]:
        neighbours.append(item_movie_matrix.index[n])
    
    df_neighbours.loc[i]=[movie,neighbours]
    
df_neighbours['neighbours'] = df_neighbours['neighbours'].astype(str).str.replace('[', '', regex = True).str.replace(']', '', regex = True)

#nearest neighbours recommendations function
def get_nn_movies(user_id):
    user_watched_items = item_movie_matrix.loc[:, user_id][item_movie_matrix.loc[:,user_id] != 0]
    
    df_nn_movies_user=pd.DataFrame(columns=['movie', 'neighbours'])

    for i in range(len(user_watched_items.index)):
        movie_name = user_watched_items.index[i]
        neighbours=df_neighbours.loc[df_neighbours['movies'] == movie_name, 'neighbours'].iloc[0]
        df_nn_movies_user.loc[i]=[movie_name,neighbours]
        
    df_nn_movies_user['neighbours'] = df_nn_movies_user['neighbours'].str.split(', ')
    df_nn_movies_user = df_nn_movies_user.explode('neighbours')
    
    df_nn_movies_user['neighbours']=df_nn_movies_user['neighbours'].str.strip('"')
    df_nn_movies_user['neighbours']=df_nn_movies_user['neighbours'].str.strip("'")
    
            
    return df_nn_movies_user

#nearest neighbours of the movie the user wants to watch (the selected movie whic is saved in the session state)
def nn_watch_recommendation(movie_name):
    nn_movie=df_neighbours.loc[df_neighbours['movies'] == movie_name].copy()
    nn_movie['neighbours'] = nn_movie['neighbours'].str.split(', ')
    nn_movie = nn_movie.explode('neighbours')
    
    nn_movie['neighbours']=nn_movie['neighbours'].str.strip('"')
    nn_movie['neighbours']=nn_movie['neighbours'].str.strip("'")
   
    return nn_movie['neighbours'].tolist()

#content_based recommendations (based on the user interests)
def get_recommendations_content_based(user_id):
    user_interests= df_fakeuser['interests'][df_fakeuser['user_id']==user_id].iloc[0].split(', ')
    pd_interests=pd.DataFrame(columns=['interest', 'movie'])
    for i in df_movies['title'].tolist():
        tags_list=df_movies['tags'][df_movies['title']==i].iloc[0].replace("'", '').split(', ')
        for j in tags_list :
            if j in user_interests:
                new_row={'interest': j , 'movie': i}
                pd_interests=pd.concat([pd_interests, pd.DataFrame(new_row, index=[0])], ignore_index=True)
                
    return pd_interests

#this function will display the movies recommendations based on the provided list
def display_recommendations(movies_list):
    recommended_movies = df_movies[df_movies['title'].isin(movies_list)]
            #st.subheader('Movies that are similar to this' )
    num_cols = len(recommended_movies)
    image_size = (480, 720)
    cols = st.columns(num_cols)
    for i, col in enumerate(cols):
        # Download the movie image from its URL
        response = requests.get(recommended_movies["image"].iloc[i])
        image = Image.open(BytesIO(response.content))

        # Resize the image to the desired size
        image = image.resize(image_size)

        # Display the image in the current column
        col.image(image, caption=recommended_movies["title"].iloc[i], use_column_width=True)
        watch = col.button('watch', key=random(), on_click=select_movie,
                            args=[recommended_movies["title"].iloc[i]])


# the authentication function
def authenticate(user_id, password):
    #check the user_id
    ids= df_fakeuser['user_id'].tolist()
    #check that the password is related to the user_id
    passwords=df_fakeuser['password'][df_fakeuser['user_id']==user_id].iloc[0]
    if  user_id in ids and password == passwords:
        return True
    else:
        return False
    
#save the selected movie in the session state    
def select_movie(title):
  st.session_state['title'] = title


#start The interface 
login = st.sidebar.container()
with login:
        st.title("Welcome to TVNZ ")
        st.image("Tvnz.png")
        st.write("Please log in to continue.")

        # create input fields for the username and password
        id_input=st.text_input("User ID")
        pass_input = st.text_input("Password", type="password")

        # create a button to submit the login form
        login_button = st.button("Log In")

        if login_button:
            # authenticate the user
            if authenticate(id_input, pass_input):
                st.session_state['id_input'] = id_input
                st.success("Logged in as user {}".format(id_input), icon="✅")
            else:
                st.error('The combination of User Id, Username and Password is invalid', icon="🚨")

#start the recommendations
recommendations = st.container()
if 'id_input' not in st.session_state:
    st.session_state['id_input'] = None
    recommendations.empty()

if st.session_state['id_input']!=None:
    with recommendations:

        # select a movie to kickstart the interface
        if 'title' not in st.session_state:
            st.session_state['title'] = 'Mammuth'

        df_movie = df_movies[df_movies['title'] == st.session_state['title']]

        # create a cover and info column to display the selected movie
        cover, info = st.columns([2, 3])
        with cover:
            # display the image
            response = requests.get(df_movie['image'].iloc[0])
            image = Image.open(BytesIO(response.content))
            image_size = (480, 720)
            image = image.resize(image_size)
            st.image(image, use_column_width='always')

        with info:
            # display the movie information
            st.title(df_movie['title'].iloc[0])
            st.markdown(df_movie['tags'].iloc[0])
            st.write(str(df_movie['description'].iloc[0]))
            subtitles = st.selectbox('Choose the subtitles language',
                                     ('without subtitles', 'English', 'Māori','Arabic', 'chinese', 'Russian', 'French',
                                      'spanish'))
            play = st.button('Play Movie')
            if play:
                webbrowser.open_new_tab(df_movie['image'].iloc[0])

            Rate= st.slider('Rate',1,5,3)

            nearest_neighbour_movies=nn_watch_recommendation(st.session_state['title'])
            st.subheader('Movies that are similar to this' )
            my_expander = st.expander(label='More information about the recommentions generation process')
            with my_expander:
                    """The recommended movies here are based on finding movies that are similar to the ones you've selected, and suggests them to you.
                          It looks for the movies that are similar to the one you've selected to watch. Then, it suggests those similar movies to you."""
            display_recommendations(nearest_neighbour_movies)
            

        st.title("Movie Recommendations")
        # get user_based recommendations
        st.subheader('Recommendations based on users who speaks your mother language')
        my_expander = st.expander(label='More information about the recommentions generation process')
        with my_expander:
                """These recommendtions appears to you based on what other people who speak the same language as you have watched. 
                It looks at what movies you have already watched and compares it to what other people have watched to suggest similar movies."""
        UBR = get_jaccard_recommendations(id_input)
        display_recommendations(UBR)
        
        # item-Based recommendations based on all view history
        st.subheader('Recommendations Based on the movies you\'ve watched,')
        my_expander = st.expander(label='More information about the recommentions generation process')
        with my_expander:
                """The recommended movies here are based on your previously watched movies by calculating their similarity.
                  It suggests movies that are most similar to the ones you have already watched."""
        IBR = get_recommendations_IBF(id_input)
        display_recommendations(IBR)


        #content_based recommendations based on user interests type
        CBR=get_recommendations_content_based(id_input)
        interest_list=list(set(CBR['interest']))
        for i in interest_list:
            interest_tag= i
            movie_interest=CBR['movie'][CBR['interest']==i].tolist()[0:10] # get just 10 recommendations of each type
            st.subheader('Movies you may like based on your interest in  '+  interest_tag +' Movies')
            my_expander = st.expander(label='More information about the recommentions generation process')
            with my_expander:
                    """The movies that are recommended to you here based on what kinds of movies you like. 
                    It looks at the types of movies you're interested in, and recommends movies that have similar types."""
            display_recommendations(movie_interest)


        # nearest-neighbour movies for each movie item-Based recommendation
        nn_movie = get_nn_movies(id_input)
        nn_movie_list = list(set(nn_movie['movie']))
        #for i in range(len(nn_movie_list)): #loop over all the view history and present all the nn for each movie in the view history 
        for i in range(2): # we loop just over the first two movies in the view history 
            movie_name = nn_movie_list[i]
            neighbour_movies = nn_movie.loc[nn_movie['movie'] == movie_name, 'neighbours'].tolist()
            st.subheader('You\'ve watched '+  movie_name +', '+'these movies are similer to it')
            my_expander = st.expander(label='More information about the recommentions generation process')
            with my_expander:
                    """The recommended movies here are based on finding movies that are similar to the ones you've watched, and suggests them to you.
                          It looks at movies you've seen before, and finds other movies that are similar. Then, it suggests those similar movies to you."""
            display_recommendations(neighbour_movies)
            











            

          



