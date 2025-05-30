import streamlit as st #streamlit library is used to create web applications
from streamlit_option_menu import option_menu 
import requests #requests library is used to send HTTP requests
import seaborn as sns #seaborn library is used for data visualization
import matplotlib.pyplot as plt #matplotlib library is used for data visualization
import pandas as pd #pandas library is used for data manipulation
import random 
import nltk #nltk library is used for natural language processing
import re #re library is used for regular expressions
import streamlit as st
import PyPDF2
import re
import string
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import joblib
import sys
import streamlit as st
import pdfplumber
from Resume_scanner import compare
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
model=joblib.load('resume_screening_model.pkl')
def extract_pdf_data(file_path):
    data = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += text
    return data


def extract_text_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfFileReader(pdf_file)
    number_of_pages = reader.numPages
    content = ""
    for page_number in range(number_of_pages):
        page = reader.getPage(page_number)
        content += page.extractText()
    return content
#load the model
# Function to preprocess text
def preprocess_text(content):
    content = content.lower()
    content = re.sub(r'[0-9]+', '', content)
    content = content.translate(str.maketrans('', '', string.punctuation))
    return content
def search_and_display_images(query, num_images=20):
    try:
        # Initialize an empty list for image URLs
        k=[]  
        # Initialize an index for iterating through the list of images
        idx=0  
        # Construct Google Images search URL
        url = f"https://www.google.com/search?q={query}&tbm=isch"  
         # Make an HTTP request to the URL
        response = requests.get(url) 
        # Parse the HTML content with BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")  
        # Initialize an empty list for storing image URLs
        images = []  
        # Iterate through image tags in the HTML content
        for img in soup.find_all("img"):  
             # Limit the number of images to the specified amount
            if len(images) == num_images: 
                break
            # Get the image source URL
            src = img.get("src")  
            # Check if the source URL is valid
            if src.startswith("http") and not src.endswith("gif"):  
                # Add the image URL to the list
                images.append(src)  
        # Iterate through the list of image URLs
        for image in images:  
            # Add each image URL to the list 'k'
            k.append(image)  
        # Reset the index for iterating through the list of image URLs
        idx = 0  
        # Iterate through the list of image URLs
        while idx < len(k):
            # Iterate through the columns in a 4-column layout 
            for _ in range(len(k)): 
                # Create 4 columns for displaying images 
                cols = st.columns(3)  
                # Display the first image in the first column
                cols[0].image(k[idx], width=200)  
                idx += 1 
                # Move to the next image in the list
                cols[1].image(k[idx], width=200)
                # Display the second image in the second column
                idx += 1  
                # Move to the next image in the list
                cols[2].image(k[idx], width=200)  
                # Display the third image in the third column
                idx += 1  
    except:
         # Handle exceptions gracefully if there is an error while displaying images
        pass  
YOUTUBE_API_KEY = "AIzaSyDYEeSTrT7pPpVzpmaJ491gxogVxfWwpvM"

def fetch_youtube_videos(query, max_results=12):
    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'key': YOUTUBE_API_KEY,
        'maxResults': max_results
    }
    response = requests.get(url, params=params)
    videos = []
    if response.status_code == 200:
        data = response.json()
        for item in data['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            videos.append({'video_id': video_id, 'title': video_title})
    return videos

# Function to calculate scores
def calculate_scores(content):
    model=joblib.load('resume_screening_model.pkl')
    Area_with_key_term = {
        'Data science': ['algorithm', 'analytics', 'hadoop', 'machine learning', 'data mining', 'python',
                        'statistics', 'data', 'statistical analysis', 'data wrangling', 'algebra', 'Probability',
                        'visualization'],
        'Programming': ['python', 'r programming', 'sql', 'c++', 'scala', 'julia', 'tableau', 'javascript',
                        'powerbi', 'code', 'coding'],
        'Experience': ['project', 'years', 'company', 'excellency', 'promotion', 'award', 'outsourcing', 'work in progress'],
        'Management skill': ['administration', 'budget', 'cost', 'direction', 'feasibility analysis', 'finance', 
                            'leader', 'leadership', 'management', 'milestones', 'planning', 'problem', 'project', 
                            'risk', 'schedule', 'stakeholders', 'English'],
        'Data analytics': ['api', 'big data', 'clustering', 'code', 'coding', 'data', 'database', 'data mining', 
                        'data science', 'deep learning', 'hadoop', 'hypothesis test', 'machine learning', 'dbms', 
                        'modeling', 'nlp', 'predictive', 'text mining', 'visualization'],
        'Statistics': ['parameter', 'variable', 'ordinal', 'ratio', 'nominal', 'interval', 'descriptive', 
                        'inferential', 'linear', 'correlations', 'probability', 'regression', 'mean', 'variance', 
                        'standard deviation'],
        'Machine learning': ['supervised learning', 'unsupervised learning', 'ann', 'artificial neural network', 
                            'overfitting', 'computer vision', 'natural language processing', 'database'],
        'Data analyst': ['data collection', 'data cleaning', 'data processing', 'interpreting data', 
                        'streamlining data', 'visualizing data', 'statistics', 'tableau', 'tables', 'analytical'],
        'Software': ['django', 'cloud', 'gcp', 'aws', 'javascript', 'react', 'redux', 'es6', 'node.js', 
                    'typescript', 'html', 'css', 'ui', 'ci/cd', 'cashflow'],
        'Web skill': ['web design', 'branding', 'graphic design', 'seo', 'marketing', 'logo design', 'video editing', 
                    'es6', 'node.js', 'typescript', 'html/css', 'ci/cd'],
        'Personal Skill': ['leadership', 'team work', 'integrity', 'public speaking', 'team leadership', 
                            'problem solving', 'loyalty', 'quality', 'performance improvement', 'six sigma', 
                            'quality circles', 'quality tools', 'process improvement', 'capability analysis', 
                            'control'],
        'Accounting': ['communication', 'sales', 'sales process', 'solution selling', 'crm', 'sales management', 
                    'sales operations', 'marketing', 'direct sales', 'trends', 'b2b', 'marketing strategy', 
                    'saas', 'business development'],
        'Sales & marketing': ['retail', 'manufacture', 'corporate', 'goods sale', 'consumer', 'package', 'fmcg', 
                            'account', 'management', 'lead generation', 'cold calling', 'customer service', 
                            'inside sales', 'sales', 'promotion'],
        'Graphic': ['brand identity', 'editorial design', 'design', 'branding', 'logo design', 'letterhead design', 
                    'business card design', 'brand strategy', 'stationery design', 'graphic design', 'exhibition graphic design'],
        'Content skill': ['editing', 'creativity', 'content idea', 'problem solving', 'writer', 'content thinker', 
                        'copy editor', 'researchers', 'technology geek', 'public speaking', 'online marketing'],
        'Graphical content': ['photographer', 'videographer', 'graphic artist', 'copywriter', 'search engine optimization', 
                            'seo', 'social media', 'page insight', 'gain audience'],
        'Finanace': ['financial reporting', 'budgeting', 'forecasting', 'strong analytical thinking', 'financial planning', 
                    'payroll tax', 'accounting', 'productivity', 'reporting costs', 'balance sheet', 'financial statements'],
        'Health/Medical': ['abdominal surgery', 'laparoscopy', 'trauma surgery', 'adult intensive care', 'pain management', 
                        'cardiology', 'patient', 'surgery', 'hospital', 'healthcare', 'doctor', 'medicine'],
        'Language': ['english', 'malay', 'mandarin', 'bangla', 'hindi', 'tamil']
    }
    model = {domain: sum(1 for word in terms if word in content) for domain, terms in Area_with_key_term.items()}
    return pd.DataFrame(list(model.items()), columns=['Domain/Area', 'Score']).sort_values(by='Score', ascending=False)

# Main UI
st.set_page_config(page_title="AI Recruitment System", page_icon=":memo:")


with st.sidebar:
    page = option_menu("DashBoard", ["Home",'Resume Screening','Candidate Matching','Interview Process','Interview Preparation'], 
        icons=['house','newspaper','person-check','building','person-bounding-box'], menu_icon="cast", default_index=0,
        styles={
        "nav-link-selected": {"background-color": "#5bd9c4", "color": "black", "border-radius": "5px"},
})

if page == "Home":
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: violet;'>Leveraging</span> 
        <span style='color: green;'>AI</span> 
        <span style='color: indigo'>in</span> 
        <span style='color: red;'>Recruitment</span>
    </h1>
    """, 
    unsafe_allow_html=True
    )
    st.image("https://www.recruiter.com/recruiting/wp-content/uploads/2023/05/23HIRE-Leveraging-AI-Featured-Image-1-scaled.webp", use_column_width=True)
if page == "Resume Screening":
    st.markdown(
    """ 
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: indigo;'>Resume</span>
        <span style='color: indigo;'>Parsing</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    uploaded_file = st.file_uploader("Upload a Resume (PDF)", type="pdf")

    if uploaded_file:
        # Extract text
        # Extract and preprocess the text
        content = extract_text_from_pdf(uploaded_file)
        content = preprocess_text(content)

        # Calculate scores
        scored_df = calculate_scores(content)

        # Filter high scores (e.g., scores greater than 80)
        high_scores_df = scored_df[scored_df['Score'] > 1]

        # Display high scores
        st.markdown(
            """
            <p style='font-size: 30px;'>
                <span style='color: red;'>Resume</span>
                <span style='color: red;'>Scores</span>
            </p>
            """,
            unsafe_allow_html=True
        )
        if not high_scores_df.empty:
            # display in 3 cols, each col has skill and score in percentage
            cols = st.columns(3)
            for i, row in high_scores_df.iterrows():
                cols[i % 3].write(f"{row['Domain/Area']}")
                
                # Check if the Score is a fraction (0 to 1) or a percentage (0 to 100)
                if row['Score'] <= 1:
                    # Score is a fraction, so multiply by 100 to get percentage
                    score_percentage = row['Score'] * 100
                else:
                    # Score is already a percentage
                    score_percentage = row['Score']  # Do not multiply by sum(scored_df['Score'])
                
                # Clamp the score between 0 and 100
                score_percentage = max(0, min(100, score_percentage))

                # Display progress bar with the score
                cols[i % 3].progress(score_percentage)

            # Filter out rows where the score is 0
            filtered_df = scored_df[scored_df['Score'] > 0]

            # Visualization: Bar chart for scores by domain
            st.markdown(
                """
                <p style='font-size: 30px;'>
                    <span style='color: indigo;'>Domain</span>
                    <span style='color: indigo;'>Expertise</span>
                </p>
                """,
                unsafe_allow_html=True
            )

            # Set the figure size
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create a bar chart using seaborn to get a nice color palette
            sns.barplot(x='Domain/Area', y='Score', data=filtered_df, ax=ax, palette='viridis')

            # Rotate the x-axis labels for better visibility
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            # Set labels and title
            ax.set_xlabel('Domain')
            ax.set_ylabel('Score')

            # Display the plot
            st.pyplot(fig)


            # Analyze and recommend
            total_score = scored_df['Score'].sum()
            st.markdown(
                """
                <p style='font-size: 30px;'>
                    <span style='color: green;'>Application Status</span>
                </p>
                """,
                unsafe_allow_html=True
            )
            # if data scientist has high score in data science, programming, statistics, machine learning, data analytics
            if total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Data science', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Programming', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Statistics', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Machine learning', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Data analytics', 'Score'].values[0] >= 1:
                st.write("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Data Scientist.")
            # if account executive has high score in sales & marketing, personal skill, management skill, experience
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Sales & marketing', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Personal Skill', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Management skill', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Experience', 'Score'].values[0] >= 1:
                st.write("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Account Executive.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Content skill', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Graphic', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Personal Skill', 'Score'].values[0] >= 1:
                st.write("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Content Creator.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Data analytics', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Statistics', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Data analyst', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Data science', 'Score'].values[0] >= 1:
                st.write("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Data Analyst.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Sales & marketing', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Accounting', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Management skill', 'Score'].values[0] >= 1 :
                st.write("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Sales Executive.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Programming', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Software', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Experience', 'Score'].values[0] >= 1:
                st.write("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Software Engineer.")
            elif total_score >= 50 and scored_df.loc[scored_df['Domain/Area'] == 'Web skill', 'Score'].values[0] >= 2 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Graphic', 'Score'].values[0] >= 1 \
                    and scored_df.loc[scored_df['Domain/Area'] == 'Personal Skill', 'Score'].values[0] >= 1:
                st.write("Congratulations! 👏 Resume Meets The Requirement. Suggest To Recruit as Web and Graphic Designer.")
            else:
                st.write("Resume Does Not Meet The Requirement for any Role Based on Scores.")
    
        else:
            st.write("Resume Does Not Meet The Requirement for any Role.")
elif page == "Candidate Matching":
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: red;'>Skill Based Matching</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    flag = 'HuggingFace-BERT'
    uploaded_files = st.file_uploader(
        '**Upload your Resume File:** ', type="pdf", accept_multiple_files=True)
    JD = st.text_area("**Enter Job Description:**")
    comp_pressed = st.button("Get Score")
    if comp_pressed and uploaded_files:
        # Streamlit file_uploader gives file-like objects, not paths
        uploaded_file_paths = [extract_pdf_data(
            file) for file in uploaded_files]
        score = compare(uploaded_file_paths, JD, flag)
    my_dict = {}
    if comp_pressed and uploaded_files:
        for i in range(len(score)):
            # Populate the dictionary with file names and corresponding scores
            my_dict[uploaded_files[i].name] = score[i]
        
        # Sort the dictionary by keys
        sorted_dict = dict(sorted(my_dict.items()))
        
        # Convert the sorted dictionary to a list of tuples
        ct_items = list(sorted_dict.items())
        score = ct_items[0][1]  # Access the score of the first file
        sc=float(score)
        if sc >= 75:
            st.markdown(
                """
                <h1 style='text-align: center; font-size: 50px;'>
                    <span style='color: green;'>The Candiate is good match for the Job.</span>
                </h1>
                """,
                unsafe_allow_html=True
            )
            st.image("https://media.istockphoto.com/id/1385218939/vector/people-crowd-applause-hands-clapping-business-teamwork-cheering-ovation-delight-vector.jpg?s=612x612&w=0&k=20&c=7NMaUB4zGoXoePxiy-XxKap53GMBQvmIYOSW1tVSFMY=", use_column_width=True)
            st.balloons()
        elif sc >= 50:
            st.markdown(
                """
                <h1 style='text-align: center; font-size: 50px;'>
                    <span style='color: orange;'>The Candiate is moderate match for the Job.</span>
                </h1>
                """,
                unsafe_allow_html=True
            )
            # place horizontal line
            st.markdown(
                """
                <hr style='border: 1px solid green;'>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <p style='text-align: center; font-size: 30px; font-family: Garamond, sans-serif;'>
                    <span style='color: purple;'>Improvements</span>
                </p>
                """,
                unsafe_allow_html=True
            )
            #parse Job Description and Extract the skills
            JD = preprocess_text(JD)
            scored_df = calculate_scores(JD)
            high_scores_df = scored_df[scored_df['Score'] > 1]
            if not high_scores_df.empty:
                #make a list of skills
                data = pd.read_csv('image.csv')
                # Number of columns per row
                columns_per_row = 3
                # Count the total number of rows needed
                total_items = len(high_scores_df)
                total_rows = -(-total_items // columns_per_row)  # Ceiling division

                for row_index in range(total_rows):
                    # Create a new row of columns
                    cols = st.columns(columns_per_row)
                    
                    # Iterate through the items in this row
                    for col_index in range(columns_per_row):
                        # Calculate the actual index in the dataframe
                        item_index = row_index * columns_per_row + col_index
                        
                        # Check if the index is valid
                        if item_index < total_items:
                            # Get the row data
                            row = high_scores_df.iloc[item_index]
                            skill_name = row['Domain/Area']
                            img = data[data['Skill'] == str(skill_name)]['Image'].values[0]
                            cols[col_index].write(skill_name)
                            cols[col_index].image(img, width=150)
 
            else:
                st.write("No Improvements Needed...")    
        else:
            st.markdown(
                """
                <h1 style='text-align: center; font-size: 50px;'>
                    <span style='color: red;'>The Candiate is not a good match for the Job.</span>
                </h1>
                """,
                unsafe_allow_html=True
            )
            # place horizontal line
            st.markdown(
                """
                <hr style='border: 1px solid green;'>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                """
                <p style='text-align: center; font-size: 30px; font-family: Garamond, sans-serif;'>
                    <span style='color: purple;'>Improvements</span>
                </p>
                """,
                unsafe_allow_html=True
            )
            #parse Job Description and Extract the skills
            JD = preprocess_text(JD)
            scored_df = calculate_scores(JD)
            high_scores_df = scored_df[scored_df['Score'] >= 1]
            if not high_scores_df.empty:
                #make a list of skills
                data = pd.read_csv('image.csv')
                # Number of columns per row
                columns_per_row = 3
                # Count the total number of rows needed
                total_items = len(high_scores_df)
                total_rows = -(-total_items // columns_per_row)  # Ceiling division

                for row_index in range(total_rows):
                    # Create a new row of columns
                    cols = st.columns(columns_per_row)
                    
                    # Iterate through the items in this row
                    for col_index in range(columns_per_row):
                        # Calculate the actual index in the dataframe
                        item_index = row_index * columns_per_row + col_index
                        
                        # Check if the index is valid
                        if item_index < total_items:
                            # Get the row data
                            row = high_scores_df.iloc[item_index]
                            skill_name = row['Domain/Area']
                            img = data[data['Skill'] == str(skill_name)]['Image'].values[0]
                            
                            # Display the skill name and image
                            cols[col_index].write(skill_name)
                            cols[col_index].image(img, width=150)
            else:
                st.write("No Improvements Needed...")
        
            
elif page == "Interview Process":
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: green;'>Recruitment Process</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    data=pd.read_csv('company.csv',encoding='latin1')
    company_names=data['Company'].tolist()
    #display the company names in alphabetical order
    company_names.sort()
    company_name = st.selectbox("Select Company Name",company_names)
    search_and_display_images(company_name+'comapany',3)
    # draw horizontal line
    st.markdown(
        """
        <hr style='border: 1px solid green;'>
        """,
        unsafe_allow_html=True
    )
    try:
        #About the company which is in Info column
        info=data[data['Company']==company_name]['Info'].values[0]
        st.write(info)
        round1 = data[data['Company'] == company_name]['Round 1'].values[0]
        round2 = data[data['Company'] == company_name]['Round 2'].values[0]
        round3 = data[data['Company'] == company_name]['Round 3'].values[0]
        round4 = data[data['Company'] == company_name]['Round 4'].values[0]
        round5 = data[data['Company'] == company_name]['Round 5'].values[0]
        round6 = data[data['Company'] == company_name]['Round 6'].values[0]
        # Helper function to style headings and text
        def style_round(round_heading, round_text):
            # Separate heading from details using ":"
            heading, details = round_text.split(":", 1)
            # Style the heading and details
            styled_heading = f"<span style='font-weight:bold;'>{heading}:</span>"
            styled_details = f"<span>{details.strip()}</span>"
            return f"<p style='color:red; font-weight:bold;'>{round_heading}</p><p>{styled_heading} {styled_details}</p>"

        # Display rounds with styled text
        st.markdown(style_round("Round 1", round1), unsafe_allow_html=True)
        st.markdown(style_round("Round 2", round2), unsafe_allow_html=True)
        st.markdown(style_round("Round 3", round3), unsafe_allow_html=True)
        st.markdown(style_round("Round 4", round4), unsafe_allow_html=True)
        st.markdown(style_round("Round 5", round5), unsafe_allow_html=True)
        st.markdown(style_round("Round 6", round6), unsafe_allow_html=True)
    except:
        pass
    
elif page == "Interview Preparation":
    st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        <span style='color: #a39407;'>Mock Interview Sessions</span>
    </h1>
    """,
    unsafe_allow_html=True
    )
    #place slider for Easy, Medium, Hard levels
    level = st.select_slider(
    "Choose the level of difficulty", 
    options=["Easy", "Medium", "Hard"]
    )
    data=pd.read_csv('Software Questions.csv',encoding='latin1')
    #filter the questions based on the level
    questions = data[data['Difficulty']==level]['Question'].tolist()
    answers = data[data['Difficulty']==level]['Answer'].tolist()
    category = data[data['Difficulty']==level]['Category'].tolist()
    # apply the select box to select the category
    category = list(set(category))
    category.sort()
    category = st.selectbox("Select Category",category)
    #filter the questions based on the category
    #apply cosine similarity to get the questions on data
    #apply tfidf vectorizer to convert the text data into numerical data
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(questions)
    #calculate the cosine similarity
    # Convert the sparse matrix to a dense matrix
    # Calculate the cosine similarity
    similarity=cosine_similarity(tfidf_matrix,tfidf_matrix) 
    # Get the index of the question
    question = random.randint(0, len(questions)-1)
    questions = data[(data['Difficulty']==level) & (data['Category']==category)]['Question'].tolist()
    answers = data[(data['Difficulty']==level) & (data['Category']==category)]['Answer'].tolist()
    #display the questions with answers
    for i in range(len(questions)):
        # Display the question in bold and red
        st.markdown(
            f"<b style='color: red;'>Q{i+1}.</b> <span style='color: red;'>{questions[i]}</span>",
            unsafe_allow_html=True
        )
        # Display the answer in bold and black
        st.markdown(
            f"<b>Ans:</b> {answers[i]}",
            unsafe_allow_html=True
        )
        # Add a horizontal green line
        st.markdown(
            """
            <hr style='border: 1px solid green;'>
            """,
            unsafe_allow_html=True
        )

    query = f"{category} {level} courses"
    videos = fetch_youtube_videos(query)

    # Display videos in rows of 3
    for i in range(0, len(videos), 2):
        cols = st.columns(2)  # Create 3 columns
        for j, video in enumerate(videos[i:i+2]):  # Iterate over videos for the current row
            with cols[j]:
                st.video(f"https://www.youtube.com/watch?v={video['video_id']}")