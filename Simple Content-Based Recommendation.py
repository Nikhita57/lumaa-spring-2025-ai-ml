#!/usr/bin/env python
# coding: utf-8

# In[38]:


# Import so that TF-IDF classification can be performed later
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Initialize lists for titles and summaries
titles = []
summaries = []
# Open file and start reading
f = open("BookList", "r", encoding="utf-8")
f.readline()
f.readline()
currLine = " "
# Once at the first title, loop through file to get all titles, authors, and book summaries
while currLine != "":
    currLine = f.readline()
    # Remove newlines from titles
    currLine = currLine[0:len(currLine)-1]
    titles.append(currLine)
    f.readline()
    currLine = f.readline()
    # Get full summaries
    summary = currLine
    while currLine != "" and currLine != "\n":
        currLine = f.readline()
        summary = summary + currLine
    summaries.append(summary)
# Query user for novel preferences
print("What kinds of novels do you want?")
preferences = input()
# Add preferences to the array so TF-IDF classification includes it
summaries.append(preferences)
# Perform TF-IDF classification
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(summaries)
# Loop to get the top three most similar novels by calculating the cosine similarity of the user input with each novel summary
topThree = ["N/A", "N/A", "N/A"]
topThreeValues = [-0.1, -0.1, -0.1]
for i in range(len(summaries)-1):
    s = cosine_similarity(X[i], X[100])[0][0]
    # Recalculate top three if a more similar novel summary is found
    if s > topThreeValues[2]:
        temp = topThree[0]
        temp2 = topThree[1]
        tempVal = topThreeValues[0]
        tempVal2 = topThreeValues[1]
        if s <= topThreeValues[1]:
            topThreeValues[2] = s
            topThree[2] = titles[i]
        elif s <= topThreeValues[0]:
            topThreeValues[1] = s
            topThree[1] = titles[i]
            topThreeValues[2] = tempVal2
            topThree[2] = temp2
        else:
            topThreeValues[0] = s
            topThree[0] = titles[i]
            topThreeValues[1] = tempVal
            topThree[1] = temp
            topThreeValues[2] = tempVal2
            topThree[2] = temp2

# Print recommendations            
print(topThree)
# Close file
f.close()


# In[ ]:





# In[ ]:




