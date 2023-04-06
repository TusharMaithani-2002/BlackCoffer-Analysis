import pandas as pd
from nltk.tokenize import word_tokenize,sent_tokenize
import re
from nltk.stem import PorterStemmer
import os
from bs4 import BeautifulSoup
import requests


# scrapping data
# laoding input file
input = pd.read_csv('input.csv')


urls = input['URL']

# for storing scrapped files
os.makedirs('scrapped files')
# looping over all the urls and scrapping their text
print('File scrapping started...')
for url in urls:
    url_id = int(input[input['URL'] == url]['URL_ID'])
    page_content = requests.get(url)
    if page_content.status_code == 404:
        continue
    page  = page_content.text.encode(encoding='utf-8')
    result = BeautifulSoup(page,'html.parser')
    heading = result.find_all('h1')
    title = heading[0].text if len(heading) > 0 else ''
    article_content = result.body.find_all('p',class_ = "")
    scrapped = []

    for info in article_content:
        scrapped.append(info)
    
    # appending any list items if present
    li_list = result.body.find_all('li',class_="")
    
    for li in li_list:
        scrapped.append(li)

    # creating text file
    try:
        with open("scrapped files/{}.txt".format(url_id),'a+') as file:
            file.write(str(title)+"\n")
            for line in scrapped:
                file.write(str(line.text))
        
            file.close()
    except:
        print('{} couldnot be scrapped'.format(url))
        continue    

print('Scrapping Completed')

print('Starting Analysis...')

# for stemming words used in counting syllables
porter = PorterStemmer()

# reading output file
given_out_df = pd.read_csv('output.csv')

# set for storing all the stopwords
stopwords = {None}

# creating stopwords given in the assigment

stopword_files = os.listdir('StopWords')

for file in stopword_files:
    with open("StopWords/"+file) as f:
        out = f.readlines()
        f.close()
    
    for word in out:
        stopwords.add(word[:-1])

# creating dictionaries
positive_dict = {None}
negative_dict = {None}

# creating positive dictionary
with open('MasterDictionary/positive-words.txt') as file:
    output = file.readlines()
    file.close()

for out in output:
    positive_dict.add(out[:-1])

# creating negative dictionary
with open('MasterDictionary/negative-words.txt') as file:
    output = file.readlines()
    file.close()

for out in output:
    negative_dict.add(out[:-1])




# creating output dataframe
columns = given_out_df.columns
index = [i for i in range(37,151)]
ouput_df = pd.DataFrame(columns=columns,index=index)



for file_num in range(37,151):
    try:
        with open("scrapped files/{}.txt".format(file_num)) as file:
            data = file.readlines()
            file.close()
    except:
        continue

    # converting data into string
    data = " ".join(data)
    sentences = sent_tokenize(data)
    token = word_tokenize(data)

    no_stopwords = []
    for word in token[:len(token) - 12]:
        if word not in stopwords:
            no_stopwords.append(word)
            
    # we have to remove punctuations if any from token
    no_stopwords = [word.lower() for word in no_stopwords if word.isalpha()]
    
    negative_score = 0
    positive_score = 0

    for word in no_stopwords:
        if word in positive_dict:
            positive_score += 1
        if word in negative_dict:
            negative_score += 1
    
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(no_stopwords) + 0.000001)
    
    average_sentence_length = len(token)/len(sentences)

    vowels = {'a','i','o','u','e'}
    complex_words = 0

    for word in no_stopwords:
        no_vowels = 0
        for char in word:
            if char in vowels:
                no_vowels += 1
        if no_vowels >= 2:
            complex_words += 1
    
    percentage_of_complex_words = complex_words / len(token)
    fog_index = 0.4 * (average_sentence_length + complex_words)

    average_number_of_words = len(token)/len(sentences)

    char_count = 0 # total number of charaters
    for word in token:
        char_count += len(word)
    
    average_word_length = char_count / len(token)

    # couting number of personal pronouns
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(data)
    
    syllables = 0

    for word in token:
        stemed = porter.stem(word)
        no_vowels = 0
        for char in stemed:
            if char in vowels:
                no_vowels += 1
        if no_vowels > 2:
             syllables += 1

    url = given_out_df[given_out_df['URL_ID'] == file_num]['URL']
    ouput_df.loc[file_num] = pd.Series({
        'URL_ID':file_num,
        'URL':url,
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH' : average_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS' : percentage_of_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': average_number_of_words,
        'COMPLEX WORD COUNT': complex_words,
        'WORD COUNT' : len(no_stopwords),
        'SYLLABLE PER WORD' : syllables,
        'PERSONAL PRONOUNS': len(pronouns),
        'AVG WORD LENGTH': average_word_length
    })

ouput_df.to_csv('myoutput.csv')

print('Analysis done. myoutput.csv contains result')