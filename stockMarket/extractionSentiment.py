import csv
from goose import Goose
import pandas as pd
from bs4 import BeautifulSoup
import urllib



def createDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)


goose = Goose()

#createDirectory

def parseReutersLink(link):
    print link
    wwwIndex = link.find('www.reuters')
    
    slice1 = link[wwwIndex:]
    
    print slice1
    
    
    slice1 = slice1.replace('%2F', '/')
        
    andIndex = slice1.find('&e')
    
    slice2 = slice1[0:andIndex]
    print slice2
    print ""
    
    return 'https://' + slice2
    

articles = []

with open('googlelinks.txt', 'r') as f:    
    
    i = 0
    for link in f:
        i+=1
        
        print i
        
        
        if link[0:2] == '//':
            link = link[2:]
            print link
            
            
        if "%2Fwww.reuters.com" in link and 'google.com' in link:
            link = parseReutersLink(link)
        
            r = urllib.urlopen(link).read()
            soup = BeautifulSoup(r, 'lxml')
            #soup = BeautifulSoup(r, 'html.parser')
            
                        
            date = soup.find('span', class_='timestamp')
            text = soup.find('div', class_='columnLeft')
            
            if date is not None and text is not None:
                
                date = date.get_text().encode('utf-8')
                text = text.find('p')
            
            
            
                #text = soup.find('div', id_='sigDevArticleText', class_='gridPanel grid6').a['p']
            
                text = text.get_text().encode('utf-8')
            
                print text
            
                articles.append([date, text])
            
        else:
            #print link
            #print link
            article = goose.extract(url=link) 
            
            date = article.publish_date
            text = article.cleaned_text.encode('utf-8')
            
            if date is not None and text is not None:
                articles.append([date, text])

print len(articles)


print articles[:][0]
print len(articles[:][0])
with open('GOOGLarticles.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(articles)
    
    #for article in articles:



    #    f.write(article)    

    
    


    


#article = goose.extract(url="https://www.smallsurething.com/web-scraping-article-extraction-and-sentiment-analysis-with-scrapy-goose-and-textblob/")
#print article.cleaned_text

