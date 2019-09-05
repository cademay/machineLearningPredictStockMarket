


from bs4 import BeautifulSoup
import urllib
import csv
import pandas as pd

#from urllib.request import urlopen



linkPrefix = 'https://finance.google.com/'

#r = urllib.urlopen('https://finance.google.com/finance/company_news?q=NASDAQ%3AAAPL&ei=-74OWsG6K4KHjAG7lK74Cw').read()
r = urllib.urlopen('https://finance.google.com/finance/company_news?q=NASDAQ%3ASBUX&ei=BNQOWoD6AcTejAGf24XwCA').read()


soup = BeautifulSoup(r, "lxml")
#print type(soup)

#articles = soup.find_all('div', class_='g-section news sfe-break-bottom-16')
linksdivs = []
linksdivs.extend(soup.find_all('span', class_='name'))

nextPage = soup.find_all('td', class_='nav_b')[0]


while True:

	nextPageLinkSuffix = nextPage.a['href']
	r = urllib.urlopen(linkPrefix + nextPageLinkSuffix).read()
	

	soup = BeautifulSoup(r, "lxml")
	linksdivs.extend(soup.find_all('span', class_='name'))
	next_prev = soup.find_all('td', class_='nav_b')
	if len(next_prev) < 2:
		break
	nextPage = next_prev[1]
	print nextPageLinkSuffix

#with open('applenews.csv', 'w') as toWrite:
	#writer = csv.writer


links = []
for div in linksdivs:
	links.append(div.a['href'])


path = 'starbucks.txt'
with open(path, 'w') as f:
	for link in links:
		f.write(link[1:])
		f.write('\n')	

"""
for link in links:
	print link
"""
