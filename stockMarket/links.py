# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import urllib

def pulllinks(mainpage, filepath):
    r = urllib.urlopen(mainpage).read()
    soup = BeautifulSoup(r, "lxml")
    linkdivs = soup.find_all('span', class_='name')
    links = []

    for div in linkdivs:
      links.append(div.a['href'])

    with open(filepath, 'w') as f:
      for link in links:
          f.write(link + '\n')
    print len(links)

mainpage = ['https://finance.google.com/finance/company_news?q=NASDAQ%3AAAPL&ei=MyoPWoDHEoH2jAHvqKyoDw&start=0&num=250', 'https://finance.google.com/finance/company_news?q=NASDAQ%3AGOOGL&ei=NS0PWunfNofSjAHDvqqQDA&start=0&num=214', 'https://finance.google.com/finance/company_news?q=NYSE%3AF&ei=bC0PWoHxBI_E2AbljKKACg&start=0&num=206', 'https://finance.google.com/finance/company_news?q=NASDAQ%3ASBUX&ei=my0PWsGyBIPwjAHk-6vQCQ&start=0&num=104']
filepath = ['applelinks.txt', 'googlelinks.txt', 'fordlinks.txt', 'starbuckslinks.txt']
for i in range(4):
    pulllinks(mainpage[i], filepath[i])