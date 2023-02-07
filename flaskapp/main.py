from flask import Flask, jsonify
import lxml
import requests
import os
import regex as re
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hi():
    return 'Web Application with flask and docker!'


@app.route('/ping',methods=['GET','POST'])
def ping():
    return jsonify(success=True)

@app.route('/parse_news',methods=['GET','POST'])
def scrape():
    URL = "https://rusdisinfo.voxukraine.org"
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'lxml')
    result = soup.find_all('a')

    topic_hrefs_full = []
    for i in result:
        if re.search('/narratives/', i['href']) != None:
            topic_hrefs_full.append('https://rusdisinfo.voxukraine.org'+i['href'])

    news_links = []
    from selenium import webdriver
    path = r'chromedriver_win32\chromedriver.exe'
    driver = webdriver.Chrome(path)
    for link in topic_hrefs_full:
        driver.get(link)
        all_links = driver.find_elements('class name', 'Narrative_fakeLink___YbTe')
        for i in all_links:
            i.click()
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        list_dirty = soup.find_all('div', {'class': 'Narrative_media__B2gNZ'})
        for u in list_dirty:
            link_news = u.find('a').get('href')
            news_links.append(link_news)
    driver.close()

    title = ['<ul>']
    for i in news_links[:3]:
        try:
            soup = BeautifulSoup(requests.get(i).content, 'html.parser')
            title += [str(soup.title.text)]
            print("<li><a href=" + i + ">" + str(soup.title.text) + "</a></li>")
        except:
            pass
    title += ['</ul>']
    return title


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
