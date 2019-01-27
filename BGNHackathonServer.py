from flask import Flask , request
import requests
import json
import twitscrape

application = Flask(__name__)


@application.route('/')
def index():
    return 'Twitter Locational Interests'

@application.route('/<location>/<topic>' , methods=['GET', 'POST'])
def sentanalyse(location,topic):
    if (request.method == 'GET'):
        query = topic
        response = requests.get("https://api.datamuse.com/words?ml="+topic)
        result = json.dumps(twitscrape.fetch(topic))
        related_word = json.loads(response.text)
        x = []

        for r in related_word[0:5]:
            x.append(r['word'])
        y =  twitscrape.fetch(topic)

        y['related_words']  = x
        return  json.dumps(y)




if __name__ == '__main__':

    application.run(host='127.0.0.1', port=8080, debug=False)
