from html.parser import HTMLParser
from re import sub
from sys import stderr
from traceback import print_exc

import requests


class _DeHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.__text = []
        self._results = []

        self.apt_list = []
        self.em_line = ["\n", ""]

    def print_reslut(self):
        print(self._results)

    # 当我们调用feed函数时，他将整个HTML文档传递进来并在遇到每个标签时调用下面这个函数(根据函数名就容易理解)
    def handle_starttag(self, tag, attrs):

        target_attrs = [
            "fp-col-title",
            "fp-col-text",
            # "small-text",
            "primary-action",
            "secondary-action",
        ]

        keep_line = [
            "special-details",
        ]

        if tag == "h4":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value == "fp-name":
                    self.__text.append("\n🟢:")
            # self._results.append('\n\n\nMyavailableTarget: ')

        # elif tag == 'h6':
        #     self.__text.append('\n\n\nMyavailableTarget: ')
        #     tag_name = None
        #     tag_value = None
        #     for attrs_name, attrs_value in attrs:
        #         #input标签里面不是有name,value,type等属性吗，这里只判断name和value
        #         #不过我觉得第二个if是多余的
        #         if attrs_name == "class":
        #             tag_name = attrs_value
        #         if tag_name is not None:
        #             self.tag_results[tag_name] = attrs_value

        elif tag == "h6":
            self.__text.append(";")

        elif tag == "span":
            for attrs_name, attrs_value in attrs:

                if attrs_name == "class" and attrs_value == "accordion-title":
                    self.__text.append("\n")

                elif attrs_name == "class" and attrs_value in target_attrs:
                    self.__text.append(";")

        elif tag == "div":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value in target_attrs:
                    self.__text.append(";")

        elif tag == "a":

            for attrs_name, attrs_value in attrs:

                if attrs_name == "class" and attrs_value in target_attrs:
                    self.__text.append(";")

        # elif tag == 'strong':
        #     self.__text.append(';')

        # elif tag == 'p':
        #     self.__text.append('\n\n')

        elif tag == "p":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value not in keep_line:
                    self.__text.append("\n")

        # elif tag == 'br':
        #     self.__text.append('\n')

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.__text.append("\n\n")

    def text(self):
        return "".join(self.__text).strip()

    def handle_data(self, data):
        text = data.strip()
        if len(text) > 0:
            text = sub("[ \t\r\n]+", " ", text)
            self.__text.append(text + " ")

    # #重写handle_data方法
    # def handle_data(self, data):

    #     text = data.strip()
    #     if len(text) > 0:
    #         text = sub('/[ \t\r\n]+', ' ', text)
    #         # text = sub('^[^b\r\n]*b.*[\r\n]*', ' ', text)
    #         self.__text.append(text + ' ')
    #         # print(self.__text)

    #     data = data.strip()
    #     # if len(data) > 0:
    #     # data = sub('/[ \t\r\n]+', ' ', data)
    #         # text = sub('^[^b\r\n]*b.*[\r\n]*', ' ', text)
    #         # self.__text.append(text + ' ')
    #         # print(self.__text)
    #     if len(data) > 0:

    #         # if data != '':
    #         #     print("Data     :" + data)
    #         # if data.__contains__("/month"):
    #         #     print("Data     :" + data)
    #         # # if self.lasttag == 'span' and data.__contains__("/month"):
    #         #     print("Data     :" + data)
    #         # if self.lasttag == 'span' and data not in self.em_line:
    #         if self.lasttag == 'span':
    #             print("Data     :" + data)
    #             # print(len(data))
    #         # elif self.lasttag == 'h4' and data != "":
    #         #     print("Data     :" + data)
    #         #     # print data


def dehtml(text):
    try:
        parser = _DeHTMLParser()
        parser.feed(text)
        parser.close()
        # parser.print_reslut()
        return parser.text()
    except:
        print_exc(file=stderr)
        return text


def output(text):
    lines = text.split("\n")
    # print(lines)
    output = []
    for line in lines:
        # print(line)
        if "🟢:" in line:
            print(line)
            output.append(line)


def main():

    text = r"""
        <html>
            <body>
                <b>Project:</b> DeHTML<br>
                <b>Description</b>:<br>
                Cannot get correct content from the URL.
            </body>
        </html>
    """

    the_url = "https://www.livetalisman.com/redmond/talisman/conventional/"

    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.66 Safari/537.36"
    }

    r = requests.get(
        the_url,
        timeout=30,
        headers=header,
        # cookies=jar,
    )
    code = r.status_code
    # print(code)
    if code == 200:
        text = r.text
    else:
        print("error")

    # dehtml(text)
    text = dehtml(text)

    output(text)

    # print(text)


if __name__ == "__main__":
    main()
