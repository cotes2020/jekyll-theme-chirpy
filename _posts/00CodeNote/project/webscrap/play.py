import argparse
import csv
import logging
import os
import sys
from datetime import date, datetime
from html.parser import HTMLParser
from re import sub
from sys import stderr
from traceback import print_exc

import requests

# Logs will go to CloudWatch log group corresponding to lambda,
# If Lambda has the necessary IAM permissions.
# Set logLevel to logging.INFO or logging.DEBUG for debugging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)
# Retrieve log level from Lambda Environment Variables
LOGGER.setLevel(level=os.environ.get("LOG_LEVEL", "INFO").upper())

APT_DIC = {
    "talisman": "https://www.livetalisman.com/redmond/talisman/conventional/",
}


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

        keep_line = [
            "special-details",
        ]

        avoid_attrs = [
            "small-abbr",
        ]

        target_attrs = [
            "fp-col-title",  # for Beds / Baths
            "fp-col-text",  # for Beds / Baths info
            # "small-text",
            "primary-action",
            "secondary-action",
        ]

        # target_attrs = [
        #     "fp-col bed-bath",  # for Beds / Baths
        #     "fp-col rent",
        #     "fp-col deposit",
        #     "fp-col sq-feet",
        #     "fp-col special",
        #     "fp-col action",
        #     "fp-col-title",  # for Beds / Baths
        #     "fp-col-text",   # for Beds / Baths info
        # ]

        # if tag == "meta":
        #     for attrs_name, attrs_value in attrs:
        #         if attrs_name == "content":
        #             info = attrs_value
        #         if attrs_name == "numberOfBedrooms":
        #             self.__text.append(info)
        #         if attrs_name == "numberOfBathroomsTotal":
        #             self.__text.append(":").append(info).append(";")

        if tag == "h3":
            for attrs_name, attrs_value in attrs:
                if (
                    attrs_name == "class"
                    and attrs_value == "fp-group-header accordion-trigger"
                ):
                    # for title
                    self.__text.append("\n")

        if tag == "h4":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value == "fp-name":
                    # for floor plan
                    self.__text.append("\n🟢:")
                    # self._results.append('\n\n\nMyavailableTarget: ')

        if tag == "h6":
            self.__text.append(";")

        if tag == "span":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value in target_attrs:
                    # and attrs_value not in avoid_attrs:
                    # for Beds / Baths
                    # for Beds / Baths info
                    self.__text.append(";")

        if tag == "div":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value in target_attrs:
                    self.__text.append(";")

        if tag == "p":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value not in keep_line:
                    self.__text.append("\n")

    def text(self):
        return "".join(self.__text).strip()

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self.__text.append("\n\n")

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
    #            print("Data     :" + data)
    #            # print(len(data))
    #         # elif self.lasttag == 'h4' and data != "":
    #         #     print("Data     :" + data)
    #         #     # print data


def dehtml_talisman(text):
    try:
        parser = _DeHTMLParser()
        parser.feed(text)
        parser.close()
        # parser.print_reslut()
        return parser.text()
    except:
        print_exc(file=stderr)
        return text


def get_html(url):
    text = r"""
        <html>
          <body>
              <b>Project:</b> DeHTML<br>
              <b>Description</b>:<br>
              Cannot get correct content from the URL.
          </body>
        </html>
    """
    header = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.66 Safari/537.36"
    }

    r = requests.get(
        url,
        timeout=30,
        headers=header,
        # cookies=jar,
    )
    code = r.status_code
    if code == 200:
        LOGGER.info("======= get info from %s =======" % url)
        html_text = r.text
    else:
        LOGGER.info("======= Error: can not get info from %s =======" % url)
        # os.Exit(1)
    return html_text


def output(text):
    lines = text.split("\n")
    # print(lines)
    output = []
    for line in lines:
        # print(line)
        if "🟢:" in line:
            # print(line)
            LOGGER.info(line)
            output.append(line)
    return output


def create_dic(apt, text):
    lines = text.split("\n")
    output = []
    for line in lines:
        if "🟢:" in line:
            info = line
            info = info.replace("🟢:", "")
            info = info.replace(" ;Beds / Baths ;", ";")
            info = info.replace(" ;Rent ;", ";")
            info = info.replace(" ;Deposit ;", ";")
            info = info.replace(" ;Sq. Ft ;", ";")
            info = info.replace(" ;Limited Time Offer: Valid Through : ", ";")
            info = info.replace("! ", ";")
            # LOGGER.info(info)
            info = info.split(";")
            # LOGGER.info(info)
            # [
            #     '🟢:Urban with Kitchen Bar',
            #     '1 bd / 1 ba',
            #     '$1,921/month',
            #     '$300',
            #     '605',
            #     'Oct 27, 2022 - Jan 31, 2023 8 Weeks Free on Select Homes and Move-In Dates',
            #     '4 Available Details '
            # ]
            dic = {}
            dic["Apt"] = apt
            dic["Floor_plan"] = info[0]
            dic["Beds_Baths"] = info[1]
            dic["Rent"] = info[2]
            dic["Deposit"] = info[3]
            dic["Sq_Ft"] = info[4]
            dic["Limited_Time_Offer"] = info[5]
            dic["Available"] = info[6]
            output.append(dic)
    # for i in output:
    #     print(i)
    return output


def create_csv(dic_list):
    """
    Create the csv snapshot from APTSCRAPPER
    :param dic: info from the date
    :return: a csv contains the snapshot of the info from each web
    """
    # to set the target_date in the csv filename
    target_date = date.today().strftime("%Y/%m/%d")  # "2022/06/01"
    filedate = target_date.replace("/", "")
    file_name = f"apt_{filedate}.csv"
    LOGGER.info("======= creating file %s =======" % file_name)

    header = [
        "Apt",
        "Floor_plan",
        "Beds_Baths",
        "Rent",
        "Deposit",
        "Sq_Ft",
        "Limited_Time_Offer",
        "Available",
    ]
    with open(f"./apt_output/{file_name}", "w") as f:
        LOGGER.info("======= created file %s =======" % file_name)
        # create the csv writer
        writer = csv.writer(f)
        # write a header to the csv file
        writer.writerow(header)
        # write a row to the csv file
        for input_dic in dic_list:
            LOGGER.info(input_dic)
            # print(type(input_dic))
            # for info in input_dic.values():
            writer.writerow(input_dic.values())
    LOGGER.info("======= info loaded in the file %s =======\n" % file_name)


def run(target_apt):

    target_url = APT_DIC[target_apt]

    if target_apt == "talisman":
        html_text = get_html(target_url)
        text = dehtml_talisman(html_text)
        output(text)
        dic_list = create_dic(target_apt, text)
        print(type(dic_list))
        create_csv(dic_list)

    else:
        LOGGER.info("======= Error: invalid target: %s =======", target_apt)


def main(target_Apts):
    for apt in target_Apts.keys():
        LOGGER.info("======= Target Apartment: %s =======", apt)
        run(apt)


if __name__ == "__main__":
    # Simple commandline parser to accept inputs for the script
    # all website
    # one website
    parser = argparse.ArgumentParser(description="todo.")
    parser.add_argument(
        "-t",
        "--target",
        required=True,
        help="Whether run locally or on centralized account",
    )
    # Assign the inputs:
    args = parser.parse_args()
    target = args.target.lower()

    # """ Main method for app. """
    timestamp = datetime.now().strftime("%c")
    LOGGER.info("======= Apt_scrapper loaded at %s" % timestamp)

    if target == "all":
        target_Apts = APT_DIC
        LOGGER.info(
            "============ Apt_scrapper run for url: %s ============\n" % target_Apts
        )
        main(target_Apts)
    else:
        LOGGER.info("Invalid --target\n")
        # os.Exit(1)
