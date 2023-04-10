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

# APT Object
class _DeHTMLParser_General(HTMLParser):
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

        target_attrs_div = [
            "fp-col-title",  # for Beds / Baths
            "fp-col-text",  # for Beds / Baths info
        ]
        target_attrs_a = [
            # "small-text",
            "primary-action",
            "secondary-action",
        ]

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
                if attrs_name == "class" and attrs_value in target_attrs_div:
                    # and attrs_value not in avoid_attrs:
                    # for Beds / Baths
                    # for Beds / Baths info
                    self.__text.append(";")

        if tag == "div":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value in target_attrs_div:
                    self.__text.append(";")
                if attrs_name == "class" and attrs_value == "page-description":
                    self.__text.append("\n")

        if tag == "a":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value in target_attrs_a:
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


class _DeHTMLParser_General_Talisman(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.__text = []
        self._results = []
        self.apt_list = []
        self.em_line = ["\n", ""]

    def handle_starttag(self, tag, attrs):
        # for floorplan
        if tag == "h2":
            for attrs_name, attrs_value in attrs:
                if (
                    attrs_name == "class"
                    and attrs_value
                    == "floorplan-listing-b__title color__secondary--text"
                ):
                    self.__text.append("\n🟢:")
        # for Beds / Baths info
        if tag == "p":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value == "floorplan-listing-b__info":
                    self.__text.append(";")
        # for rent
        if tag == "div":
            for attrs_name, attrs_value in attrs:
                if (
                    attrs_name == "class"
                    and attrs_value == "floorplan-listing-b__info-row"
                ):
                    self.__text.append(";")

        if tag == "div":
            for attrs_name, attrs_value in attrs:
                if attrs_name == "class" and attrs_value == "page__disclaimer":
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

    def print_reslut(self):
        print(self._results)


OUTPUTDIR = "./_posts/00CodeNote/project/webscrap_apt/apt_output"

URL_DIC = {
    # "talisman": "https://www.livetalisman.com/redmond/talisman/conventional/",
    "talisman": "https://livetalisman.com/floorplans/",
    "modera": "https://www.moderaredmond.com/redmond/modera-redmond/conventional/",
}
CLASS_DIC = {
    "talisman": _DeHTMLParser_General_Talisman(),
    "modera": _DeHTMLParser_General(),
}

# Method_div
def get_html(url):
    """
    get plan html
    """
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
        LOGGER.info("======= Load info from %s =======" % url)
        html_text = r.text
    else:
        LOGGER.info("======= Error: can not get info from %s =======" % url)
        # os.Exit(1)
    return html_text


def dehtml(target_apt, text):
    """
    switch html into plan text
    """
    try:
        parser = CLASS_DIC[target_apt]
        parser.feed(text)
        parser.close()
        # parser.print_reslut()
        return parser.text()
    except:
        print_exc(file=stderr)
        return text


def output_content(text):
    """
    set marker in text
    """
    lines = text.split("\n")
    output = []
    for line in lines:
        if "🟢:" in line:
            line = (
                line.replace(" ;", ";")
                .replace(". ", ".")
                .replace(" / ", "/")
                .replace(" +", "+")
            )
            # LOGGER.info(line)
            output.append(line)
    return output


def create_dic(apt, lines, output_list):
    """
    [
        '🟢:Urban with Kitchen Bar',
        '1 bd / 1 ba',
        '$1,921/month',
        '$300',
        '605',
        'Oct 27, 2022 - Jan 31, 2023 8 Weeks Free on Select Homes and Move-In Dates',
        '4 Available Details '
    ]
    create dic from above info
    """
    inputdate = date.today().strftime("%Y%m%d")
    for line in lines:
        if "🟢:" in line:
            line = line.replace("🟢:", "")
            info_list = line.split(";")
            # LOGGER.info(info_list)

            dic = {}
            dic["Date"] = inputdate
            dic["Apt"] = apt
            dic["Floor_plan"] = info_list[0]

            # for talisman
            if apt == "talisman":
                dic["Beds/Baths"] = info_list[1][: (info_list[1].index("Bath") + 4)]

                dic["Rent"] = info_list[2].replace(" ", "")
                if dic["Rent"] != "ContactUs":
                    dic["Rent"] = (
                        dic["Rent"].split("-")[0].replace("$", "").replace(",", "")
                    )
                    LOGGER.info("!!!!!!!!!")
                    if dic["Rent"].isdigit():
                        dic["Rent"] = int(dic["Rent"])
                        LOGGER.info(dic["Rent"])

                dic["Deposit"] = "$300"
                dic["Sq.Ft"] = info_list[1].split("Bath ")[1]
                dic["Limited_Time_Offer"] = "None"
                dic["Available"] = "N/A"

            # for modera
            if apt == "modera":
                if "Beds/Baths" in line:
                    dic["Beds/Baths"] = info_list[2]
                if "Rent" in line:
                    dic["Rent"] = info_list[4]
                if "Deposit" in line:
                    dic["Deposit"] = info_list[6]
                if "Sq.Ft" in line:
                    dic["Sq.Ft"] = info_list[8]
                if "Limited Time Offer" in line:
                    dic["Limited_Time_Offer"] = info_list[9]
                    dic["Available"] = info_list[10]
                else:
                    dic["Limited_Time_Offer"] = "/"
                    dic["Available"] = info_list[9]
            output_list.append(dic)
    # for i in output_list:
    #     print(i)
    return output_list


def create_csv(all_dic_list):
    """
    Create the csv snapshot from APTSCRAPPER
    :param dic: info from the date
    :return: a csv contains the snapshot of the info from each web
    """
    # to set the target_date in the csv filename
    target_date = date.today().strftime("%Y/%m/%d")  # "2022/06/01"
    filedate = target_date.replace("/", "")
    file_name = f"apt_{filedate}.csv"
    LOGGER.info("\n======= creating file: %s =======" % file_name)

    header = [
        "Date",
        "Apt",
        "Floor_plan",
        "Beds/Baths",
        "Rent",
        "Deposit",
        "Sq.Ft",
        "Limited_Time_Offer",
        "Available",
    ]
    with open(f"{OUTPUTDIR}/{file_name}", "w") as f:
        LOGGER.info("======= filing file: %s =======" % file_name)
        # create the csv writer
        writer = csv.writer(f)
        # write a header to the csv file
        writer.writerow(header)
        # write a row to the csv file
        for input_dic in all_dic_list:
            LOGGER.info(input_dic)
            # print(type(input_dic))
            # for info in input_dic.values():
            writer.writerow(input_dic.values())
    LOGGER.info("======= info loaded in the file %s =======\n" % file_name)


def run(apt, all_dic_list):
    if apt in URL_DIC.keys():
        html_text = get_html(URL_DIC[apt])
        # print(html_text)
        text = dehtml(apt, html_text)
        lines = output_content(text)
        LOGGER.info("======= Got info for Apartment: %s =======", apt)
        all_dic_list = create_dic(apt, lines, all_dic_list)
        # print(all_dic_list)
        LOGGER.info("======= Got Dic for Apartment: %s =======\n", apt)
        return all_dic_list
    else:
        LOGGER.info("======= Error: invalid target: %s =======", apt)
        return []


def main(apt):
    all_dic_list = []
    for apt in URL_DIC.keys():
        LOGGER.info("======= Target Apartment: %s =======", apt)
        all_dic_list = run(apt, all_dic_list)
    create_csv(all_dic_list)


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
        target_Apts = URL_DIC
        LOGGER.info(
            "============ Apt_scrapper run for url: %s ============\n" % target_Apts
        )
        main(target_Apts)
    else:
        LOGGER.info("Invalid --target\n")
        # os.Exit(1)
