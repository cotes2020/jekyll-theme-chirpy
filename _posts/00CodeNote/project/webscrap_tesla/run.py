import logging
import os
import sys
from celery import shared_task
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from django.core.mail import send_mail

from django.conf import settings
import datetime


# Logs will go to CloudWatch log group corresponding to lambda,
# If Lambda has the necessary IAM permissions.
# Set logLevel to logging.INFO or logging.DEBUG for debugging.
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)
# Retrieve log level from Lambda Environment Variables
LOGGER.setLevel(level=os.environ.get("LOG_LEVEL", "INFO").upper())

TESLA_URL = "https://www.tesla.com/inventory/new/my?TRIM=LRAWD&arrangeby=plh&zip=98011&range=0"

OUTPUTDIR = "./_posts/00CodeNote/project/webscrap_tesla/output"

def create_text_file(content_list):
    try:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{current_datetime}_tesla-list.txt"
        with open(f"{OUTPUTDIR}/{file_name}", 'w') as file:
            for item in content_list:
                file.write(item + '\n')  # Write each item followed by a newline character

        LOGGER.info(f"File '{file_name}' created successfully with content: {content_list}")
    except Exception as e:
        LOGGER.info(f"Error occurred: {str(e)}")
 

def watch_tesla():
    
    LOGGER.info("======= watch_tesla =======")
        
    # test using Chrome Selenium
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-gpu')
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.50 Safari/537.36'
    options.add_argument('user-agent={0}'.format(user_agent))

    driver = webdriver.Chrome(options=options)
    
    driver.set_page_load_timeout(90)

    # Load the URL and get the page source
    driver.get(TESLA_URL)
    LOGGER.info("======= loaded TESLA_URL =======")

    car_prices = []
    output_prices = []

    try:
        LOGGER.info("======= Start the search =======")
        
        results_container = WebDriverWait(driver, 100).until(
            EC.presence_of_element_located((By.CLASS_NAME, "results-container"))
        )
        
        print(results_container)

        car_sections = results_container.find_elements(By.CLASS_NAME, "result-header")

        # Now you can iterate over these article elements and perform any actions you desire.
        
        for car_section in car_sections:

            car_price_str = car_section.find_element(By.CLASS_NAME, "result-purchase-price").get_attribute("innerHTML")
            car_price = int(car_price_str.replace('$', '').replace(',', ''))
            car_prices.append(car_price)
            
            LOGGER.info("======= %s =======" % car_price) 
            
            if car_price < 50000:
                
                email_content = f'There is a model Y for sale for {car_price_str}' 
                LOGGER.info("======= %s =======" % email_content) 
                # send_mail(
                #     'Model Y for sale gucci price',  # Subject of the email
                #     f'There is a model Y for sale for {car_price_str}!',  # Message body
                #     settings.EMAIL_HOST_USER,  # From email address (sender)
                #     ['chriskuis@hotmail.com'],  # List of recipient email addresses
                #     fail_silently=False,  # Set to True to suppress exceptions if sending fails
                # )
                output_prices.append(email_content)
                
        create_text_file(output_prices)
        
    finally:
        driver.quit()

    return car_prices


if __name__ == "__main__":
    LOGGER.info("======= Start the watch =======")
    watch_tesla()
