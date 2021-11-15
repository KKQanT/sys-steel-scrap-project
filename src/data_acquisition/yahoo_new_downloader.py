from selenium.webdriver import Chrome, ChromeOptions
from selenium.common.exceptions import NoSuchElementException
import os
import time

def initiate(save_path):
    options = ChromeOptions()
    options.add_argument("--start-maximized")

    prefs = {
    "download.default_directory": save_path,
    "download.prompt_for_download": False,
    "download.directory_upgrade": True
    }

    options.add_experimental_option('prefs', prefs)
    driver = Chrome(options=options)
    driver.get('https://finance.yahoo.com/screener?.tsrc=fin-srch')
    
    return driver

if __name__ == "__main__":

    SAVE_PATH = os.path.realpath('../../data/yahoo/')
    DELAY_TIME = 2
    TRIAL_LIMIT = 30

    STOCKS_NAME = [
    'X', 
    'TSM', 
    'SCHN', 
    'MT', 
    'AH.BK', 
    '000333.SZ',
    '601899.SS',
    '600019.SS',
    '000932.SZ',
    '^TWII'
    ]

    exist_files = os.listdir(os.path.realpath('../../data/yahoo/'))

    for file in exist_files:
        path_todel = os.path.join(os.path.realpath('../../data/yahoo/'), file)
        os.remove(path_todel)
    
    driver = initiate(SAVE_PATH)

    driver.get('https://finance.yahoo.com/')

    time.sleep(DELAY_TIME)

    n_trial = 0

    for stock_name in STOCKS_NAME:
        while True:
            try:
                input_ = driver.find_element_by_xpath('//*[@id="yfin-usr-qry"]')
                input_.send_keys(stock_name)
                time.sleep(DELAY_TIME)
                button = driver.find_element_by_xpath('//*[@id="header-desktop-search-button"]')
                button.click()

                for li_ in range(1,6):
                    historical = driver.find_element_by_xpath(f'//*[@id="quote-nav"]/ul/li[{li_}]/a/span')
                    if historical.text == 'Historical Data':
                        historical.click()
                        break

                time.sleep(DELAY_TIME)

                period = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[1]/div[1]/div/div/div/span')
                period.click()

                time.sleep(DELAY_TIME+3)

                max_button = driver.find_element_by_xpath('//*[@id="dropdown-menu"]/div/ul[2]/li[4]/button')
                max_button.click()

                time.sleep(DELAY_TIME+3)

                period_validate = period.text

                download = driver.find_element_by_xpath('//*[@id="Col1-1-HistoricalDataTable-Proxy"]/section/div[1]/div[2]/span[2]/a/span')
                download.click()
                break
            except NoSuchElementException:
                print(f'error at {stock_name}')
                driver.close()
                driver = initiate(SAVE_PATH)
                time.sleep(DELAY_TIME)
                n_trial+=1
                print(n_trial)
                if n_trial > TRIAL_LIMIT:
                    break

        if n_trial > TRIAL_LIMIT:
            break
    
    time.sleep(5)

    driver.close()

